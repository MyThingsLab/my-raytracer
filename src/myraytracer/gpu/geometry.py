from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from myraytracer.gpu.vec import cross, dot, normalize

_PARALLEL_EPS = 1e-12


@dataclass(frozen=True)
class Sphere:
    center: torch.Tensor  # (3,)
    radius: torch.Tensor  # scalar


@dataclass(frozen=True)
class Plane:
    point: torch.Tensor  # (3,)
    normal: torch.Tensor  # (3,)


@dataclass(frozen=True)
class Mesh:
    """A triangle mesh, as a batch of triangle-vertex positions.

    `vertices` is (M, 3, 3): triangle index, vertex-in-triangle (v0/v1/v2),
    xyz. Typically produced by `myraytracer.gpu.mesh.load_obj`.
    """

    vertices: torch.Tensor  # (M, 3, 3)


@dataclass(frozen=True)
class HitBatch:
    hit: torch.Tensor  # (N,) bool
    t: torch.Tensor  # (N,)
    point: torch.Tensor  # (N, 3)
    normal: torch.Tensor  # (N, 3)


def _face_forward(normal: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    flip = (dot(normal, direction) > 0.0).unsqueeze(-1)
    return torch.where(flip, -normal, normal)


def sphere_hit(
    sphere: Sphere,
    ray_origin: torch.Tensor,
    ray_dir: torch.Tensor,
    t_min: float,
    t_max: float,
) -> HitBatch:
    oc = ray_origin - sphere.center
    a = dot(ray_dir, ray_dir)
    half_b = dot(oc, ray_dir)
    c = dot(oc, oc) - sphere.radius * sphere.radius
    discriminant = half_b * half_b - a * c

    # Clamp before sqrt so rows with no real root still produce a finite
    # (if meaningless) gradient path instead of NaN poisoning the batch.
    sqrt_discriminant = torch.sqrt(torch.clamp(discriminant, min=0.0))

    root_near = (-half_b - sqrt_discriminant) / a
    root_far = (-half_b + sqrt_discriminant) / a
    near_in_range = (root_near > t_min) & (root_near < t_max)
    far_in_range = (root_far > t_min) & (root_far < t_max)

    t = torch.where(near_in_range, root_near, root_far)
    hit = (discriminant >= 0.0) & (near_in_range | far_in_range)

    point = ray_origin + ray_dir * t.unsqueeze(-1)
    outward_normal = (point - sphere.center) / sphere.radius
    normal = _face_forward(outward_normal, ray_dir)

    return HitBatch(hit=hit, t=t, point=point, normal=normal)


def plane_hit(
    plane: Plane,
    ray_origin: torch.Tensor,
    ray_dir: torch.Tensor,
    t_min: float,
    t_max: float,
) -> HitBatch:
    denom = dot(plane.normal, ray_dir)
    denom_safe = torch.where(denom.abs() < _PARALLEL_EPS, torch.ones_like(denom), denom)

    t = dot(plane.point - ray_origin, plane.normal) / denom_safe
    in_range = (t > t_min) & (t < t_max)
    hit = in_range & (denom.abs() >= _PARALLEL_EPS)

    point = ray_origin + ray_dir * t.unsqueeze(-1)
    normal = _face_forward(plane.normal.expand_as(ray_dir), ray_dir)

    return HitBatch(hit=hit, t=t, point=point, normal=normal)


def triangle_hit(
    v0: torch.Tensor,
    v1: torch.Tensor,
    v2: torch.Tensor,
    ray_origin: torch.Tensor,
    ray_dir: torch.Tensor,
    t_min: float,
    t_max: float,
) -> HitBatch:
    """Batched Moller-Trumbore intersection of N rays against M triangles.

    `v0`/`v1`/`v2` are (M, 3) triangle corners; `ray_origin`/`ray_dir` are
    (N, 3). Broadcasts every ray against every triangle as an (N, M)
    intermediate, then reduces over the triangle axis to the nearest hit
    per ray, so the result is shaped (N,) / (N, 3) like `sphere_hit`/
    `plane_hit`.
    """
    edge1 = v1 - v0  # (M, 3)
    edge2 = v2 - v0  # (M, 3)

    ray_dir_b = ray_dir.unsqueeze(1)  # (N, 1, 3)
    ray_origin_b = ray_origin.unsqueeze(1)  # (N, 1, 3)
    edge1_b = edge1.unsqueeze(0)  # (1, M, 3)
    edge2_b = edge2.unsqueeze(0)  # (1, M, 3)

    h = cross(ray_dir_b, edge2_b)  # (N, M, 3)
    a = dot(edge1_b, h)  # (N, M)
    not_parallel = a.abs() >= _PARALLEL_EPS
    a_safe = torch.where(not_parallel, a, torch.ones_like(a))
    f = 1.0 / a_safe

    s = ray_origin_b - v0.unsqueeze(0)  # (N, M, 3)
    u = f * dot(s, h)

    q = cross(s, edge1_b)  # (N, M, 3)
    v = f * dot(ray_dir_b, q)
    t = f * dot(edge2_b, q)

    in_range = (t > t_min) & (t < t_max)
    barycentric_ok = (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0)
    valid = not_parallel & barycentric_ok & in_range

    # Nearest valid hit per ray: mask invalid (ray, triangle) pairs out to
    # +inf before taking the row-wise min, so gradients only flow through
    # the triangle that owns each ray's nearest hit.
    t_candidates = torch.where(valid, t, torch.full_like(t, math.inf))
    best_t, best_index = t_candidates.min(dim=1)  # (N,), (N,)
    hit = torch.isfinite(best_t)

    point = ray_origin + ray_dir * best_t.unsqueeze(-1)

    face_normal = normalize(cross(edge1, edge2))  # (M, 3), watertight per triangle winding
    outward_normal = face_normal.unsqueeze(0).expand(ray_dir.shape[0], -1, -1)  # (N, M, 3)
    gather_index = best_index.view(-1, 1, 1).expand(-1, 1, 3)
    selected_normal = torch.gather(outward_normal, 1, gather_index).squeeze(1)  # (N, 3)
    normal = _face_forward(selected_normal, ray_dir)

    return HitBatch(hit=hit, t=best_t, point=point, normal=normal)


def mesh_hit(
    mesh: Mesh,
    ray_origin: torch.Tensor,
    ray_dir: torch.Tensor,
    t_min: float,
    t_max: float,
) -> HitBatch:
    """Nearest-hit test of N rays against every triangle in `mesh`.

    Unpacks `mesh.vertices` (M, 3, 3) into the three (M, 3) corner tensors
    `triangle_hit` expects; see `triangle_hit` for the batching invariant.
    """
    v0 = mesh.vertices[:, 0]
    v1 = mesh.vertices[:, 1]
    v2 = mesh.vertices[:, 2]
    return triangle_hit(v0, v1, v2, ray_origin, ray_dir, t_min=t_min, t_max=t_max)
