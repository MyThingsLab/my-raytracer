from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from myraytracer.core.backend import Array, Backend, backend_of
from myraytracer.core.linalg import cross, dot, normalize
from myraytracer.core.material import Material

_PARALLEL_EPS = 1e-12


@dataclass(frozen=True)
class Hit:
    """Per-ray geometric intersection result, all shaped over the ray batch.

    Purely geometric (no material): the Scene layers albedo/emission on top by
    tracking which primitive owns each ray's nearest hit.
    """

    hit: Array  # (N,) bool
    t: Array  # (N,)
    point: Array  # (N, 3)
    normal: Array  # (N, 3), face-forwarded to oppose the ray (shading normal)
    geo_normal: Array = None  # (N, 3), outward geometric normal (for refraction)


@dataclass(frozen=True)
class Sphere:
    center: tuple[float, float, float]
    radius: float
    material: Material


@dataclass(frozen=True)
class Plane:
    point: tuple[float, float, float]
    normal: tuple[float, float, float]
    material: Material


@dataclass(frozen=True)
class Quad:
    corner: tuple[float, float, float]
    edge1: tuple[float, float, float]
    edge2: tuple[float, float, float]
    material: Material


@dataclass(frozen=True)
class Mesh:
    """A triangle mesh as a batch of triangle-vertex positions.

    `vertices` is (M, 3, 3): triangle index, vertex-in-triangle (v0/v1/v2),
    xyz -- a numpy array (backend-agnostic; converted per backend at hit time),
    typically from `myraytracer.core.mesh.load_obj`.
    """

    vertices: Any
    material: Material


def _face_forward(normal: Array, direction: Array) -> Array:
    # Flip the normal to oppose the ray, so shading always sees the front face.
    backend = backend_of(direction)
    flip = (dot(normal, direction) > 0.0)[..., None]
    return backend.where(flip, -normal, normal)


def sphere_hit(
    sphere: Sphere, ray_origin: Array, ray_dir: Array, t_min: float, t_max: float, backend: Backend
) -> Hit:
    center = backend.asarray(sphere.center)
    radius = sphere.radius

    oc = ray_origin - center
    a = dot(ray_dir, ray_dir)
    half_b = dot(oc, ray_dir)
    c = dot(oc, oc) - radius * radius
    discriminant = half_b * half_b - a * c

    # Clamp before the sqrt so no-real-root rows stay finite (and, on torch,
    # differentiable) instead of NaN-poisoning the batch -- matches gpu.geometry.
    sqrt_discriminant = backend.clip(discriminant, lo=0.0) ** 0.5
    root_near = (-half_b - sqrt_discriminant) / a
    root_far = (-half_b + sqrt_discriminant) / a
    near_in_range = (root_near > t_min) & (root_near < t_max)
    far_in_range = (root_far > t_min) & (root_far < t_max)

    t = backend.where(near_in_range, root_near, root_far)
    hit = (discriminant >= 0.0) & (near_in_range | far_in_range)

    point = ray_origin + ray_dir * t[..., None]
    outward_normal = (point - center) / radius
    normal = _face_forward(outward_normal, ray_dir)
    return Hit(hit=hit, t=t, point=point, normal=normal, geo_normal=outward_normal)


def plane_hit(
    plane: Plane, ray_origin: Array, ray_dir: Array, t_min: float, t_max: float, backend: Backend
) -> Hit:
    point0 = backend.asarray(plane.point)
    # Normalize so the shaded normal is unit even when the scene supplies a
    # non-unit plane normal (t is scale-invariant, but shading needs a unit).
    normal0 = normalize(backend.asarray(plane.normal))

    denom = dot(normal0, ray_dir)
    denom_safe = backend.where(abs(denom) < _PARALLEL_EPS, backend.ones_like(denom), denom)
    t = dot(point0 - ray_origin, normal0) / denom_safe
    in_range = (t > t_min) & (t < t_max)
    hit = in_range & (abs(denom) >= _PARALLEL_EPS)

    point = ray_origin + ray_dir * t[..., None]
    normal = _face_forward(normal0, ray_dir)
    geo_normal = backend.broadcast_to(normal0, point.shape)
    return Hit(hit=hit, t=t, point=point, normal=normal, geo_normal=geo_normal)


def quad_hit(
    quad: Quad, ray_origin: Array, ray_dir: Array, t_min: float, t_max: float, backend: Backend
) -> Hit:
    corner = backend.asarray(quad.corner)
    edge1 = backend.asarray(quad.edge1)
    edge2 = backend.asarray(quad.edge2)

    normal = normalize(cross(edge1, edge2))
    denom = dot(normal, ray_dir)
    denom_safe = backend.where(abs(denom) < _PARALLEL_EPS, backend.ones_like(denom), denom)
    t = dot(corner - ray_origin, normal) / denom_safe

    point = ray_origin + ray_dir * t[..., None]
    hit_vec = point - corner

    e1_len_sq = dot(edge1, edge1)
    e2_len_sq = dot(edge2, edge2)
    e1_dot_e2 = dot(edge1, edge2)
    denom_ab = e1_len_sq * e2_len_sq - e1_dot_e2 * e1_dot_e2

    hit_dot_e1 = dot(hit_vec, edge1)
    hit_dot_e2 = dot(hit_vec, edge2)
    alpha = (hit_dot_e1 * e2_len_sq - hit_dot_e2 * e1_dot_e2) / denom_ab
    beta = (hit_dot_e2 * e1_len_sq - hit_dot_e1 * e1_dot_e2) / denom_ab

    in_range = (t > t_min) & (t < t_max)
    inside = (alpha >= 0.0) & (alpha <= 1.0) & (beta >= 0.0) & (beta <= 1.0)
    hit = in_range & inside & (abs(denom) >= _PARALLEL_EPS)

    face_normal = _face_forward(normal, ray_dir)
    geo_normal = backend.broadcast_to(normal, point.shape)
    return Hit(hit=hit, t=t, point=point, normal=face_normal, geo_normal=geo_normal)


def mesh_hit(
    mesh: Mesh, ray_origin: Array, ray_dir: Array, t_min: float, t_max: float, backend: Backend
) -> Hit:
    # Batched Moller-Trumbore: every ray (N) against every triangle (M) as an
    # (N, M) intermediate, reduced over the triangle axis to each ray's nearest
    # hit, so the result is shaped (N,) / (N, 3) like the other primitives. On
    # torch this stays differentiable through the winning triangle.
    vertices = backend.asarray(mesh.vertices)
    v0 = vertices[:, 0]
    edge1 = vertices[:, 1] - v0
    edge2 = vertices[:, 2] - v0

    ray_dir_b = ray_dir[:, None, :]
    ray_origin_b = ray_origin[:, None, :]
    edge1_b = edge1[None, :, :]
    edge2_b = edge2[None, :, :]

    h = cross(ray_dir_b, edge2_b)
    a = dot(edge1_b, h)
    not_parallel = abs(a) >= _PARALLEL_EPS
    a_safe = backend.where(not_parallel, a, backend.ones_like(a))
    f = 1.0 / a_safe

    s = ray_origin_b - v0[None, :, :]
    u = f * dot(s, h)
    q = cross(s, edge1_b)
    v = f * dot(ray_dir_b, q)
    t = f * dot(edge2_b, q)

    # `t_max` may be a per-ray (N,) array (shadow rays); add the triangle axis.
    t_max_b = t_max[:, None] if hasattr(t_max, "ndim") else t_max
    in_range = (t > t_min) & (t < t_max_b)
    barycentric_ok = (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0)
    valid = not_parallel & barycentric_ok & in_range

    # Mask invalid (ray, triangle) pairs to +inf before the row-wise min, so
    # only the triangle owning each ray's nearest hit contributes.
    t_candidates = backend.where(valid, t, backend.full_like(t, math.inf))
    best_t, best_index = backend.min_along(t_candidates, axis=1)
    hit = backend.xp.isfinite(best_t)

    # Missed rays carry t=+inf; use a finite placeholder for the point so the
    # multiply below never produces inf/nan (the point is masked out by `hit`).
    safe_t = backend.where(hit, best_t, backend.zeros_like(best_t))
    point = ray_origin + ray_dir * safe_t[..., None]

    face_normal = normalize(cross(edge1, edge2))  # (M, 3), per triangle winding
    n_rays = ray_dir.shape[0]
    n_tris = edge1.shape[0]
    outward = backend.broadcast_to(face_normal[None, :, :], (n_rays, n_tris, 3))
    gather_index = backend.broadcast_to(best_index[:, None, None], (n_rays, 1, 3))
    selected = backend.take_along(outward, gather_index, axis=1)[:, 0, :]
    normal = _face_forward(selected, ray_dir)

    return Hit(hit=hit, t=best_t, point=point, normal=normal, geo_normal=selected)


def hit_primitive(
    obj: Sphere | Plane | Quad | Mesh,
    ray_origin: Array,
    ray_dir: Array,
    t_min: float,
    t_max: float,
    backend: Backend,
) -> Hit:
    if isinstance(obj, Sphere):
        return sphere_hit(obj, ray_origin, ray_dir, t_min, t_max, backend)
    if isinstance(obj, Plane):
        return plane_hit(obj, ray_origin, ray_dir, t_min, t_max, backend)
    if isinstance(obj, Quad):
        return quad_hit(obj, ray_origin, ray_dir, t_min, t_max, backend)
    return mesh_hit(obj, ray_origin, ray_dir, t_min, t_max, backend)
