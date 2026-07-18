from __future__ import annotations

from dataclasses import dataclass

import torch

from myraytracer.gpu.vec import dot

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
