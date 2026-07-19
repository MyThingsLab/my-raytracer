from __future__ import annotations

from dataclasses import dataclass

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
    normal: Array  # (N, 3)


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
    return Hit(hit=hit, t=t, point=point, normal=normal)


def plane_hit(
    plane: Plane, ray_origin: Array, ray_dir: Array, t_min: float, t_max: float, backend: Backend
) -> Hit:
    point0 = backend.asarray(plane.point)
    normal0 = backend.asarray(plane.normal)

    denom = dot(normal0, ray_dir)
    denom_safe = backend.where(abs(denom) < _PARALLEL_EPS, backend.ones_like(denom), denom)
    t = dot(point0 - ray_origin, normal0) / denom_safe
    in_range = (t > t_min) & (t < t_max)
    hit = in_range & (abs(denom) >= _PARALLEL_EPS)

    point = ray_origin + ray_dir * t[..., None]
    normal = _face_forward(normal0, ray_dir)
    return Hit(hit=hit, t=t, point=point, normal=normal)


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
    return Hit(hit=hit, t=t, point=point, normal=face_normal)


def hit_primitive(
    obj: Sphere | Plane | Quad,
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
    return quad_hit(obj, ray_origin, ray_dir, t_min, t_max, backend)
