from __future__ import annotations

import math
from dataclasses import dataclass

from myraytracer.material import Material
from myraytracer.ray import Ray
from myraytracer.vec import Vec3


@dataclass(frozen=True)
class Hit:
    t: float
    point: Vec3
    normal: Vec3
    material: Material


def _face_forward(normal: Vec3, direction: Vec3) -> Vec3:
    if normal.dot(direction) > 0:
        return normal * -1
    return normal


@dataclass(frozen=True)
class Sphere:
    center: Vec3
    radius: float
    material: Material

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError("sphere radius must be positive")

    def hit(self, ray: Ray, t_min: float, t_max: float) -> Hit | None:
        oc = ray.origin - self.center
        a = ray.direction.length_squared()
        half_b = oc.dot(ray.direction)
        c = oc.length_squared() - self.radius * self.radius
        discriminant = half_b * half_b - a * c
        if discriminant < 0:
            return None

        sqrt_discriminant = math.sqrt(discriminant)
        root = (-half_b - sqrt_discriminant) / a
        if not (t_min < root < t_max):
            root = (-half_b + sqrt_discriminant) / a
            if not (t_min < root < t_max):
                return None

        point = ray.at(root)
        normal = _face_forward((point - self.center).normalized(), ray.direction)
        return Hit(t=root, point=point, normal=normal, material=self.material)


@dataclass(frozen=True)
class Plane:
    point: Vec3
    normal: Vec3
    material: Material

    def __post_init__(self) -> None:
        if self.normal.length_squared() < 1e-12:
            raise ValueError("plane normal must not be near-zero-length")

    def hit(self, ray: Ray, t_min: float, t_max: float) -> Hit | None:
        denom = self.normal.dot(ray.direction)
        if abs(denom) < 1e-12:
            return None

        root = (self.point - ray.origin).dot(self.normal) / denom
        if not (t_min < root < t_max):
            return None

        point = ray.at(root)
        normal = _face_forward(self.normal.normalized(), ray.direction)
        return Hit(t=root, point=point, normal=normal, material=self.material)


@dataclass(frozen=True)
class Quad:
    corner: Vec3
    edge1: Vec3
    edge2: Vec3
    material: Material

    def __post_init__(self) -> None:
        cross = self.edge1.cross(self.edge2)
        if cross.length_squared() < 1e-12:
            raise ValueError("quad edge1/edge2 must not be (near-)parallel")

    def hit(self, ray: Ray, t_min: float, t_max: float) -> Hit | None:
        normal = self.edge1.cross(self.edge2).normalized()
        denom = normal.dot(ray.direction)
        if abs(denom) < 1e-12:
            return None

        root = (self.corner - ray.origin).dot(normal) / denom
        if not (t_min < root < t_max):
            return None

        point = ray.at(root)
        hit_vec = point - self.corner

        e1_len_sq = self.edge1.length_squared()
        e2_len_sq = self.edge2.length_squared()
        e1_dot_e2 = self.edge1.dot(self.edge2)
        denom_ab = e1_len_sq * e2_len_sq - e1_dot_e2 * e1_dot_e2

        hit_dot_e1 = hit_vec.dot(self.edge1)
        hit_dot_e2 = hit_vec.dot(self.edge2)

        alpha = (hit_dot_e1 * e2_len_sq - hit_dot_e2 * e1_dot_e2) / denom_ab
        beta = (hit_dot_e2 * e1_len_sq - hit_dot_e1 * e1_dot_e2) / denom_ab

        if not (0.0 <= alpha <= 1.0 and 0.0 <= beta <= 1.0):
            return None

        face_normal = _face_forward(normal, ray.direction)
        return Hit(t=root, point=point, normal=face_normal, material=self.material)
