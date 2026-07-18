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
