from __future__ import annotations

import math
from dataclasses import dataclass

from myraytracer.ray import Ray
from myraytracer.vec import Vec3


@dataclass(frozen=True)
class Camera:
    origin: Vec3
    look_at: Vec3
    up: Vec3
    vfov_degrees: float
    aspect_ratio: float

    def ray_for_pixel(self, px: float, py: float, width: int, height: int) -> Ray:
        forward = (self.look_at - self.origin).normalized()
        right = forward.cross(self.up).normalized()
        true_up = right.cross(forward)

        half_height = math.tan(math.radians(self.vfov_degrees) / 2)
        half_width = half_height * self.aspect_ratio

        u = (2 * px / width - 1) * half_width
        v = (1 - 2 * py / height) * half_height

        direction = forward + right * u + true_up * v
        return Ray(origin=self.origin, direction=direction.normalized())
