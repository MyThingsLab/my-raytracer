from __future__ import annotations

from dataclasses import dataclass

from myraytracer.vec import Vec3


@dataclass(frozen=True)
class Ray:
    origin: Vec3
    direction: Vec3  # assumed pre-normalized by the caller

    def at(self, t: float) -> Vec3:
        return self.origin + self.direction * t
