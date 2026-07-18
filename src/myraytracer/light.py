from __future__ import annotations

from dataclasses import dataclass

from myraytracer.vec import Vec3


@dataclass(frozen=True)
class PointLight:
    position: Vec3
    intensity: Vec3

    def __post_init__(self) -> None:
        if self.intensity.x < 0.0 or self.intensity.y < 0.0 or self.intensity.z < 0.0:
            raise ValueError("light intensity components must be non-negative")
