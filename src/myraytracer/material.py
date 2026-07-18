from __future__ import annotations

from dataclasses import dataclass, field

from myraytracer.vec import Vec3


@dataclass(frozen=True)
class Material:
    albedo: Vec3
    emission: Vec3 = field(default_factory=lambda: Vec3(0, 0, 0))
