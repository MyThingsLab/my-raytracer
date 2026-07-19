from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Material:
    """A Lambertian surface: diffuse albedo plus optional emission.

    Stored as plain 3-tuples (like core.Camera) so one scene description is
    backend-agnostic; the arrays are built per-backend at nearest-hit time.
    Emission defaults to black; a non-zero emission turns a surface into a
    light (an emissive Quad acts as an area light via Scene.area_lights).
    """

    albedo: tuple[float, float, float]
    emission: tuple[float, float, float] = field(default=(0.0, 0.0, 0.0))

    @property
    def is_emissive(self) -> bool:
        return any(c > 0.0 for c in self.emission)
