from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Material:
    """A surface material: albedo, optional emission, and a metallic-roughness
    lobe.

    Stored as plain scalars/3-tuples (like core.Camera) so one scene
    description is backend-agnostic; the arrays are built per-backend at
    nearest-hit time. Emission defaults to black; a non-zero emission turns a
    surface into a light (an emissive Quad acts as an area light via
    Scene.area_lights).

    `metallic` selects the reflection model: 0 (default) is a Lambertian
    diffuse surface (albedo = diffuse reflectance); 1 is a GGX microfacet
    conductor (albedo = specular F0 / reflection tint), with `roughness` in
    (0, 1] widening the specular lobe (small = mirror-like).
    """

    albedo: tuple[float, float, float]
    emission: tuple[float, float, float] = field(default=(0.0, 0.0, 0.0))
    metallic: float = 0.0
    roughness: float = 1.0

    @property
    def is_emissive(self) -> bool:
        return any(c > 0.0 for c in self.emission)
