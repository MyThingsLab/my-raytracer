from __future__ import annotations

from dataclasses import dataclass, field

from myraytracer.vec import Vec3


@dataclass(frozen=True)
class Material:
    albedo: Vec3
    emission: Vec3 = field(default_factory=lambda: Vec3(0, 0, 0))
    # metallic 0 (default) = Lambertian diffuse; 1 = GGX conductor with the
    # given roughness. Only the batched core integrator shades these; the
    # deprecated scalar tracer ignores them.
    metallic: float = 0.0
    roughness: float = 1.0
    transmission: float = 0.0
    ior: float = 1.5
