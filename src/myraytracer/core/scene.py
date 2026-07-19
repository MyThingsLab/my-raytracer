from __future__ import annotations

import math
from dataclasses import dataclass

from myraytracer.core.backend import Array, Backend
from myraytracer.core.geometry import Hit, Plane, Quad, Sphere, hit_primitive


@dataclass(frozen=True)
class PointLight:
    position: tuple[float, float, float]
    intensity: tuple[float, float, float]


@dataclass(frozen=True)
class SceneHit(Hit):
    # A geometric Hit plus the per-ray surface material, gathered from whichever
    # primitive owns each ray's nearest hit. This is what gives every backend
    # per-object materials (the GPU path previously had one global albedo).
    albedo: Array = None  # (N, 3)
    emission: Array = None  # (N, 3)


@dataclass
class Scene:
    objects: list[Sphere | Plane | Quad]
    lights: list[PointLight]

    def nearest_hit(
        self,
        ray_origin: Array,
        ray_dir: Array,
        t_min: float,
        t_max: float,
        backend: Backend,
    ) -> SceneHit:
        # Running nearest-hit over primitives, carrying material alongside the
        # geometry: each closer hit masks in its primitive's albedo/emission via
        # `where`, so gradients (on torch) flow only through the winning
        # primitive -- the same masked-min idiom as gpu.scene, extended to
        # per-object materials.
        column = ray_origin[..., 0]
        acc_t = backend.full_like(column, t_max)
        acc_point = backend.zeros_like(ray_origin)
        acc_normal = backend.zeros_like(ray_origin)
        acc_albedo = backend.zeros_like(ray_origin)
        acc_emission = backend.zeros_like(ray_origin)

        for obj in self.objects:
            result = hit_primitive(obj, ray_origin, ray_dir, t_min, t_max, backend)
            closer = result.hit & (result.t < acc_t)
            mask = closer[..., None]

            acc_t = backend.where(closer, result.t, acc_t)
            acc_point = backend.where(mask, result.point, acc_point)
            acc_normal = backend.where(mask, result.normal, acc_normal)
            acc_albedo = backend.where(mask, backend.asarray(obj.material.albedo), acc_albedo)
            acc_emission = backend.where(
                mask, backend.asarray(obj.material.emission), acc_emission
            )

        hit = acc_t < t_max
        return SceneHit(
            hit=hit,
            t=acc_t,
            point=acc_point,
            normal=acc_normal,
            albedo=acc_albedo,
            emission=acc_emission,
        )

    def area_lights(self) -> list[Quad]:
        return [obj for obj in self.objects if isinstance(obj, Quad) and obj.material.is_emissive]


def opaque_t_max() -> float:
    # Convenience for callers that want "no far bound"; kept as a function so
    # the sentinel lives in one place.
    return math.inf
