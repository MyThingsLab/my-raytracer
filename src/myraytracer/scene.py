from __future__ import annotations

from dataclasses import dataclass, field

from myraytracer.bvh import BVH
from myraytracer.geometry import Hit, Plane, Quad, Sphere
from myraytracer.light import PointLight
from myraytracer.ray import Ray


@dataclass
class Scene:
    objects: list[Sphere | Plane | Quad]
    lights: list[PointLight]
    _bvh: BVH | None = field(default=None, init=False, repr=False, compare=False)

    def nearest_hit(
        self, ray: Ray, t_min: float, t_max: float, use_bvh: bool = False
    ) -> Hit | None:
        if use_bvh:
            return self._get_bvh().nearest_hit(ray, t_min, t_max)

        closest: Hit | None = None
        closest_t = t_max
        for obj in self.objects:
            hit = obj.hit(ray, t_min, closest_t)
            if hit is not None:
                closest = hit
                closest_t = hit.t
        return closest

    def _get_bvh(self) -> BVH:
        # Lazily built and cached: cheap for callers that never pass
        # use_bvh=True, and reused across the many rays a render issues.
        if self._bvh is None:
            self._bvh = BVH.build(self.objects)
        return self._bvh

    def area_lights(self) -> list[Quad]:
        return [
            obj
            for obj in self.objects
            if isinstance(obj, Quad)
            and (
                obj.material.emission.x > 0.0
                or obj.material.emission.y > 0.0
                or obj.material.emission.z > 0.0
            )
        ]
