from __future__ import annotations

from dataclasses import dataclass

from myraytracer.geometry import Hit, Plane, Quad, Sphere
from myraytracer.light import PointLight
from myraytracer.ray import Ray


@dataclass
class Scene:
    objects: list[Sphere | Plane | Quad]
    lights: list[PointLight]

    def nearest_hit(self, ray: Ray, t_min: float, t_max: float) -> Hit | None:
        closest: Hit | None = None
        closest_t = t_max
        for obj in self.objects:
            hit = obj.hit(ray, t_min, closest_t)
            if hit is not None:
                closest = hit
                closest_t = hit.t
        return closest
