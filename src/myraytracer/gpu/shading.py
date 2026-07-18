from __future__ import annotations

import torch

from myraytracer.gpu.geometry import HitBatch
from myraytracer.gpu.scene import Scene
from myraytracer.gpu.vec import dot, length


def direct_lighting(hit: HitBatch, albedo: torch.Tensor, scene: Scene) -> torch.Tensor:
    radiance = torch.zeros_like(albedo)

    for light in scene.lights:
        to_light = light.position - hit.point
        distance = length(to_light)
        light_dir = to_light / distance.unsqueeze(-1)
        cos_theta = torch.clamp(dot(hit.normal, light_dir), min=0.0)

        # Visibility is detached: a true visibility gradient would need
        # reparameterization/edge-sampling at occluder silhouettes, which is
        # out of scope here, so shadowing contributes zero gradient instead
        # of an undefined one.
        with torch.no_grad():
            shadow_hit = scene.nearest_hit(
                hit.point, light_dir, t_min=1e-4, t_max=distance - 1e-4
            )
            visibility = (~shadow_hit.hit).to(albedo.dtype)

        falloff = cos_theta / (distance * distance)
        radiance = radiance + albedo * light.intensity * (falloff * visibility).unsqueeze(-1)

    return radiance
