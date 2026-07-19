from __future__ import annotations

import math

import torch

from myraytracer.gpu.camera import Camera
from myraytracer.gpu.geometry import Plane, Sphere
from myraytracer.gpu.scene import PointLight, Scene
from myraytracer.gpu.shading import direct_lighting
from myraytracer.gpu.vec import device as default_device


def _scene_to_device(scene: Scene, device: torch.device) -> Scene:
    objects: list[Sphere | Plane] = []
    for obj in scene.objects:
        if isinstance(obj, Sphere):
            objects.append(Sphere(center=obj.center.to(device), radius=obj.radius.to(device)))
        else:
            objects.append(Plane(point=obj.point.to(device), normal=obj.normal.to(device)))
    lights = [
        PointLight(position=light.position.to(device), intensity=light.intensity.to(device))
        for light in scene.lights
    ]
    return Scene(
        objects=objects,
        lights=lights,
        albedo=scene.albedo.to(device),
        emission=scene.emission.to(device),
    )


def render(
    scene: Scene,
    camera: Camera,
    *,
    width: int,
    height: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    if device is None:
        device = default_device()

    scene = _scene_to_device(scene, device)
    ray_origin, ray_dir = camera.rays(width, height, device)
    hit = scene.nearest_hit(ray_origin, ray_dir, t_min=1e-4, t_max=math.inf)

    albedo = scene.albedo.to(dtype=ray_origin.dtype)
    radiance = direct_lighting(hit, albedo.expand(ray_origin.shape[0], -1), scene)
    radiance = torch.where(hit.hit.unsqueeze(-1), radiance, torch.zeros_like(radiance))

    return radiance.reshape(height, width, 3)
