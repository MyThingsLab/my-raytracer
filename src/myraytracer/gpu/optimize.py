from __future__ import annotations

from dataclasses import replace

import torch

from myraytracer.gpu.camera import Camera
from myraytracer.gpu.render import render
from myraytracer.gpu.scene import Scene


def fit_albedo(
    scene: Scene,
    camera: Camera,
    target: torch.Tensor,
    *,
    primitive_index: int,
    width: int,
    height: int,
    lr: float = 0.05,
    steps: int = 200,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Recover a scene's Lambertian albedo by gradient descent against `target`.

    `primitive_index` is accepted for forward compatibility with a future
    per-primitive material system, but v1 has a single albedo shared by every
    object in the scene (see Scene.albedo), so that is the parameter fit here
    -- everything else in the scene, including which primitive owns the
    surface, stays fixed.
    """
    del primitive_index

    albedo = scene.albedo.detach().clone().requires_grad_(True)
    fit_scene = replace(scene, albedo=albedo)

    optimizer = torch.optim.Adam([albedo], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        result = render(fit_scene, camera, width=width, height=height, device=device)
        loss = torch.mean((result - target) ** 2)
        loss.backward()
        optimizer.step()

    return albedo.detach()
