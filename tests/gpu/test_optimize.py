from __future__ import annotations

import torch

from myraytracer.gpu.camera import Camera
from myraytracer.gpu.geometry import Plane
from myraytracer.gpu.optimize import fit_albedo
from myraytracer.gpu.render import render
from myraytracer.gpu.scene import PointLight, Scene


def _camera() -> Camera:
    return Camera(
        origin=torch.tensor([0.0, 0.0, 0.0]),
        look_at=torch.tensor([0.0, 0.0, -1.0]),
        up=torch.tensor([0.0, 1.0, 0.0]),
        vfov_degrees=90,
        aspect_ratio=1,
    )


def _scene(albedo: torch.Tensor) -> Scene:
    plane = Plane(point=torch.tensor([0.0, 0.0, -5.0]), normal=torch.tensor([0.0, 0.0, 1.0]))
    light = PointLight(
        position=torch.tensor([0.0, 0.0, -2.0]), intensity=torch.tensor([10.0, 10.0, 10.0])
    )
    return Scene(objects=[plane], lights=[light], albedo=albedo)


def test_fit_albedo_recovers_true_albedo_from_rendered_target() -> None:
    torch.manual_seed(0)
    true_albedo = torch.tensor([0.8, 0.5, 0.2])
    camera = _camera()
    device = torch.device("cpu")

    with torch.no_grad():
        target = render(_scene(true_albedo), camera, width=8, height=8, device=device)

    initial_albedo = torch.clamp(true_albedo + 0.3, 0.0, 1.0)
    scene = _scene(initial_albedo)

    fitted = fit_albedo(
        scene,
        camera,
        target,
        primitive_index=0,
        width=8,
        height=8,
        lr=0.05,
        steps=200,
        device=device,
    )

    assert torch.allclose(fitted, true_albedo, atol=0.02)


def test_fit_albedo_short_run_reduces_loss() -> None:
    torch.manual_seed(0)
    true_albedo = torch.tensor([0.8, 0.5, 0.2])
    camera = _camera()
    device = torch.device("cpu")

    with torch.no_grad():
        target = render(_scene(true_albedo), camera, width=8, height=8, device=device)

    initial_albedo = torch.clamp(true_albedo + 0.3, 0.0, 1.0)
    scene = _scene(initial_albedo)

    with torch.no_grad():
        loss_before = torch.mean(
            (render(scene, camera, width=8, height=8, device=device) - target) ** 2
        ).item()

    fitted = fit_albedo(
        scene,
        camera,
        target,
        primitive_index=0,
        width=8,
        height=8,
        lr=0.05,
        steps=5,
        device=device,
    )

    with torch.no_grad():
        loss_after = torch.mean(
            (render(_scene(fitted), camera, width=8, height=8, device=device) - target) ** 2
        ).item()

    assert loss_after < loss_before
