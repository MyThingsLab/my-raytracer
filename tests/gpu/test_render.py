from __future__ import annotations

import torch

from myraytracer.gpu.camera import Camera
from myraytracer.gpu.geometry import Plane
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


def _scene(albedo: torch.Tensor | None = None) -> Scene:
    plane = Plane(point=torch.tensor([0.0, 0.0, -5.0]), normal=torch.tensor([0.0, 0.0, 1.0]))
    light = PointLight(
        position=torch.tensor([0.0, 0.0, -2.0]), intensity=torch.tensor([10.0, 10.0, 10.0])
    )
    if albedo is None:
        return Scene(objects=[plane], lights=[light])
    return Scene(objects=[plane], lights=[light], albedo=albedo)


def test_render_output_shape_is_height_width_three() -> None:
    result = render(_scene(), _camera(), width=6, height=4, device=torch.device("cpu"))

    assert result.shape == (4, 6, 3)


def test_render_produces_non_degenerate_image() -> None:
    result = render(_scene(), _camera(), width=8, height=8, device=torch.device("cpu"))

    assert bool(torch.any(result > 0.0))


def test_render_runs_without_specifying_a_device() -> None:
    result = render(_scene(), _camera(), width=2, height=2)

    assert result.shape == (2, 2, 3)


def test_render_is_end_to_end_differentiable_wrt_scene_albedo() -> None:
    albedo = torch.tensor([0.8, 0.5, 0.2], requires_grad=True)
    scene = _scene(albedo=albedo)
    camera = _camera()
    target = torch.ones(4, 4, 3)

    result = render(scene, camera, width=4, height=4, device=torch.device("cpu"))
    loss = torch.mean((result - target) ** 2)
    loss.backward()

    assert albedo.grad is not None
    assert bool(torch.any(albedo.grad != 0.0))
