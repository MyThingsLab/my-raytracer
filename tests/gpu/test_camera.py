from __future__ import annotations

import numpy as np
import pytest
import torch

from myraytracer.camera import Camera as NumpyCamera
from myraytracer.gpu.camera import Camera
from myraytracer.vec import Vec3


def _gpu_camera() -> Camera:
    return Camera(
        origin=torch.tensor([0.0, 0.0, 0.0]),
        look_at=torch.tensor([0.0, 0.0, -1.0]),
        up=torch.tensor([0.0, 1.0, 0.0]),
        vfov_degrees=90,
        aspect_ratio=1,
    )


def _numpy_camera() -> NumpyCamera:
    return NumpyCamera(
        origin=Vec3(0, 0, 0),
        look_at=Vec3(0, 0, -1),
        up=Vec3(0, 1, 0),
        vfov_degrees=90,
        aspect_ratio=1,
    )


def test_rays_shape_matches_width_times_height() -> None:
    camera = _gpu_camera()

    origins, directions = camera.rays(width=4, height=3, device=torch.device("cpu"))

    assert origins.shape == (12, 3)
    assert directions.shape == (12, 3)


def test_rays_directions_match_numpy_backend_for_sampled_pixels() -> None:
    width, height = 8, 8
    gpu_camera = _gpu_camera()
    numpy_camera = _numpy_camera()

    origins, directions = gpu_camera.rays(width=width, height=height, device=torch.device("cpu"))

    for py in range(height):
        for px in range(width):
            expected = numpy_camera.ray_for_pixel(px=px, py=py, width=width, height=height)
            row = py * width + px
            actual = directions[row].numpy()

            assert actual == pytest.approx(
                np.array([expected.direction.x, expected.direction.y, expected.direction.z]),
                abs=1e-6,
            )


def test_rays_origins_are_all_the_camera_origin() -> None:
    camera = _gpu_camera()

    origins, _ = camera.rays(width=2, height=2, device=torch.device("cpu"))

    assert torch.allclose(origins, torch.zeros_like(origins))


def test_rays_directions_are_unit_length() -> None:
    camera = _gpu_camera()

    _, directions = camera.rays(width=5, height=5, device=torch.device("cpu"))

    norms = torch.linalg.vector_norm(directions, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
