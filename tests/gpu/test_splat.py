from __future__ import annotations

import torch

from myraytracer.gpu.camera import Camera
from myraytracer.gpu.splat import Gaussians, rasterize


def _camera(dtype: torch.dtype = torch.float32) -> Camera:
    return Camera(
        origin=torch.tensor([0.0, 0.0, 0.0], dtype=dtype),
        look_at=torch.tensor([0.0, 0.0, -1.0], dtype=dtype),
        up=torch.tensor([0.0, 1.0, 0.0], dtype=dtype),
        vfov_degrees=90,
        aspect_ratio=1,
    )


def _single_gaussian(opacity: float) -> Gaussians:
    return Gaussians(
        mean=torch.tensor([[0.0, 0.0, -5.0]]),
        scale=torch.tensor([[0.5, 0.5, 0.5]]),
        quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        opacity=torch.tensor([opacity]),
        color=torch.tensor([[1.0, 1.0, 1.0]]),
    )


def test_single_white_gaussian_produces_central_blob_that_decays_outward() -> None:
    camera = _camera()
    device = torch.device("cpu")

    image = rasterize(_single_gaussian(1.0), camera, width=64, height=64, device=device)

    center = image[32, 32].mean().item()
    mid = image[32, 16].mean().item()
    corner = image[2, 2].mean().item()

    assert center > 0.5
    assert center > mid > corner


def test_zero_opacity_yields_black_image() -> None:
    camera = _camera()
    device = torch.device("cpu")

    image = rasterize(_single_gaussian(0.0), camera, width=16, height=16, device=device)

    assert torch.allclose(image, torch.zeros_like(image))


def test_rasterize_is_deterministic_given_same_seed() -> None:
    camera = _camera()
    device = torch.device("cpu")

    def random_gaussians(seed: int) -> Gaussians:
        generator = torch.Generator().manual_seed(seed)
        count = 5
        mean_xy = torch.rand(count, 2, generator=generator) * 2 - 1
        mean_z = -torch.rand(count, 1, generator=generator) * 3 - 3
        return Gaussians(
            mean=torch.cat([mean_xy, mean_z], dim=-1),
            scale=torch.full((count, 3), 0.4),
            quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * count),
            opacity=torch.rand(count, generator=generator),
            color=torch.rand(count, 3, generator=generator),
        )

    image_a = rasterize(random_gaussians(42), camera, width=32, height=32, device=device)
    image_b = rasterize(random_gaussians(42), camera, width=32, height=32, device=device)

    assert torch.equal(image_a, image_b)


def test_rasterize_gradcheck_wrt_color_and_opacity() -> None:
    camera = _camera(dtype=torch.float64)
    device = torch.device("cpu")
    mean = torch.tensor(
        [[0.0, 0.0, -5.0], [0.3, -0.2, -4.0]], dtype=torch.float64
    )
    scale = torch.full((2, 3), 0.5, dtype=torch.float64)
    quat = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float64
    )
    opacity = torch.tensor([0.8, 0.5], dtype=torch.float64, requires_grad=True)
    color = torch.tensor(
        [[0.8, 0.5, 0.2], [0.1, 0.9, 0.4]], dtype=torch.float64, requires_grad=True
    )

    def render_fn(opacity: torch.Tensor, color: torch.Tensor) -> torch.Tensor:
        gaussians = Gaussians(mean=mean, scale=scale, quat=quat, opacity=opacity, color=color)
        return rasterize(gaussians, camera, width=4, height=4, device=device)

    assert torch.autograd.gradcheck(render_fn, (opacity, color))
