from __future__ import annotations

import pytest
import torch

from myraytracer.core.backend import NUMPY, torch_backend
from myraytracer.core.camera import Camera
from myraytracer.core.optimize import fit_gaussians
from myraytracer.core.splat import Gaussians, rasterize

_BACKEND = torch_backend(device=torch.device("cpu"))


def _camera() -> Camera:
    return Camera(
        origin=(0.0, 0.0, 0.0),
        look_at=(0.0, 0.0, -1.0),
        up=(0.0, 1.0, 0.0),
        vfov_degrees=90,
        aspect_ratio=1,
    )


def _gaussians(mean, scale, quat, opacity, color) -> Gaussians:
    return Gaussians(
        mean=torch.tensor(mean),
        scale=torch.tensor(scale),
        quat=torch.tensor(quat),
        opacity=torch.tensor(opacity),
        color=torch.tensor(color),
    )


def _single(color=(0.2, 0.7, 0.9), opacity=(0.9,), z=-3.0) -> Gaussians:
    return _gaussians(
        [[0.0, 0.0, z]], [[0.3, 0.3, 0.3]], [[1.0, 0.0, 0.0, 0.0]], list(opacity), [list(color)]
    )


def test_rasterize_center_pixel_is_color_times_opacity() -> None:
    image = rasterize(_single(), _camera(), 16, 16, backend=_BACKEND)
    assert tuple(image.shape) == (16, 16, 3)
    # A Gaussian dead-center, fully in front, deposits color * opacity at its peak.
    assert torch.allclose(image[8, 8], torch.tensor([0.18, 0.63, 0.81]), atol=1e-4)
    # Far corners are outside the ~1px footprint -> background stays black.
    assert torch.allclose(image[0, 0], torch.zeros(3), atol=1e-4)


def test_rasterize_gradients_flow_to_all_parameters() -> None:
    g = _single()
    mean = g.mean.clone().requires_grad_(True)
    scale = g.scale.clone().requires_grad_(True)
    quat = g.quat.clone().requires_grad_(True)
    opacity = g.opacity.clone().requires_grad_(True)
    color = g.color.clone().requires_grad_(True)
    leaf = Gaussians(mean=mean, scale=scale, quat=quat, opacity=opacity, color=color)

    rasterize(leaf, _camera(), 16, 16, backend=_BACKEND).sum().backward()

    for param in (mean, scale, quat, opacity, color):
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
    assert opacity.grad.abs().sum() > 0
    assert color.grad.abs().sum() > 0


def test_rasterize_ignores_gaussians_behind_camera() -> None:
    # Same Gaussian placed at +z (behind a camera looking down -z) must not appear.
    behind = _single(z=3.0)
    image = rasterize(behind, _camera(), 16, 16, backend=_BACKEND)
    assert torch.allclose(image, torch.zeros_like(image), atol=1e-6)


def test_rasterize_composites_front_to_back() -> None:
    # A near red Gaussian in front of a far blue one: the opaque near one occludes.
    near_red = [[0.0, 0.0, -2.0]]
    far_blue = [[0.0, 0.0, -6.0]]
    two = _gaussians(
        near_red + far_blue,
        [[0.3, 0.3, 0.3], [0.3, 0.3, 0.3]],
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        [1.0, 1.0],
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    )
    center = rasterize(two, _camera(), 16, 16, backend=_BACKEND)[8, 8]
    assert center[0] > center[2]  # red dominates; the far blue is transmittance-suppressed


def test_rasterize_requires_torch_backend() -> None:
    with pytest.raises(ValueError, match="torch backend"):
        rasterize(_single(), _camera(), 16, 16, backend=NUMPY)


def test_fit_gaussians_reduces_loss() -> None:
    torch.manual_seed(0)
    camera = _camera()
    target_g = _single(color=(0.9, 0.3, 0.1), opacity=(0.95,))
    with torch.no_grad():
        target = rasterize(target_g, camera, 16, 16, backend=_BACKEND)

    init = _single(color=(0.2, 0.2, 0.2), opacity=(0.5,))
    with torch.no_grad():
        loss_before = (
            ((rasterize(init, camera, 16, 16, backend=_BACKEND) - target) ** 2).mean().item()
        )

    fitted = fit_gaussians(init, camera, target, width=16, height=16, backend=_BACKEND, steps=50)

    with torch.no_grad():
        loss_after = (
            ((rasterize(fitted, camera, 16, 16, backend=_BACKEND) - target) ** 2).mean().item()
        )
    assert loss_after < loss_before


def test_fit_gaussians_requires_torch_backend() -> None:
    camera = _camera()
    target = rasterize(_single(), camera, 16, 16, backend=_BACKEND)
    with pytest.raises(ValueError, match="torch backend"):
        fit_gaussians(_single(), camera, target, width=16, height=16, backend=NUMPY)
