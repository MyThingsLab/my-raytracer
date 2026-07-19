from __future__ import annotations

import pytest
import torch

from myraytracer.core.backend import torch_backend
from myraytracer.core.camera import Camera
from myraytracer.core.geometry import Plane
from myraytracer.core.integrator import render
from myraytracer.core.material import Material
from myraytracer.core.optimize import fit_albedo
from myraytracer.core.scene import PointLight, Scene

_BACKEND = torch_backend(device=torch.device("cpu"))


def _camera() -> Camera:
    return Camera(
        origin=(0.0, 0.0, 0.0),
        look_at=(0.0, 0.0, -1.0),
        up=(0.0, 1.0, 0.0),
        vfov_degrees=90,
        aspect_ratio=1,
    )


def _scene(albedo) -> Scene:
    plane = Plane(point=(0.0, 0.0, -5.0), normal=(0.0, 0.0, 1.0), material=Material(albedo=albedo))
    light = PointLight(position=(0.0, 0.0, -2.0), intensity=(10.0, 10.0, 10.0))
    return Scene(objects=[plane], lights=[light])


def _render(scene: Scene, camera: Camera):
    # max_depth=0 with no area lights: pure direct lighting, no randomness.
    return render(scene, camera, width=8, height=8, spp=1, max_depth=0, seed=0, backend=_BACKEND)


@pytest.mark.slow
def test_fit_albedo_recovers_true_albedo_from_rendered_target() -> None:
    torch.manual_seed(0)
    true_albedo = torch.tensor([0.8, 0.5, 0.2])
    camera = _camera()

    with torch.no_grad():
        target = _render(_scene(true_albedo), camera)

    initial_albedo = torch.clamp(true_albedo + 0.3, 0.0, 1.0)
    scene = _scene(initial_albedo)

    fitted = fit_albedo(
        scene,
        camera,
        target,
        object_index=0,
        width=8,
        height=8,
        spp=1,
        max_depth=0,
        seed=0,
        backend=_BACKEND,
        lr=0.05,
        steps=200,
    )

    assert torch.allclose(fitted, true_albedo, atol=0.02)


def test_fit_albedo_short_run_reduces_loss() -> None:
    torch.manual_seed(0)
    true_albedo = torch.tensor([0.8, 0.5, 0.2])
    camera = _camera()

    with torch.no_grad():
        target = _render(_scene(true_albedo), camera)

    initial_albedo = torch.clamp(true_albedo + 0.3, 0.0, 1.0)
    scene = _scene(initial_albedo)

    with torch.no_grad():
        loss_before = torch.mean((_render(scene, camera) - target) ** 2).item()

    fitted = fit_albedo(
        scene,
        camera,
        target,
        object_index=0,
        width=8,
        height=8,
        spp=1,
        max_depth=0,
        seed=0,
        backend=_BACKEND,
        lr=0.05,
        steps=5,
    )

    with torch.no_grad():
        loss_after = torch.mean((_render(_scene(fitted), camera) - target) ** 2).item()

    assert loss_after < loss_before


def test_fit_albedo_requires_torch_backend() -> None:
    from myraytracer.core.backend import NUMPY

    scene = _scene((0.5, 0.5, 0.5))
    camera = _camera()
    target = _render(scene, camera)

    with pytest.raises(ValueError, match="torch backend"):
        fit_albedo(
            scene,
            camera,
            target,
            object_index=0,
            width=8,
            height=8,
            spp=1,
            max_depth=0,
            seed=0,
            backend=NUMPY,
        )
