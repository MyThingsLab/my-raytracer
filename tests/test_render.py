from __future__ import annotations

import numpy as np

from myraytracer.camera import Camera
from myraytracer.geometry import Plane
from myraytracer.light import PointLight
from myraytracer.material import Material
from myraytracer.render import render
from myraytracer.scene import Scene
from myraytracer.vec import Vec3


def _camera() -> Camera:
    return Camera(
        origin=Vec3(0, 0, 0),
        look_at=Vec3(0, 0, -1),
        up=Vec3(0, 1, 0),
        vfov_degrees=90,
        aspect_ratio=1,
    )


def _scene() -> Scene:
    material = Material(albedo=Vec3(0.8, 0.5, 0.2))
    plane = Plane(point=Vec3(0, 0, -5), normal=Vec3(0, 0, 1), material=material)
    light = PointLight(position=Vec3(0, 0, -2), intensity=Vec3(10, 10, 10))
    return Scene(objects=[plane], lights=[light])


def test_render_returns_expected_shape() -> None:
    result = render(
        _scene(), _camera(), width=4, height=4, spp=4, max_depth=1, seed=0
    )

    assert result.shape == (4, 4, 3)


def test_render_is_deterministic_given_same_seed() -> None:
    result_a = render(
        _scene(), _camera(), width=4, height=4, spp=4, max_depth=1, seed=42
    )
    result_b = render(
        _scene(), _camera(), width=4, height=4, spp=4, max_depth=1, seed=42
    )

    assert np.array_equal(result_a, result_b)


def test_render_produces_non_degenerate_image() -> None:
    result = render(
        _scene(), _camera(), width=4, height=4, spp=4, max_depth=1, seed=0
    )

    assert np.any(result > 0.0)
