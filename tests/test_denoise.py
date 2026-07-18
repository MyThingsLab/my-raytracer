from __future__ import annotations

import numpy as np
import pytest

from myraytracer.camera import Camera
from myraytracer.denoise import denoise
from myraytracer.geometry import Plane
from myraytracer.light import PointLight
from myraytracer.material import Material
from myraytracer.render import render_gbuffers
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


def test_denoise_reduces_variance_in_flat_lit_region_without_biasing_mean() -> None:
    height, width = 32, 32
    mean = 0.5
    rng = np.random.Generator(np.random.PCG64(0))
    noisy = mean + rng.normal(scale=0.2, size=(height, width, 3))

    normal = np.zeros((height, width, 3))
    normal[..., 2] = 1.0
    albedo = np.full((height, width, 3), 0.8)
    depth = np.full((height, width, 1), 5.0)

    result = denoise(noisy, normal, albedo, depth)

    assert result.var() < noisy.var() * 0.1
    assert result.mean() == pytest.approx(mean, abs=0.02)


def test_denoise_does_not_blur_across_a_material_boundary() -> None:
    height, width = 32, 32
    left_color = 0.1
    right_color = 0.9
    color = np.zeros((height, width, 3))
    color[:, : width // 2] = left_color
    color[:, width // 2 :] = right_color

    albedo = color.copy()
    normal = np.zeros((height, width, 3))
    normal[..., 2] = 1.0
    depth = np.full((height, width, 1), 5.0)

    result = denoise(color, normal, albedo, depth)

    left_pixel = result[height // 2, width // 2 - 2]
    right_pixel = result[height // 2, width // 2 + 1]

    assert np.all(np.abs(left_pixel - left_color) < 0.05)
    assert np.all(np.abs(right_pixel - right_color) < 0.05)


def test_render_gbuffers_returns_expected_shapes() -> None:
    material = Material(albedo=Vec3(0.8, 0.5, 0.2))
    plane = Plane(point=Vec3(0, 0, -5), normal=Vec3(0, 0, 1), material=material)
    light = PointLight(position=Vec3(0, 0, -2), intensity=Vec3(10, 10, 10))
    scene = Scene(objects=[plane], lights=[light])

    normal, albedo, depth = render_gbuffers(scene, _camera(), width=8, height=8)

    assert normal.shape == (8, 8, 3)
    assert albedo.shape == (8, 8, 3)
    assert depth.shape == (8, 8, 1)


def test_render_gbuffers_head_on_ray_hits_plausible_normal() -> None:
    material = Material(albedo=Vec3(0.8, 0.5, 0.2))
    plane = Plane(point=Vec3(0, 0, -5), normal=Vec3(0, 0, 1), material=material)
    scene = Scene(objects=[plane], lights=[])

    normal, albedo, depth = render_gbuffers(scene, _camera(), width=8, height=8)

    center_normal = normal[4, 4]
    assert center_normal == pytest.approx((0.0, 0.0, 1.0), abs=1e-6)
    assert np.all(albedo[4, 4] == pytest.approx((0.8, 0.5, 0.2)))
    assert depth[4, 4, 0] == pytest.approx(5.0, abs=0.1)
