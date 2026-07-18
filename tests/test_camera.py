from __future__ import annotations

import math

import pytest

from myraytracer.camera import Camera
from myraytracer.vec import Vec3


def test_center_pixel_looks_straight_down_negative_z() -> None:
    camera = Camera(
        origin=Vec3(0, 0, 0),
        look_at=Vec3(0, 0, -1),
        up=Vec3(0, 1, 0),
        vfov_degrees=90,
        aspect_ratio=1,
    )
    ray = camera.ray_for_pixel(px=1, py=1, width=2, height=2)
    assert ray.direction.x == pytest.approx(0.0, abs=1e-12)
    assert ray.direction.y == pytest.approx(0.0, abs=1e-12)
    assert ray.direction.z == pytest.approx(-1.0)


def test_off_center_pixel_matches_hand_computed_direction() -> None:
    camera = Camera(
        origin=Vec3(0, 0, 0),
        look_at=Vec3(0, 0, -1),
        up=Vec3(0, 1, 0),
        vfov_degrees=90,
        aspect_ratio=1,
    )
    ray = camera.ray_for_pixel(px=2, py=1, width=2, height=2)
    expected = 1 / math.sqrt(2)
    assert ray.direction.x == pytest.approx(expected)
    assert ray.direction.y == pytest.approx(0.0, abs=1e-12)
    assert ray.direction.z == pytest.approx(-expected)


def test_degenerate_single_pixel_image_does_not_crash() -> None:
    camera = Camera(
        origin=Vec3(0, 0, 0),
        look_at=Vec3(0, 0, -1),
        up=Vec3(0, 1, 0),
        vfov_degrees=90,
        aspect_ratio=1,
    )
    ray = camera.ray_for_pixel(px=0, py=0, width=1, height=1)
    assert ray.direction.length() == pytest.approx(1.0)
