from __future__ import annotations

import numpy as np

from myraytracer.camera import Camera as ScalarCamera
from myraytracer.core.backend import NUMPY, Backend
from myraytracer.core.camera import Camera
from myraytracer.vec import Vec3

_ORIGIN = (0.0, 0.0, 0.0)
_LOOK_AT = (0.0, 0.0, -1.0)
_UP = (0.0, 1.0, 0.0)
_VFOV = 60.0


def _core_camera() -> Camera:
    return Camera(
        origin=_ORIGIN, look_at=_LOOK_AT, up=_UP, vfov_degrees=_VFOV, aspect_ratio=1.0
    )


def test_grid_rays_shape_and_normalized(backend: Backend) -> None:
    width, height = 5, 4
    rays = _core_camera().grid_rays(width, height, backend)
    direction = np.asarray(rays.direction)
    origin = np.asarray(rays.origin)
    assert direction.shape == (width * height, 3)
    assert origin.shape == (width * height, 3)
    assert np.allclose(np.linalg.norm(direction, axis=-1), 1.0, atol=1e-5)
    assert np.allclose(origin, 0.0)


def test_grid_rays_match_scalar_camera() -> None:
    # The batched numpy camera must reproduce the validated scalar camera
    # ray-for-ray at integer pixel coordinates (the render loop's jitter
    # centre), so nothing downstream shifts when the core takes over.
    width, height = 6, 5
    scalar = ScalarCamera(
        origin=Vec3(*_ORIGIN),
        look_at=Vec3(*_LOOK_AT),
        up=Vec3(*_UP),
        vfov_degrees=_VFOV,
        aspect_ratio=1.0,
    )
    direction = np.asarray(_core_camera().grid_rays(width, height, NUMPY).direction)
    direction = direction.reshape(height, width, 3)

    for py in range(height):
        for px in range(width):
            ray = scalar.ray_for_pixel(px, py, width, height)
            expected = [ray.direction.x, ray.direction.y, ray.direction.z]
            assert np.allclose(direction[py, px], expected, atol=1e-12)


def test_grid_rays_agree_across_backends(backend: Backend) -> None:
    width, height = 7, 3
    reference = np.asarray(_core_camera().grid_rays(width, height, NUMPY).direction)
    other = np.asarray(_core_camera().grid_rays(width, height, backend).direction)
    assert np.allclose(reference, other, atol=1e-5)
