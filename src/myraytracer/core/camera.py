from __future__ import annotations

import math
from dataclasses import dataclass

from myraytracer.core.backend import Array, Backend
from myraytracer.core.linalg import cross, normalize
from myraytracer.core.ray import RayBatch


@dataclass(frozen=True)
class Camera:
    """A pinhole camera stored in plain Python floats, backend-agnostic.

    `origin`/`look_at`/`up` are 3-tuples; the backend arrays are built at ray
    time from whichever Backend the caller passes, so one Camera drives both
    the numpy and torch paths. The ray math and the integer-pixel sampling
    convention match both the scalar Camera.ray_for_pixel and gpu.camera, so
    this reproduces either exactly.
    """

    origin: tuple[float, float, float]
    look_at: tuple[float, float, float]
    up: tuple[float, float, float]
    vfov_degrees: float
    aspect_ratio: float

    def rays(
        self,
        px: Array,
        py: Array,
        *,
        width: int,
        height: int,
        backend: Backend,
    ) -> RayBatch:
        # px/py are arrays of continuous sample coordinates (shape (N,)); the
        # caller owns any sub-pixel jitter, exactly as the scalar render loop
        # jitters around integer pixel coordinates.
        origin = backend.asarray(self.origin)
        look_at = backend.asarray(self.look_at)
        up = backend.asarray(self.up)

        forward = normalize(look_at - origin)
        right = normalize(cross(forward, up))
        true_up = cross(right, forward)

        half_height = math.tan(math.radians(self.vfov_degrees) / 2)
        half_width = half_height * self.aspect_ratio

        u = (2 * px / width - 1) * half_width
        v = (1 - 2 * py / height) * half_height

        direction = forward + right * u[..., None] + true_up * v[..., None]
        direction = normalize(direction)
        origins = backend.broadcast_to(origin, direction.shape)
        return RayBatch(origin=origins, direction=direction)

    def grid_rays(self, width: int, height: int, backend: Backend) -> RayBatch:
        # One ray per pixel, row-major (H*W,), sampling integer pixel
        # coordinates -- the un-jittered grid matching gpu.camera.rays.
        ys, xs = backend.meshgrid(backend.arange(height), backend.arange(width))
        return self.rays(
            xs.reshape(-1), ys.reshape(-1), width=width, height=height, backend=backend
        )
