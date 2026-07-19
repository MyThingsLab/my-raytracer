from __future__ import annotations

import numpy as np

from myraytracer.core.backend import Backend
from myraytracer.core.ray import RayBatch


def test_ray_batch_at(backend: Backend) -> None:
    origin = backend.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    direction = backend.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    rays = RayBatch(origin=origin, direction=direction)

    t = backend.asarray([2.0, 3.0])
    points = np.asarray(rays.at(t))
    assert np.allclose(points, [[2.0, 0.0, 0.0], [1.0, 4.0, 1.0]])
