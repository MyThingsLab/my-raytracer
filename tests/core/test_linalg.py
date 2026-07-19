from __future__ import annotations

import numpy as np
import pytest

from myraytracer.core import linalg
from myraytracer.core.backend import Backend
from myraytracer.vec import Vec3


def test_dot_and_cross_match_scalar_vec3(backend: Backend) -> None:
    va, vb = Vec3(1.0, 2.0, 3.0), Vec3(-2.0, 0.5, 4.0)
    a = backend.asarray([va.x, va.y, va.z])
    b = backend.asarray([vb.x, vb.y, vb.z])

    assert np.allclose(float(np.asarray(linalg.dot(a, b))), va.dot(vb))
    expected = va.cross(vb)
    assert np.allclose(np.asarray(linalg.cross(a, b)), [expected.x, expected.y, expected.z])


def test_length_and_normalize(backend: Backend) -> None:
    v = backend.asarray([3.0, 4.0, 0.0])
    assert np.allclose(float(np.asarray(linalg.length(v))), 5.0)
    unit = np.asarray(linalg.normalize(v))
    assert np.allclose(unit, [0.6, 0.8, 0.0])
    assert np.allclose(np.linalg.norm(unit), 1.0)


def test_batched_dot_reduces_last_axis(backend: Backend) -> None:
    a = backend.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    b = backend.asarray([[1.0, 1.0, 1.0], [0.0, 3.0, 0.0]])
    assert np.allclose(np.asarray(linalg.dot(a, b)), [1.0, 6.0])


def test_normalize_rejects_zero_length(backend: Backend) -> None:
    v = backend.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="zero-length"):
        linalg.normalize(v)
