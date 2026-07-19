from __future__ import annotations

import numpy as np
import pytest

from myraytracer.core.backend import NUMPY, Backend, backend_of, get_backend


def test_get_backend_aliases() -> None:
    assert get_backend("cpu") is NUMPY
    assert get_backend("numpy") is NUMPY


def test_get_backend_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        get_backend("quantum")


def test_backend_of_matches_array_type(backend: Backend) -> None:
    x = backend.asarray([1.0, 2.0, 3.0])
    assert backend_of(x).name == backend.name


def test_backend_of_rejects_non_array() -> None:
    with pytest.raises(TypeError):
        backend_of([1.0, 2.0, 3.0])


def test_cross_matches_numpy_reference(backend: Backend) -> None:
    a = backend.asarray([1.0, 0.0, 0.0])
    b = backend.asarray([0.0, 1.0, 0.0])
    assert np.allclose(np.asarray(backend.cross(a, b)), [0.0, 0.0, 1.0])


def test_arange_and_broadcast_shapes(backend: Backend) -> None:
    assert np.asarray(backend.arange(4)).tolist() == [0, 1, 2, 3]
    origin = backend.asarray([1.0, 2.0, 3.0])
    broadcast = backend.broadcast_to(origin, (5, 3))
    assert np.asarray(broadcast).shape == (5, 3)
    assert np.allclose(np.asarray(broadcast)[0], [1.0, 2.0, 3.0])
