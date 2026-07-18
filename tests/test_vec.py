from __future__ import annotations

import pytest

from myraytracer.vec import Vec3


def test_add() -> None:
    assert Vec3(1, 2, 3) + Vec3(4, 5, 6) == Vec3(5, 7, 9)


def test_sub() -> None:
    assert Vec3(4, 5, 6) - Vec3(1, 2, 3) == Vec3(3, 3, 3)


def test_mul_scalar() -> None:
    assert Vec3(1, 2, 3) * 2 == Vec3(2, 4, 6)


def test_rmul_scalar() -> None:
    assert 2 * Vec3(1, 2, 3) == Vec3(2, 4, 6)


def test_dot() -> None:
    assert Vec3(1, 0, 0).dot(Vec3(0, 1, 0)) == 0
    assert Vec3(1, 2, 3).dot(Vec3(4, 5, 6)) == 32


def test_cross() -> None:
    assert Vec3(1, 0, 0).cross(Vec3(0, 1, 0)) == Vec3(0, 0, 1)


def test_length_squared() -> None:
    assert Vec3(3, 4, 0).length_squared() == 25.0


def test_length() -> None:
    assert Vec3(3, 4, 0).length() == 5.0


def test_normalized() -> None:
    normalized = Vec3(3, 4, 0).normalized()
    assert normalized.x == pytest.approx(0.6)
    assert normalized.y == pytest.approx(0.8)
    assert normalized.z == 0.0
    assert normalized.length() == pytest.approx(1.0)


def test_normalized_zero_vector_raises() -> None:
    with pytest.raises(ValueError):
        Vec3(0, 0, 0).normalized()
