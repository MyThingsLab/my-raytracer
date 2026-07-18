from myraytracer.ray import Ray
from myraytracer.vec import Vec3


def test_at_origin():
    ray = Ray(origin=Vec3(0, 0, 0), direction=Vec3(1, 0, 0))
    assert ray.at(0) == Vec3(0, 0, 0)


def test_at_moves_along_direction():
    ray = Ray(origin=Vec3(1, 1, 1), direction=Vec3(0, 1, 0))
    assert ray.at(3) == Vec3(1, 4, 1)


def test_at_negative_t():
    ray = Ray(origin=Vec3(0, 0, 0), direction=Vec3(1, 0, 0))
    assert ray.at(-2) == Vec3(-2, 0, 0)
