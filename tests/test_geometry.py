from __future__ import annotations

import math

import pytest

from myraytracer.geometry import Plane, Quad, Sphere
from myraytracer.material import Material
from myraytracer.ray import Ray
from myraytracer.vec import Vec3

MATERIAL = Material(albedo=Vec3(1, 0, 0))


def test_sphere_hit_returns_nearest_root() -> None:
    sphere = Sphere(center=Vec3(0, 0, -5), radius=1, material=MATERIAL)
    ray = Ray(origin=Vec3(0, 0, 0), direction=Vec3(0, 0, -1))

    hit = sphere.hit(ray, t_min=0.001, t_max=math.inf)

    assert hit is not None
    assert hit.t == pytest.approx(4.0)
    assert hit.point == Vec3(0, 0, -4)
    assert hit.normal == Vec3(0, 0, 1)
    assert hit.material is MATERIAL


def test_sphere_miss_returns_none() -> None:
    sphere = Sphere(center=Vec3(0, 0, -5), radius=1, material=MATERIAL)
    ray = Ray(origin=Vec3(0, 0, 0), direction=Vec3(0, 1, 0))

    assert sphere.hit(ray, t_min=0.001, t_max=math.inf) is None


def test_sphere_tangent_hit() -> None:
    sphere = Sphere(center=Vec3(0, 0, -5), radius=1, material=MATERIAL)
    ray = Ray(origin=Vec3(1, 0, 0), direction=Vec3(0, 0, -1))

    hit = sphere.hit(ray, t_min=0.001, t_max=math.inf)

    assert hit is not None
    assert hit.t == pytest.approx(5.0)
    assert hit.point == Vec3(1, 0, -5)
    assert hit.normal.x == pytest.approx(1.0)
    assert hit.normal.y == pytest.approx(0.0)
    assert hit.normal.z == pytest.approx(0.0)


def test_sphere_hit_outside_t_range_is_none() -> None:
    sphere = Sphere(center=Vec3(0, 0, -5), radius=1, material=MATERIAL)
    ray = Ray(origin=Vec3(0, 0, 0), direction=Vec3(0, 0, -1))

    assert sphere.hit(ray, t_min=0.001, t_max=3.0) is None


def test_sphere_rejects_non_positive_radius() -> None:
    with pytest.raises(ValueError):
        Sphere(center=Vec3(0, 0, 0), radius=0, material=MATERIAL)
    with pytest.raises(ValueError):
        Sphere(center=Vec3(0, 0, 0), radius=-1, material=MATERIAL)


def test_plane_hit_returns_expected_t() -> None:
    plane = Plane(point=Vec3(0, 0, -5), normal=Vec3(0, 0, 1), material=MATERIAL)
    ray = Ray(origin=Vec3(0, 0, 0), direction=Vec3(0, 0, -1))

    hit = plane.hit(ray, t_min=0.001, t_max=math.inf)

    assert hit is not None
    assert hit.t == pytest.approx(5.0)
    assert hit.point == Vec3(0, 0, -5)
    assert hit.normal == Vec3(0, 0, 1)
    assert hit.material is MATERIAL


def test_plane_miss_when_parallel_to_ray() -> None:
    plane = Plane(point=Vec3(0, 0, -5), normal=Vec3(0, 0, 1), material=MATERIAL)
    ray = Ray(origin=Vec3(0, 0, 0), direction=Vec3(1, 0, 0))

    assert plane.hit(ray, t_min=0.001, t_max=math.inf) is None


def test_plane_miss_when_intersection_outside_t_range() -> None:
    plane = Plane(point=Vec3(0, 0, -5), normal=Vec3(0, 0, 1), material=MATERIAL)
    ray = Ray(origin=Vec3(0, 0, 0), direction=Vec3(0, 0, 1))

    assert plane.hit(ray, t_min=0.001, t_max=math.inf) is None


def test_plane_rejects_near_zero_length_normal() -> None:
    with pytest.raises(ValueError):
        Plane(point=Vec3(0, 0, 0), normal=Vec3(0, 0, 0), material=MATERIAL)


def test_quad_hit_through_interior() -> None:
    quad = Quad(
        corner=Vec3(-1, -1, -5), edge1=Vec3(2, 0, 0), edge2=Vec3(0, 2, 0), material=MATERIAL
    )
    ray = Ray(origin=Vec3(0, 0, 0), direction=Vec3(0, 0, -1))

    hit = quad.hit(ray, t_min=0.001, t_max=math.inf)

    assert hit is not None
    assert hit.t == pytest.approx(5.0)
    assert hit.point == Vec3(0, 0, -5)
    assert hit.normal == Vec3(0, 0, 1)
    assert hit.material is MATERIAL


def test_quad_miss_when_hit_point_outside_parallelogram_bounds() -> None:
    quad = Quad(
        corner=Vec3(-1, -1, -5), edge1=Vec3(2, 0, 0), edge2=Vec3(0, 2, 0), material=MATERIAL
    )
    ray = Ray(origin=Vec3(3, 0, 0), direction=Vec3(0, 0, -1))

    assert quad.hit(ray, t_min=0.001, t_max=math.inf) is None


def test_quad_hit_at_parallelogram_boundary() -> None:
    quad = Quad(
        corner=Vec3(-1, -1, -5), edge1=Vec3(2, 0, 0), edge2=Vec3(0, 2, 0), material=MATERIAL
    )
    ray = Ray(origin=Vec3(1, 0, 0), direction=Vec3(0, 0, -1))

    hit = quad.hit(ray, t_min=0.001, t_max=math.inf)

    assert hit is not None
    assert hit.point == Vec3(1, 0, -5)


def test_quad_hit_outside_t_range_is_none() -> None:
    quad = Quad(
        corner=Vec3(-1, -1, -5), edge1=Vec3(2, 0, 0), edge2=Vec3(0, 2, 0), material=MATERIAL
    )
    ray = Ray(origin=Vec3(0, 0, 0), direction=Vec3(0, 0, -1))

    assert quad.hit(ray, t_min=0.001, t_max=3.0) is None


def test_quad_normal_faces_incoming_ray() -> None:
    quad = Quad(
        corner=Vec3(-1, -1, -5), edge1=Vec3(2, 0, 0), edge2=Vec3(0, 2, 0), material=MATERIAL
    )
    ray = Ray(origin=Vec3(0, 0, -10), direction=Vec3(0, 0, 1))

    hit = quad.hit(ray, t_min=0.001, t_max=math.inf)

    assert hit is not None
    assert hit.normal == Vec3(0, 0, -1)


def test_quad_rejects_parallel_edges() -> None:
    with pytest.raises(ValueError):
        Quad(corner=Vec3(0, 0, 0), edge1=Vec3(1, 0, 0), edge2=Vec3(2, 0, 0), material=MATERIAL)
