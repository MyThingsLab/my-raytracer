from __future__ import annotations

import math

import pytest

from myraytracer.geometry import Plane, Quad, Sphere
from myraytracer.light import PointLight
from myraytracer.material import Material
from myraytracer.ray import Ray
from myraytracer.scene import Scene
from myraytracer.vec import Vec3

MATERIAL = Material(albedo=Vec3(1, 0, 0))


def test_point_light_rejects_negative_intensity_component() -> None:
    with pytest.raises(ValueError):
        PointLight(position=Vec3(0, 0, 0), intensity=Vec3(-1, 0, 0))


def test_nearest_hit_returns_closer_plane_in_front_of_sphere() -> None:
    sphere = Sphere(center=Vec3(0, 0, -10), radius=1, material=MATERIAL)
    plane = Plane(point=Vec3(0, 0, -5), normal=Vec3(0, 0, 1), material=MATERIAL)
    scene = Scene(objects=[sphere, plane], lights=[])
    ray = Ray(origin=Vec3(0, 0, 0), direction=Vec3(0, 0, -1))

    hit = scene.nearest_hit(ray, t_min=0.001, t_max=math.inf)

    assert hit is not None
    assert hit.t == pytest.approx(5.0)
    assert hit.material is MATERIAL


def test_nearest_hit_on_empty_scene_is_none() -> None:
    scene = Scene(objects=[], lights=[])
    ray = Ray(origin=Vec3(0, 0, 0), direction=Vec3(0, 0, -1))

    assert scene.nearest_hit(ray, t_min=0.001, t_max=math.inf) is None


def test_nearest_hit_outside_t_range_is_none() -> None:
    sphere = Sphere(center=Vec3(0, 0, -5), radius=1, material=MATERIAL)
    scene = Scene(objects=[sphere], lights=[])
    ray = Ray(origin=Vec3(0, 0, 0), direction=Vec3(0, 0, -1))

    assert scene.nearest_hit(ray, t_min=0.001, t_max=3.0) is None


def test_area_lights_returns_only_emissive_quads() -> None:
    emissive_material = Material(albedo=Vec3(0, 0, 0), emission=Vec3(5, 5, 5))
    emissive_quad = Quad(
        corner=Vec3(-1, -1, -5),
        edge1=Vec3(2, 0, 0),
        edge2=Vec3(0, 2, 0),
        material=emissive_material,
    )
    dark_quad = Quad(
        corner=Vec3(-1, -1, -6), edge1=Vec3(2, 0, 0), edge2=Vec3(0, 2, 0), material=MATERIAL
    )
    sphere = Sphere(center=Vec3(0, 0, -10), radius=1, material=emissive_material)
    point_light = PointLight(position=Vec3(0, 0, 0), intensity=Vec3(1, 1, 1))
    scene = Scene(objects=[emissive_quad, dark_quad, sphere], lights=[point_light])

    assert scene.area_lights() == [emissive_quad]


def test_area_lights_on_scene_with_no_quads_is_empty() -> None:
    sphere = Sphere(center=Vec3(0, 0, -5), radius=1, material=MATERIAL)
    scene = Scene(objects=[sphere], lights=[])

    assert scene.area_lights() == []
