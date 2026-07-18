from __future__ import annotations

import math
import time

import pytest

from myraytracer.bvh import BVH, bounding_box
from myraytracer.geometry import Plane, Quad, Sphere
from myraytracer.material import Material
from myraytracer.ray import Ray
from myraytracer.scene import Scene
from myraytracer.vec import Vec3

MATERIAL = Material(albedo=Vec3(1, 0, 0))


def _mixed_scene() -> Scene:
    objects = [
        Sphere(center=Vec3(0, 0, -5), radius=1, material=MATERIAL),
        Sphere(center=Vec3(3, 0, -6), radius=1, material=MATERIAL),
        Sphere(center=Vec3(-3, 0, -6), radius=1, material=MATERIAL),
        Quad(
            corner=Vec3(-1, -1, -8), edge1=Vec3(2, 0, 0), edge2=Vec3(0, 2, 0), material=MATERIAL
        ),
        Sphere(center=Vec3(0, 3, -10), radius=1.5, material=MATERIAL),
        Plane(point=Vec3(0, -2, 0), normal=Vec3(0, 1, 0), material=MATERIAL),
    ]
    return Scene(objects=objects, lights=[])


_RAYS = [
    Ray(origin=Vec3(0, 0, 0), direction=Vec3(0, 0, -1)),  # hits nearest sphere
    Ray(origin=Vec3(3, 0, 0), direction=Vec3(0, 0, -1)),  # hits side sphere
    Ray(origin=Vec3(-3, 0, 0), direction=Vec3(0, 0, -1)),  # hits other side sphere
    Ray(origin=Vec3(0, 0, 0), direction=Vec3(0, 0, 1)),  # miss, points away
    Ray(origin=Vec3(0, 3, 0), direction=Vec3(0, 0, -1)),  # hits far sphere at y=3
    Ray(origin=Vec3(0.99, 0, 0), direction=Vec3(0, 0, -1)),  # grazes nearest sphere
    Ray(origin=Vec3(1.5, 1.5, 0), direction=Vec3(0, 0, -1)),  # hits quad through gap
    Ray(origin=Vec3(0, 10, 0), direction=Vec3(0, -1, 0)),  # hits the unbounded plane
    Ray(origin=Vec3(100, 100, 100), direction=Vec3(1, 0, 0)),  # total miss
]


def _assert_hits_match(scene: Scene, ray: Ray) -> None:
    linear = scene.nearest_hit(ray, t_min=0.001, t_max=math.inf, use_bvh=False)
    accelerated = scene.nearest_hit(ray, t_min=0.001, t_max=math.inf, use_bvh=True)

    if linear is None:
        assert accelerated is None
        return

    assert accelerated is not None
    assert accelerated.t == pytest.approx(linear.t)
    assert accelerated.point == linear.point
    assert accelerated.normal == linear.normal
    assert accelerated.material is linear.material


def test_bvh_nearest_hit_matches_linear_scan_for_hits_and_misses() -> None:
    scene = _mixed_scene()
    for ray in _RAYS:
        _assert_hits_match(scene, ray)


def test_bvh_still_hits_plane_excluded_from_the_tree() -> None:
    scene = _mixed_scene()
    ray = Ray(origin=Vec3(0, 10, 0), direction=Vec3(0, -1, 0))

    hit = scene.nearest_hit(ray, t_min=0.001, t_max=math.inf, use_bvh=True)

    assert hit is not None
    assert hit.point == Vec3(0, -2, 0)


def test_bvh_on_empty_scene_is_none() -> None:
    scene = Scene(objects=[], lights=[])
    ray = Ray(origin=Vec3(0, 0, 0), direction=Vec3(0, 0, -1))

    assert scene.nearest_hit(ray, t_min=0.001, t_max=math.inf, use_bvh=True) is None


def test_bvh_on_single_primitive_scene_matches_linear() -> None:
    scene = Scene(objects=[Sphere(center=Vec3(0, 0, -5), radius=1, material=MATERIAL)], lights=[])
    ray = Ray(origin=Vec3(0, 0, 0), direction=Vec3(0, 0, -1))

    _assert_hits_match(scene, ray)


def test_bounding_box_of_plane_is_none() -> None:
    plane = Plane(point=Vec3(0, 0, -5), normal=Vec3(0, 0, 1), material=MATERIAL)

    assert bounding_box(plane) is None


def test_bvh_traversal_visits_near_child_first_along_negative_axis() -> None:
    # Enough spheres spread along x to force an internal SAH split; a ray
    # travelling in -x should reach the right-hand spheres (higher centroid)
    # first, exercising the near/far child swap for negative-direction rays.
    objects = [
        Sphere(center=Vec3(float(x), 0, -5), radius=1, material=MATERIAL)
        for x in (-9, -6, -3, 3, 6, 9)
    ]
    scene = Scene(objects=objects, lights=[])
    ray = Ray(origin=Vec3(20, 0, -5), direction=Vec3(-1, 0, 0))

    _assert_hits_match(scene, ray)


def test_bvh_build_on_all_unbounded_scene_has_no_root() -> None:
    objects = [Plane(point=Vec3(0, 0, -5), normal=Vec3(0, 0, 1), material=MATERIAL)]
    bvh = BVH.build(objects)

    assert bvh._root is None
    assert bvh.bounded == []
    assert len(bvh.unbounded) == 1


@pytest.mark.slow
def test_bvh_nearest_hit_is_faster_than_linear_scan_at_high_primitive_count() -> None:
    objects = [
        Sphere(
            center=Vec3(float(i % 50) * 3, float(i // 50) * 3, -20 - i),
            radius=0.5,
            material=MATERIAL,
        )
        for i in range(2000)
    ]
    scene = Scene(objects=objects, lights=[])
    ray = Ray(origin=Vec3(0, 0, 0), direction=Vec3(0, 0, -1))
    # Force the BVH to build before timing so the comparison is apples-to-apples.
    scene.nearest_hit(ray, t_min=0.001, t_max=math.inf, use_bvh=True)

    linear_start = time.perf_counter()
    for _ in range(50):
        scene.nearest_hit(ray, t_min=0.001, t_max=math.inf, use_bvh=False)
    linear_elapsed = time.perf_counter() - linear_start

    bvh_start = time.perf_counter()
    for _ in range(50):
        scene.nearest_hit(ray, t_min=0.001, t_max=math.inf, use_bvh=True)
    bvh_elapsed = time.perf_counter() - bvh_start

    assert bvh_elapsed < linear_elapsed
