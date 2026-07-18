from __future__ import annotations

import math

import numpy as np
import pytest

from myraytracer.camera import Camera
from myraytracer.geometry import Plane, Quad, Sphere
from myraytracer.light import PointLight
from myraytracer.material import Material
from myraytracer.ray import Ray
from myraytracer.scene import Scene
from myraytracer.tracer import render_pixel, trace_ray
from myraytracer.vec import Vec3


def _camera() -> Camera:
    return Camera(
        origin=Vec3(0, 0, 0),
        look_at=Vec3(0, 0, -1),
        up=Vec3(0, 1, 0),
        vfov_degrees=90,
        aspect_ratio=1,
    )


def test_render_pixel_matches_analytic_lambertian_direct_lighting() -> None:
    # Plane facing the camera at z=-5, lit by a point light directly along
    # its normal at z=-2 (distance 3 from the hit point), no bounce
    # contribution reaches the camera since the plane is the only object.
    albedo = Vec3(0.8, 0.5, 0.2)
    material = Material(albedo=albedo)
    plane = Plane(point=Vec3(0, 0, -5), normal=Vec3(0, 0, 1), material=material)
    light = PointLight(position=Vec3(0, 0, -2), intensity=Vec3(10, 10, 10))
    scene = Scene(objects=[plane], lights=[light])
    camera = _camera()
    rng = np.random.Generator(np.random.PCG64(0))

    result = render_pixel(
        camera, scene, px=50, py=50, width=100, height=100, rng=rng, spp=256, max_depth=1
    )

    distance = 3.0
    cos_theta = 1.0
    expected = albedo * (10.0 * cos_theta / distance**2)

    assert result.x == pytest.approx(expected.x, rel=0.05)
    assert result.y == pytest.approx(expected.y, rel=0.05)
    assert result.z == pytest.approx(expected.z, rel=0.05)


def test_trace_ray_returns_background_on_miss() -> None:
    scene = Scene(objects=[], lights=[])
    ray = Ray(origin=Vec3(0, 0, 0), direction=Vec3(0, 0, -1))
    rng = np.random.Generator(np.random.PCG64(0))

    result = trace_ray(ray, scene, rng, max_depth=1)

    assert result == Vec3(0, 0, 0)


def test_blocking_object_zeroes_light_contribution() -> None:
    # The blocker sits on the shadow segment from the plane hit point
    # (0, 0, -5) to the light (1, 0, -2), but is offset far enough from the
    # primary ray's line (x=0, y=0) that it doesn't shadow the primary hit
    # itself.
    material = Material(albedo=Vec3(1, 1, 1))
    plane = Plane(point=Vec3(0, 0, -5), normal=Vec3(0, 0, 1), material=material)
    blocker = Sphere(center=Vec3(0.5, 0, -3.5), radius=0.3, material=material)
    light = PointLight(position=Vec3(1, 0, -2), intensity=Vec3(10, 10, 10))
    scene = Scene(objects=[plane, blocker], lights=[light])
    ray = Ray(origin=Vec3(0, 0, 0), direction=Vec3(0, 0, -1))
    rng = np.random.Generator(np.random.PCG64(0))

    result = trace_ray(ray, scene, rng, max_depth=0)

    assert result == Vec3(0, 0, 0)


def test_render_pixel_is_deterministic_given_same_seed() -> None:
    material = Material(albedo=Vec3(0.8, 0.5, 0.2))
    plane = Plane(point=Vec3(0, 0, -5), normal=Vec3(0, 0, 1), material=material)
    light = PointLight(position=Vec3(0, 0, -2), intensity=Vec3(10, 10, 10))
    scene = Scene(objects=[plane], lights=[light])
    camera = _camera()

    rng_a = np.random.Generator(np.random.PCG64(42))
    result_a = render_pixel(
        camera, scene, px=50, py=50, width=100, height=100, rng=rng_a, spp=16, max_depth=2
    )

    rng_b = np.random.Generator(np.random.PCG64(42))
    result_b = render_pixel(
        camera, scene, px=50, py=50, width=100, height=100, rng=rng_b, spp=16, max_depth=2
    )

    assert result_a == result_b


def test_max_depth_zero_returns_direct_lighting_with_no_recursion() -> None:
    albedo = Vec3(0.8, 0.5, 0.2)
    material = Material(albedo=albedo, emission=Vec3(0.1, 0.1, 0.1))
    plane = Plane(point=Vec3(0, 0, -5), normal=Vec3(0, 0, 1), material=material)
    light = PointLight(position=Vec3(0, 0, -2), intensity=Vec3(10, 10, 10))
    scene = Scene(objects=[plane], lights=[light])
    ray = Ray(origin=Vec3(0, 0, 0), direction=Vec3(0, 0, -1))
    rng = np.random.Generator(np.random.PCG64(0))

    result = trace_ray(ray, scene, rng, max_depth=0)

    distance = 3.0
    cos_theta = 1.0
    expected = material.emission + albedo * (10.0 * cos_theta / distance**2)

    assert result.x == pytest.approx(expected.x)
    assert result.y == pytest.approx(expected.y)
    assert result.z == pytest.approx(expected.z)


def test_max_depth_zero_never_calls_rng() -> None:
    # No recursion or roulette should happen at max_depth=0, so the RNG
    # state must be untouched -- verified by checking the same generator
    # instance yields the same next value regardless of tracing.
    material = Material(albedo=Vec3(1, 1, 1))
    plane = Plane(point=Vec3(0, 0, -5), normal=Vec3(0, 0, 1), material=material)
    scene = Scene(objects=[plane], lights=[])
    ray = Ray(origin=Vec3(0, 0, 0), direction=Vec3(0, 0, -1))

    rng_traced = np.random.Generator(np.random.PCG64(7))
    trace_ray(ray, scene, rng_traced, max_depth=0)
    next_after_trace = rng_traced.random()

    rng_untouched = np.random.Generator(np.random.PCG64(7))
    next_without_trace = rng_untouched.random()

    assert next_after_trace == next_without_trace


def test_sphere_geometry_still_hit_correctly_within_scene() -> None:
    # Sanity check that trace_ray works with Sphere geometry too, not just
    # Plane, using math.inf-bounded t_max the same way Scene.nearest_hit does.
    material = Material(albedo=Vec3(1, 1, 1), emission=Vec3(0.3, 0.3, 0.3))
    sphere = Sphere(center=Vec3(0, 0, -5), radius=1, material=material)
    scene = Scene(objects=[sphere], lights=[])
    ray = Ray(origin=Vec3(0, 0, 0), direction=Vec3(0, 0, -1))
    rng = np.random.Generator(np.random.PCG64(0))

    result = trace_ray(ray, scene, rng, max_depth=0)

    assert result == material.emission
    assert not math.isinf(result.x)


def test_area_light_nee_matches_closed_form_solid_angle() -> None:
    # Ray offset in x so the primary ray hits the plane but does not pass
    # through the small light quad centered on the z-axis at z=-2.
    albedo = Vec3(0.8, 0.5, 0.2)
    material = Material(albedo=albedo)
    plane = Plane(point=Vec3(0, 0, -5), normal=Vec3(0, 0, 1), material=material)

    light_corner = Vec3(-0.05, -0.05, -2)
    light_edge1 = Vec3(0.1, 0, 0)
    light_edge2 = Vec3(0, -0.1, 0)
    light_emission = Vec3(10, 10, 10)
    light_material = Material(albedo=Vec3(0, 0, 0), emission=light_emission)
    light_quad = Quad(
        corner=light_corner, edge1=light_edge1, edge2=light_edge2, material=light_material
    )
    light_center = light_corner + light_edge1 * 0.5 + light_edge2 * 0.5

    scene = Scene(objects=[plane, light_quad], lights=[])
    ray = Ray(origin=Vec3(0.3, 0, 0), direction=Vec3(0, 0, -1))
    rng = np.random.Generator(np.random.PCG64(0))

    samples = 4000
    accumulator = Vec3(0, 0, 0)
    for _ in range(samples):
        accumulator = accumulator + trace_ray(ray, scene, rng, max_depth=0)
    result = accumulator * (1.0 / samples)

    hit_point = Vec3(0.3, 0, -5)
    to_light = light_center - hit_point
    distance = to_light.length()
    light_dir = to_light * (1.0 / distance)
    cos_theta_surface = Vec3(0, 0, 1).dot(light_dir)
    light_normal = light_edge1.cross(light_edge2).normalized()
    cos_theta_light = light_normal.dot(light_dir * -1.0)
    area = light_edge1.cross(light_edge2).length()
    expected = albedo * (
        light_emission.x * cos_theta_surface * cos_theta_light * area / distance**2
    )

    assert result.x == pytest.approx(expected.x, rel=0.1, abs=1e-4)
    assert result.y == pytest.approx(expected.y, rel=0.1, abs=1e-4)
    assert result.z == pytest.approx(expected.z, rel=0.1, abs=1e-4)


def test_occluded_area_light_zeroes_nee_contribution() -> None:
    material = Material(albedo=Vec3(1, 1, 1))
    plane = Plane(point=Vec3(0, 0, -5), normal=Vec3(0, 0, 1), material=material)

    light_material = Material(albedo=Vec3(0, 0, 0), emission=Vec3(10, 10, 10))
    light_quad = Quad(
        corner=Vec3(0.45, -0.05, -2),
        edge1=Vec3(0.1, 0, 0),
        edge2=Vec3(0, -0.1, 0),
        material=light_material,
    )
    blocker = Sphere(center=Vec3(0.3, 0, -3.2), radius=0.25, material=material)

    scene = Scene(objects=[plane, light_quad, blocker], lights=[])
    ray = Ray(origin=Vec3(0, 0, 0), direction=Vec3(0, 0, -1))
    rng = np.random.Generator(np.random.PCG64(0))

    result = trace_ray(ray, scene, rng, max_depth=0)

    assert result == Vec3(0, 0, 0)


def test_scene_with_no_area_lights_behaves_as_before() -> None:
    albedo = Vec3(0.8, 0.5, 0.2)
    material = Material(albedo=albedo)
    plane = Plane(point=Vec3(0, 0, -5), normal=Vec3(0, 0, 1), material=material)
    light = PointLight(position=Vec3(0, 0, -2), intensity=Vec3(10, 10, 10))
    scene = Scene(objects=[plane], lights=[light])
    ray = Ray(origin=Vec3(0, 0, 0), direction=Vec3(0, 0, -1))
    rng = np.random.Generator(np.random.PCG64(0))

    result = trace_ray(ray, scene, rng, max_depth=0)

    distance = 3.0
    cos_theta = 1.0
    expected = albedo * (10.0 * cos_theta / distance**2)

    assert result.x == pytest.approx(expected.x)
    assert result.y == pytest.approx(expected.y)
    assert result.z == pytest.approx(expected.z)
