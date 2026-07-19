from __future__ import annotations

import math

import numpy as np

from myraytracer.core.backend import Backend
from myraytracer.core.camera import Camera
from myraytracer.core.geometry import Plane, Quad, Sphere
from myraytracer.core.integrator import integrate
from myraytracer.core.integrator import render as render_image
from myraytracer.core.material import Material
from myraytracer.core.scene import PointLight, Scene

_INV_PI = 1.0 / math.pi


def _rays(backend: Backend, origin, direction, n: int):
    ro = backend.asarray([list(origin)] * n)
    rd = backend.asarray([list(direction)] * n)
    return ro, rd


def _integrate(scene, ro, rd, backend, *, max_depth=0, seed=0):
    radiance = integrate(
        scene, ro, rd, max_depth=max_depth, generator=backend.rng(seed), backend=backend
    )
    return np.asarray(radiance)


def test_direct_lambertian_matches_closed_form(backend: Backend) -> None:
    # A plane facing the camera lit head-on by a point light: at max_depth=0
    # there is no randomness, so direct lighting is exact.
    albedo = (0.8, 0.5, 0.2)
    scene = Scene(
        objects=[Plane(point=(0.0, 0.0, -5.0), normal=(0.0, 0.0, 1.0), material=Material(albedo))],
        lights=[PointLight(position=(0.0, 0.0, -2.0), intensity=(10.0, 10.0, 10.0))],
    )
    ro, rd = _rays(backend, (0.0, 0.0, 0.0), (0.0, 0.0, -1.0), 1)
    result = _integrate(scene, ro, rd, backend)

    distance = 3.0
    # Lambertian BRDF albedo/pi: Lo = (albedo/pi) * I * cos / dist^2.
    expected = np.array(albedo) * _INV_PI * (10.0 * 1.0 / distance**2)
    assert np.allclose(result[0], expected, atol=1e-5)


def test_miss_returns_black(backend: Backend) -> None:
    scene = Scene(objects=[], lights=[])
    ro, rd = _rays(backend, (0.0, 0.0, 0.0), (0.0, 0.0, -1.0), 1)
    result = _integrate(scene, ro, rd, backend, max_depth=1)
    assert np.allclose(result[0], 0.0)


def test_emission_adds_to_direct_at_depth_zero(backend: Backend) -> None:
    albedo = (0.8, 0.5, 0.2)
    emission = (0.1, 0.1, 0.1)
    material = Material(albedo, emission)
    scene = Scene(
        objects=[Plane(point=(0.0, 0.0, -5.0), normal=(0.0, 0.0, 1.0), material=material)],
        lights=[PointLight(position=(0.0, 0.0, -2.0), intensity=(10.0, 10.0, 10.0))],
    )
    ro, rd = _rays(backend, (0.0, 0.0, 0.0), (0.0, 0.0, -1.0), 1)
    result = _integrate(scene, ro, rd, backend)

    expected = np.array(emission) + np.array(albedo) * _INV_PI * (10.0 * 1.0 / 3.0**2)
    assert np.allclose(result[0], expected, atol=1e-5)


def test_blocker_zeroes_point_light(backend: Backend) -> None:
    white = Material((1.0, 1.0, 1.0))
    scene = Scene(
        objects=[
            Plane(point=(0.0, 0.0, -5.0), normal=(0.0, 0.0, 1.0), material=white),
            Sphere(center=(0.5, 0.0, -3.5), radius=0.3, material=white),  # on the shadow segment
        ],
        lights=[PointLight(position=(1.0, 0.0, -2.0), intensity=(10.0, 10.0, 10.0))],
    )
    ro, rd = _rays(backend, (0.0, 0.0, 0.0), (0.0, 0.0, -1.0), 1)
    result = _integrate(scene, ro, rd, backend)
    assert np.allclose(result[0], 0.0)


def test_area_light_nee_matches_solid_angle(backend: Backend) -> None:
    albedo = (0.8, 0.5, 0.2)
    corner = np.array([-0.05, -0.05, -2.0])
    edge1 = np.array([0.1, 0.0, 0.0])
    edge2 = np.array([0.0, -0.1, 0.0])
    emission = np.array([10.0, 10.0, 10.0])
    scene = Scene(
        objects=[
            Plane(point=(0.0, 0.0, -5.0), normal=(0.0, 0.0, 1.0), material=Material(albedo)),
            Quad(
                corner=tuple(corner),
                edge1=tuple(edge1),
                edge2=tuple(edge2),
                material=Material((0.0, 0.0, 0.0), tuple(emission)),
            ),
        ],
        lights=[],
    )
    n = 4000
    ro, rd = _rays(backend, (0.3, 0.0, 0.0), (0.0, 0.0, -1.0), n)
    samples = _integrate(scene, ro, rd, backend)
    result = samples.mean(axis=0)

    hit_point = np.array([0.3, 0.0, -5.0])
    light_center = corner + edge1 * 0.5 + edge2 * 0.5
    to_light = light_center - hit_point
    distance = np.linalg.norm(to_light)
    light_dir = to_light / distance
    cos_surface = np.array([0.0, 0.0, 1.0]) @ light_dir
    light_normal = np.cross(edge1, edge2)
    light_normal = light_normal / np.linalg.norm(light_normal)
    cos_light = light_normal @ (-light_dir)
    area = np.linalg.norm(np.cross(edge1, edge2))
    # Physically-correct Lambertian NEE (albedo/pi). The MIS light weight is
    # ~1 here (a tiny, distant light has a far larger light pdf than the BSDF
    # pdf), so it does not shift the expectation within tolerance.
    expected = np.array(albedo) * _INV_PI * emission * cos_surface * cos_light * area / distance**2
    assert np.allclose(result, expected, rtol=0.1, atol=1e-4)


def test_occluded_area_light_is_black(backend: Backend) -> None:
    white = Material((1.0, 1.0, 1.0))
    scene = Scene(
        objects=[
            Plane(point=(0.0, 0.0, -5.0), normal=(0.0, 0.0, 1.0), material=white),
            Quad(
                corner=(0.45, -0.05, -2.0),
                edge1=(0.1, 0.0, 0.0),
                edge2=(0.0, -0.1, 0.0),
                material=Material((0.0, 0.0, 0.0), (10.0, 10.0, 10.0)),
            ),
            Sphere(center=(0.3, 0.0, -3.2), radius=0.25, material=white),
        ],
        lights=[],
    )
    ro, rd = _rays(backend, (0.0, 0.0, 0.0), (0.0, 0.0, -1.0), 16)
    result = _integrate(scene, ro, rd, backend)
    assert np.allclose(result, 0.0)


def _camera() -> Camera:
    return Camera(
        origin=(0.0, 0.0, 0.0),
        look_at=(0.0, 0.0, -1.0),
        up=(0.0, 1.0, 0.0),
        vfov_degrees=60.0,
        aspect_ratio=1.0,
    )


def test_render_shape_and_determinism(backend: Backend) -> None:
    scene = Scene(
        objects=[Sphere(center=(0.0, 0.0, -3.0), radius=1.0, material=Material((0.7, 0.3, 0.3)))],
        lights=[PointLight(position=(2.0, 2.0, 0.0), intensity=(15.0, 15.0, 15.0))],
    )
    kwargs = dict(width=8, height=8, spp=4, max_depth=2, seed=3, backend=backend)
    first = np.asarray(render_image(scene, _camera(), **kwargs))
    second = np.asarray(render_image(scene, _camera(), **kwargs))

    assert first.shape == (8, 8, 3)
    assert np.array_equal(first, second)
    assert first.max() > 0.0
