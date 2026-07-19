from __future__ import annotations

import math

import numpy as np

from myraytracer.core.backend import NUMPY, Backend
from myraytracer.core.camera import Camera
from myraytracer.core.geometry import Plane, Quad, Sphere, hit_primitive
from myraytracer.core.material import Material
from myraytracer.geometry import Plane as ScalarPlane
from myraytracer.geometry import Quad as ScalarQuad
from myraytracer.geometry import Sphere as ScalarSphere
from myraytracer.material import Material as ScalarMaterial
from myraytracer.ray import Ray
from myraytracer.vec import Vec3

_MAT = Material(albedo=(0.7, 0.7, 0.7))
_SCALAR_MAT = ScalarMaterial(albedo=Vec3(0.7, 0.7, 0.7))
_T_MIN = 1e-4


def _rays(backend: Backend, width: int = 8, height: int = 8):
    camera = Camera(
        origin=(0.0, 0.0, 0.0),
        look_at=(0.0, 0.0, -1.0),
        up=(0.0, 1.0, 0.0),
        vfov_degrees=70.0,
        aspect_ratio=1.0,
    )
    rays = camera.grid_rays(width, height, backend)
    return np.asarray(rays.origin), np.asarray(rays.direction)


def _assert_matches_scalar(hit, origins, directions, scalar_obj) -> None:
    hit_mask = np.asarray(hit.hit)
    t = np.asarray(hit.t)
    point = np.asarray(hit.point)
    normal = np.asarray(hit.normal)

    for i in range(origins.shape[0]):
        ray = Ray(origin=Vec3(*origins[i]), direction=Vec3(*directions[i]))
        expected = scalar_obj.hit(ray, _T_MIN, math.inf)
        assert bool(hit_mask[i]) == (expected is not None), f"hit mask mismatch at {i}"
        if expected is None:
            continue
        assert np.allclose(t[i], expected.t, atol=1e-4)
        expected_point = [expected.point.x, expected.point.y, expected.point.z]
        expected_normal = [expected.normal.x, expected.normal.y, expected.normal.z]
        assert np.allclose(point[i], expected_point, atol=1e-4)
        assert np.allclose(normal[i], expected_normal, atol=1e-4)


def test_sphere_matches_scalar(backend: Backend) -> None:
    origins, directions = _rays(backend)
    ro = backend.asarray(origins.tolist())
    rd = backend.asarray(directions.tolist())
    core = Sphere(center=(0.0, 0.0, -3.0), radius=0.9, material=_MAT)
    scalar = ScalarSphere(center=Vec3(0.0, 0.0, -3.0), radius=0.9, material=_SCALAR_MAT)
    _assert_matches_scalar(
        hit_primitive(core, ro, rd, _T_MIN, math.inf, backend), origins, directions, scalar
    )


def test_plane_matches_scalar(backend: Backend) -> None:
    origins, directions = _rays(backend)
    ro = backend.asarray(origins.tolist())
    rd = backend.asarray(directions.tolist())
    core = Plane(point=(0.0, -1.0, 0.0), normal=(0.0, 1.0, 0.0), material=_MAT)
    scalar = ScalarPlane(
        point=Vec3(0.0, -1.0, 0.0), normal=Vec3(0.0, 1.0, 0.0), material=_SCALAR_MAT
    )
    _assert_matches_scalar(
        hit_primitive(core, ro, rd, _T_MIN, math.inf, backend), origins, directions, scalar
    )


def test_quad_matches_scalar(backend: Backend) -> None:
    origins, directions = _rays(backend)
    ro = backend.asarray(origins.tolist())
    rd = backend.asarray(directions.tolist())
    core = Quad(
        corner=(-1.0, -1.0, -3.0), edge1=(2.0, 0.0, 0.0), edge2=(0.0, 2.0, 0.0), material=_MAT
    )
    scalar = ScalarQuad(
        corner=Vec3(-1.0, -1.0, -3.0),
        edge1=Vec3(2.0, 0.0, 0.0),
        edge2=Vec3(0.0, 2.0, 0.0),
        material=_SCALAR_MAT,
    )
    _assert_matches_scalar(
        hit_primitive(core, ro, rd, _T_MIN, math.inf, backend), origins, directions, scalar
    )


def test_sphere_agrees_across_backends(backend: Backend) -> None:
    origins, directions = _rays(NUMPY)
    core = Sphere(center=(0.2, -0.1, -3.0), radius=0.9, material=_MAT)

    ref = hit_primitive(
        core, NUMPY.asarray(origins.tolist()), NUMPY.asarray(directions.tolist()),
        _T_MIN, math.inf, NUMPY,
    )
    other = hit_primitive(
        core, backend.asarray(origins.tolist()), backend.asarray(directions.tolist()),
        _T_MIN, math.inf, backend,
    )
    assert np.array_equal(np.asarray(ref.hit), np.asarray(other.hit))
    hit_mask = np.asarray(ref.hit)
    ref_point = np.asarray(ref.point)[hit_mask]
    other_point = np.asarray(other.point)[hit_mask]
    assert np.allclose(ref_point, other_point, atol=1e-4)
