from __future__ import annotations

import math

import numpy as np

from myraytracer.core.backend import Backend
from myraytracer.core.geometry import Quad, Sphere
from myraytracer.core.material import Material
from myraytracer.core.scene import PointLight, Scene

_FRONT = Material(albedo=(1.0, 0.0, 0.0))
_BACK = Material(albedo=(0.0, 0.0, 1.0))
_T_MIN = 1e-4


def _scene() -> Scene:
    return Scene(
        objects=[
            Sphere(center=(0.0, 0.0, -5.0), radius=0.5, material=_BACK),
            Sphere(center=(0.0, 0.0, -2.0), radius=0.5, material=_FRONT),
        ],
        lights=[PointLight(position=(0.0, 2.0, 0.0), intensity=(1.0, 1.0, 1.0))],
    )


def test_nearest_hit_picks_closest_and_returns_its_material(backend: Backend) -> None:
    # Two spheres on the -z axis; a forward ray must hit the near one and pick
    # up *its* albedo (red), proving per-object materials.
    origin = backend.asarray([[0.0, 0.0, 0.0]])
    direction = backend.asarray([[0.0, 0.0, -1.0]])
    hit = _scene().nearest_hit(origin, direction, _T_MIN, math.inf, backend)

    assert bool(np.asarray(hit.hit)[0])
    assert np.allclose(np.asarray(hit.t)[0], 1.5, atol=1e-4)  # near sphere front face
    assert np.allclose(np.asarray(hit.albedo)[0], [1.0, 0.0, 0.0], atol=1e-5)


def test_miss_returns_no_hit_and_zero_material(backend: Backend) -> None:
    origin = backend.asarray([[0.0, 0.0, 0.0]])
    direction = backend.asarray([[0.0, 0.0, 1.0]])  # away from both spheres
    hit = _scene().nearest_hit(origin, direction, _T_MIN, math.inf, backend)

    assert not bool(np.asarray(hit.hit)[0])
    assert np.allclose(np.asarray(hit.albedo)[0], [0.0, 0.0, 0.0])
    assert np.allclose(np.asarray(hit.emission)[0], [0.0, 0.0, 0.0])


def test_shadow_ray_t_max_bounds_the_hit(backend: Backend) -> None:
    # With a far bound short of the near sphere, the ray reports no occluder.
    origin = backend.asarray([[0.0, 0.0, 0.0]])
    direction = backend.asarray([[0.0, 0.0, -1.0]])
    hit = _scene().nearest_hit(origin, direction, _T_MIN, 1.0, backend)
    assert not bool(np.asarray(hit.hit)[0])


def test_area_lights_detects_emissive_quads() -> None:
    lamp = Quad(
        corner=(-0.3, 0.99, -2.3),
        edge1=(0.6, 0.0, 0.0),
        edge2=(0.0, 0.0, 0.6),
        material=Material(albedo=(0.7, 0.7, 0.7), emission=(15.0, 15.0, 15.0)),
    )
    wall = Quad(
        corner=(-1.0, -1.0, -3.0),
        edge1=(2.0, 0.0, 0.0),
        edge2=(0.0, 2.0, 0.0),
        material=Material(albedo=(0.7, 0.7, 0.7)),
    )
    scene = Scene(objects=[wall, lamp], lights=[])
    assert scene.area_lights() == [lamp]
