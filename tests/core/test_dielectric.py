from __future__ import annotations

import math

import numpy as np

from myraytracer.core import bsdf
from myraytracer.core.backend import Backend

_N = 60000


def _const(backend: Backend, vec):
    return backend.asarray([list(vec)] * _N)


def _scalar(backend: Backend, x):
    return backend.asarray([float(x)] * _N)


def _glass_sample(backend: Backend, view, geo_normal, u1, ior=1.5):
    albedo = _const(backend, (1.0, 1.0, 1.0))
    metallic = _scalar(backend, 0.0)
    roughness = _scalar(backend, 1.0)
    transmission = _scalar(backend, 1.0)
    u2 = _scalar(backend, 0.5)
    return bsdf.sample(
        view, geo_normal, geo_normal, albedo, metallic, roughness, transmission,
        _scalar(backend, ior), u1, u2, backend,
    )


def test_snell_refraction_at_normal_incidence(backend: Backend) -> None:
    # Straight-on ray through a flat interface: ~4% Fresnel reflection (back up
    # +z), the rest refracts straight through (down -z), and no bending.
    view = _const(backend, (0.0, 0.0, 1.0))
    geo = _const(backend, (0.0, 0.0, 1.0))
    u1 = backend.random(backend.rng(0), _N)
    wi, _, _ = _glass_sample(backend, view, geo, u1)
    z = np.asarray(wi)[:, 2]
    reflect_fraction = float((z > 0).mean())
    assert math.isclose(reflect_fraction, 0.04, abs_tol=0.01)  # r0 = ((1.5-1)/(1.5+1))^2


def test_snell_refraction_bends_by_ior(backend: Backend) -> None:
    # A 30-degree incident ray forced to refract must obey n1 sin(t1) = n2 sin(t2).
    theta = math.radians(30.0)
    view = _const(backend, (math.sin(theta), 0.0, math.cos(theta)))
    geo = _const(backend, (0.0, 0.0, 1.0))
    u1 = _scalar(backend, 0.5)  # above the ~0.04 reflectance, so refract
    wi, _, _ = _glass_sample(backend, view, geo, u1)
    w = np.asarray(wi)[0]
    assert w[2] < 0.0  # transmitted to the far side
    sin_t = math.hypot(w[0], w[1])
    assert math.isclose(math.sin(theta) / sin_t, 1.5, rel_tol=1e-3)


def test_total_internal_reflection(backend: Backend) -> None:
    # Exiting glass at 60 degrees (> the ~41.8-degree critical angle) reflects
    # every ray -- deterministically, with no transmitted component.
    theta = math.radians(60.0)
    view = _const(backend, (math.sin(theta), 0.0, -math.cos(theta)))  # ray exits the glass
    geo = _const(backend, (0.0, 0.0, 1.0))
    u1 = _scalar(backend, 0.99)  # would refract if it could; TIR overrides
    wi, _, _ = _glass_sample(backend, view, geo, u1)
    w = np.asarray(wi)
    assert np.allclose(w.std(axis=0), 0.0, atol=1e-3)  # all identical (pure reflection)
    assert np.allclose(w.mean(axis=0), [-math.sin(theta), 0.0, -math.cos(theta)], atol=1e-3)


def test_dielectric_evaluates_to_zero_for_nee(backend: Backend) -> None:
    # A delta BSDF cannot be connected to by next-event estimation.
    view = _const(backend, (0.0, 0.0, 1.0))
    wi = _const(backend, (0.0, 0.0, -1.0))
    f_cos, pdf = bsdf.evaluate(
        view, wi, _const(backend, (0.0, 0.0, 1.0)),
        _const(backend, (1.0, 1.0, 1.0)),
        _scalar(backend, 0.0), _scalar(backend, 1.0), _scalar(backend, 1.0), backend,
    )
    assert np.allclose(np.asarray(f_cos), 0.0)
    assert np.allclose(np.asarray(pdf), 0.0)


def test_glass_sphere_scene_renders_finite(backend: Backend) -> None:
    from myraytracer.core.camera import Camera
    from myraytracer.core.geometry import Quad, Sphere
    from myraytracer.core.integrator import render as render_image
    from myraytracer.core.material import Material
    from myraytracer.core.scene import Scene

    scene = Scene(
        objects=[
            Quad(
                corner=(-0.5, 0.99, -2.3),
                edge1=(1.0, 0.0, 0.0),
                edge2=(0.0, 0.0, 0.6),
                material=Material((0.0, 0.0, 0.0), (12.0, 12.0, 12.0)),
            ),
            Quad(
                corner=(-1.0, -1.0, -3.0),
                edge1=(2.0, 0.0, 0.0),
                edge2=(0.0, 2.0, 0.0),
                material=Material((0.7, 0.7, 0.7)),
            ),
            Sphere(
                center=(0.0, 0.0, -2.0),
                radius=0.5,
                material=Material((1.0, 1.0, 1.0), transmission=1.0, ior=1.5),
            ),
        ],
        lights=[],
    )
    camera = Camera(
        origin=(0.0, 0.0, 0.0),
        look_at=(0.0, 0.0, -1.0),
        up=(0.0, 1.0, 0.0),
        vfov_degrees=60.0,
        aspect_ratio=1.0,
    )
    image = np.asarray(
        render_image(
            scene, camera, width=24, height=24, spp=4, max_depth=6, seed=0, backend=backend
        )
    )
    assert np.all(np.isfinite(image))
    assert image.max() > 0.0
