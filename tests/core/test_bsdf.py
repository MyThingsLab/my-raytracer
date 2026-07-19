from __future__ import annotations

import math

import numpy as np

from myraytracer.core import bsdf
from myraytracer.core.backend import Backend
from myraytracer.core.linalg import normalize

_N = 40000


def _const(backend: Backend, vec, n=_N):
    return backend.asarray([list(vec)] * n)


def _material(backend: Backend, albedo, metallic, roughness, n=_N):
    return (
        _const(backend, albedo, n),
        backend.asarray([float(metallic)] * n),
        backend.asarray([float(roughness)] * n),
    )


def _sample(backend, view, normal, albedo, metallic, roughness, u1, u2):
    # Opaque (non-transmissive) sample: geo_normal = shading normal, no glass.
    n = normal.shape[0]
    zero = backend.asarray([0.0] * n)
    ior = backend.asarray([1.5] * n)
    return bsdf.sample(
        view, normal, normal, albedo, metallic, roughness, zero, ior, u1, u2, backend
    )


def _evaluate(backend, view, wi, normal, albedo, metallic, roughness):
    n = normal.shape[0]
    zero = backend.asarray([0.0] * n)
    return bsdf.evaluate(view, wi, normal, albedo, metallic, roughness, zero, backend)


def test_diffuse_sample_pdf_is_cosine_weighted(backend: Backend) -> None:
    normal = _const(backend, (0.0, 0.0, 1.0))
    view = _const(backend, (0.0, 0.0, 1.0))
    albedo, metallic, roughness = _material(backend, (0.6, 0.6, 0.6), 0.0, 1.0)
    gen = backend.rng(0)
    u1, u2 = backend.random(gen, _N), backend.random(gen, _N)
    wi, weight, pdf = _sample(backend, view, normal, albedo, metallic, roughness, u1, u2)

    cos = np.asarray(wi)[:, 2]
    assert np.all(cos >= -1e-6)  # sampled into the upper hemisphere
    assert np.allclose(np.asarray(pdf), cos / math.pi, atol=1e-5)
    assert np.allclose(np.asarray(weight), np.asarray(albedo), atol=1e-6)  # f*cos/pdf = albedo


def test_sample_and_evaluate_pdf_agree(backend: Backend) -> None:
    # The pdf returned by sampling must equal evaluate()'s pdf for the sampled
    # direction -- the invariant MIS relies on. Checked for a GGX conductor.
    normal = _const(backend, (0.0, 0.0, 1.0))
    view = normalize(_const(backend, (0.4, 0.0, 1.0)))
    albedo, metallic, roughness = _material(backend, (0.9, 0.7, 0.3), 1.0, 0.35)
    gen = backend.rng(1)
    u1, u2 = backend.random(gen, _N), backend.random(gen, _N)
    wi, _, pdf = _sample(backend, view, normal, albedo, metallic, roughness, u1, u2)

    _, pdf_eval = _evaluate(backend, view, wi, normal, albedo, metallic, roughness)
    pdf = np.asarray(pdf)
    valid = pdf > 1e-3
    assert np.allclose(pdf[valid], np.asarray(pdf_eval)[valid], rtol=1e-4)


def test_conductor_reflectance_matches_f0_at_normal_incidence(backend: Backend) -> None:
    # At normal incidence a GGX conductor's directional reflectance (the mean
    # Monte Carlo throughput weight) equals its F0 = albedo, so it is energy
    # conserving and tinted by the base colour.
    normal = _const(backend, (0.0, 0.0, 1.0))
    view = _const(backend, (0.0, 0.0, 1.0))
    f0 = (0.9, 0.7, 0.3)
    albedo, metallic, roughness = _material(backend, f0, 1.0, 0.3)
    gen = backend.rng(2)
    u1, u2 = backend.random(gen, _N), backend.random(gen, _N)
    _, weight, _ = _sample(backend, view, normal, albedo, metallic, roughness, u1, u2)
    assert np.allclose(np.asarray(weight).mean(axis=0), f0, atol=0.02)


def test_smooth_conductor_is_mirror_like(backend: Backend) -> None:
    # A near-zero roughness conductor reflects a view about the normal: at
    # normal incidence every sampled direction is ~the normal.
    normal = _const(backend, (0.0, 0.0, 1.0))
    view = _const(backend, (0.0, 0.0, 1.0))
    albedo, metallic, roughness = _material(backend, (1.0, 1.0, 1.0), 1.0, 0.01)
    gen = backend.rng(3)
    u1, u2 = backend.random(gen, _N), backend.random(gen, _N)
    wi, _, _ = _sample(backend, view, normal, albedo, metallic, roughness, u1, u2)
    assert np.allclose(np.asarray(wi).mean(axis=0), [0.0, 0.0, 1.0], atol=1e-2)


def test_metal_sphere_scene_renders_finite(backend: Backend) -> None:
    from myraytracer.core.camera import Camera
    from myraytracer.core.geometry import Quad, Sphere
    from myraytracer.core.integrator import render as render_image
    from myraytracer.core.material import Material
    from myraytracer.core.scene import Scene

    scene = Scene(
        objects=[
            Quad(
                corner=(-1.0, -1.0, -3.0),
                edge1=(2.0, 0.0, 0.0),
                edge2=(0.0, 2.0, 0.0),
                material=Material((0.7, 0.7, 0.7)),
            ),
            Sphere(
                center=(0.0, 0.0, -2.0),
                radius=0.6,
                material=Material((0.95, 0.85, 0.55), metallic=1.0, roughness=0.15),
            ),
        ],
        lights=[],
    )
    scene.objects.insert(
        0,
        Quad(
            corner=(-0.5, 0.99, -2.3),
            edge1=(1.0, 0.0, 0.0),
            edge2=(0.0, 0.0, 0.6),
            material=Material((0.0, 0.0, 0.0), (12.0, 12.0, 12.0)),
        ),
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
            scene, camera, width=24, height=24, spp=4, max_depth=4, seed=0, backend=backend
        )
    )
    assert np.all(np.isfinite(image))
    assert image.max() > 0.0
