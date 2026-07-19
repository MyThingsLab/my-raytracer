from __future__ import annotations

import numpy as np

from myraytracer.core.backend import NUMPY, Backend
from myraytracer.core.camera import Camera
from myraytracer.core.geometry import Plane, Quad
from myraytracer.core.integrator import _light_sampling_pdf, integrate
from myraytracer.core.integrator import render as render_image
from myraytracer.core.material import Material
from myraytracer.core.scene import Scene

_T_MIN = 1e-4


def _lit_scene() -> Scene:
    # A diffuse floor under a large emissive quad, so BSDF-sampled bounce rays
    # frequently land on the light -- the regime where MIS matters.
    return Scene(
        objects=[
            Plane(
                point=(0.0, -1.0, 0.0),
                normal=(0.0, 1.0, 0.0),
                material=Material((0.7, 0.7, 0.7)),
            ),
            Quad(
                corner=(-2.0, 1.0, -4.0),
                edge1=(4.0, 0.0, 0.0),
                edge2=(0.0, 0.0, 4.0),
                material=Material((0.0, 0.0, 0.0), (3.0, 3.0, 3.0)),
            ),
        ],
        lights=[],
    )


def _camera() -> Camera:
    return Camera(
        origin=(0.0, 0.0, 0.0),
        look_at=(0.0, -0.3, -1.0),
        up=(0.0, 1.0, 0.0),
        vfov_degrees=60.0,
        aspect_ratio=1.0,
    )


def test_mis_is_unbiased_against_nee_reference(backend: Backend) -> None:
    # MIS must estimate the same radiance as pure NEE (mis=False), just with
    # lower variance -- if it double-counted the area light it would come out
    # measurably brighter.
    kwargs = dict(width=32, height=32, spp=64, max_depth=2, seed=0, backend=backend)
    mis = np.asarray(render_image(_lit_scene(), _camera(), mis=True, **kwargs))
    nee = np.asarray(render_image(_lit_scene(), _camera(), mis=False, **kwargs))

    floor = slice(20, 32)  # lower rows: floor lit indirectly, not the light itself
    mis_mean = mis[floor].reshape(-1, 3).mean(axis=0)
    nee_mean = nee[floor].reshape(-1, 3).mean(axis=0)
    assert np.allclose(mis_mean, nee_mean, rtol=0.05)


def test_primary_ray_sees_full_emission(backend: Backend) -> None:
    # A camera ray hitting a light directly is weighted 1 (no MIS reduction).
    emission = (4.0, 4.0, 4.0)
    scene = Scene(
        objects=[
            Quad(
                corner=(-1.0, -1.0, -2.0),
                edge1=(2.0, 0.0, 0.0),
                edge2=(0.0, 2.0, 0.0),
                material=Material((0.0, 0.0, 0.0), emission),
            )
        ],
        lights=[],
    )
    ro = backend.asarray([[0.0, 0.0, 0.0]])
    rd = backend.asarray([[0.0, 0.0, -1.0]])
    result = integrate(scene, ro, rd, max_depth=1, generator=backend.rng(0), backend=backend)
    assert np.allclose(np.asarray(result)[0], emission, atol=1e-4)


def test_light_sampling_pdf_on_and_off_light() -> None:
    quad = Quad(
        corner=(-1.0, -1.0, -2.0),
        edge1=(2.0, 0.0, 0.0),
        edge2=(0.0, 2.0, 0.0),
        material=Material((0.0, 0.0, 0.0), (5.0, 5.0, 5.0)),
    )
    scene = Scene(objects=[quad], lights=[])

    prev_point = NUMPY.asarray([[0.0, 0.0, 0.0]])
    on_light_point = NUMPY.asarray([[0.0, 0.0, -2.0]])  # centre of the quad
    direction = NUMPY.asarray([[0.0, 0.0, -1.0]])
    pdf = np.asarray(_light_sampling_pdf(scene, on_light_point, direction, prev_point, NUMPY))
    area, cos_light, dist = 4.0, 1.0, 2.0
    assert np.allclose(pdf[0], dist * dist / (cos_light * area))

    off_light_point = NUMPY.asarray([[5.0, 5.0, -2.0]])  # coplanar but outside the quad
    pdf_off = np.asarray(_light_sampling_pdf(scene, off_light_point, direction, prev_point, NUMPY))
    assert np.allclose(pdf_off[0], 0.0)
