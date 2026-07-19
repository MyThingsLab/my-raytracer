# End-to-end regression oracle for the canonical Cornell box
# (examples/cornell_box.json). These tests pin the *observable* behaviour of
# the renderer -- coloured walls, colour bleeding onto the white floor, a soft
# shadow cast by the sphere -- as region-averaged image statistics with wide
# margins, so they survive Monte Carlo noise at low sample counts yet still
# fail if the physics regresses. They exist to guard the CPU/GPU-unification
# refactor: any batched rewrite of the core must keep reproducing this image.
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from myraytracer.geometry import Sphere
from myraytracer.render import render
from myraytracer.scene import Scene
from myraytracer.sceneio import load_scene

_SCENE = pathlib.Path(__file__).resolve().parents[1] / "examples" / "cornell_box.json"


def _region(image: np.ndarray, y0: float, y1: float, x0: float, x1: float) -> np.ndarray:
    # Fractional window (0..1) -> mean RGB, so thresholds are resolution-independent.
    h, w, _ = image.shape
    block = image[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)]
    return block.reshape(-1, 3).mean(axis=0)


@pytest.fixture(scope="module")
def cornell_image() -> np.ndarray:
    scene, camera = load_scene(_SCENE)
    return render(scene, camera, width=64, height=64, spp=8, max_depth=3, seed=0)


@pytest.mark.slow
def test_left_wall_is_red(cornell_image: np.ndarray) -> None:
    r, g, b = _region(cornell_image, 0.0, 1.0, 0.0, 0.2)
    assert r > 2.0 * g and r > 2.0 * b


@pytest.mark.slow
def test_right_wall_is_green(cornell_image: np.ndarray) -> None:
    r, g, b = _region(cornell_image, 0.0, 1.0, 0.8, 1.0)
    assert g > 1.4 * r and g > 1.4 * b


@pytest.mark.slow
def test_back_wall_is_neutral(cornell_image: np.ndarray) -> None:
    # The back wall is white: no channel dominates, and it is clearly lit.
    channels = _region(cornell_image, 0.33, 0.67, 0.4, 0.6)
    assert channels.min() > 0.3
    assert channels.max() < 1.6 * channels.min()


@pytest.mark.slow
def test_color_bleeds_onto_floor(cornell_image: np.ndarray) -> None:
    # The white floor picks up the wall colours via indirect bounces: its
    # left half reddens (near the red wall), its right half greens.
    lr, lg, _ = _region(cornell_image, 0.5, 1.0, 0.0, 0.5)
    rr, rg, _ = _region(cornell_image, 0.5, 1.0, 0.5, 1.0)
    assert lr > lg
    assert rg > rr


@pytest.mark.slow
def test_area_light_is_the_brightest_region(cornell_image: np.ndarray) -> None:
    ceiling = _region(cornell_image, 0.0, 0.167, 0.0, 1.0).mean()
    floor = _region(cornell_image, 0.72, 1.0, 0.0, 1.0).mean()
    assert ceiling > floor
    assert cornell_image.max() > 5.0  # the emissive quad itself


def test_render_is_deterministic_given_same_seed() -> None:
    scene, camera = load_scene(_SCENE)
    a = render(scene, camera, width=24, height=24, spp=4, max_depth=2, seed=7)
    b = render(scene, camera, width=24, height=24, spp=4, max_depth=2, seed=7)
    assert np.array_equal(a, b)


@pytest.mark.slow
def test_sphere_casts_a_soft_shadow() -> None:
    # Differential oracle: rendering the box with vs without the sphere,
    # sharing a seed, isolates the sphere's effect from Monte Carlo noise.
    # The floor patch beneath the sphere darkens sharply (occluded from the
    # ceiling light); a reference floor patch away from the sphere does not.
    scene, camera = load_scene(_SCENE)
    without = Scene(
        objects=[o for o in scene.objects if not isinstance(o, Sphere)],
        lights=scene.lights,
    )
    kwargs = dict(width=48, height=48, spp=16, max_depth=3, seed=0)
    with_sphere = render(scene, camera, **kwargs).mean(axis=2)
    no_sphere = render(without, camera, **kwargs).mean(axis=2)

    def patch(image: np.ndarray, x0: float, x1: float) -> float:
        h, w = image.shape
        return float(image[int(0.72 * h) : h, int(x0 * w) : int(x1 * w)].mean())

    shadow_with = patch(with_sphere, 0.15, 0.55)
    shadow_without = patch(no_sphere, 0.15, 0.55)
    reference_with = patch(with_sphere, 0.55, 0.9)
    reference_without = patch(no_sphere, 0.55, 0.9)

    # The occluder darkens its floor patch well below half brightness...
    assert shadow_with < 0.6 * shadow_without
    # ...while a floor patch it does not shade is essentially unchanged...
    assert reference_with > 0.85 * reference_without
    # ...so in the final image the shadow patch is clearly darker than the ref.
    assert shadow_with < 0.7 * reference_with
