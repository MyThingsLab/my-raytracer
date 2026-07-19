from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

from myraytracer.render import render
from myraytracer.sceneio import load_scene

_CORNELL = pathlib.Path(__file__).resolve().parents[1] / "examples" / "cornell_box.json"
_TORCH = importlib.util.find_spec("torch") is not None
_needs_torch = pytest.mark.skipif(not _TORCH, reason="torch (gpu backend) not installed")


def _render(backend: str) -> np.ndarray:
    scene, camera = load_scene(_CORNELL)
    return render(scene, camera, width=48, height=48, spp=16, max_depth=3, seed=0, backend=backend)


def _region(image: np.ndarray, x0: float, x1: float) -> np.ndarray:
    _, w, _ = image.shape
    return image[:, int(x0 * w) : int(x1 * w)].reshape(-1, 3).mean(axis=0)


def _has_cornell_walls(image: np.ndarray) -> None:
    left_r, left_g, left_b = _region(image, 0.0, 0.2)
    right_r, right_g, right_b = _region(image, 0.8, 1.0)
    assert left_r > 2.0 * left_g and left_r > 2.0 * left_b  # red wall on the left
    assert right_g > 1.4 * right_r and right_g > 1.4 * right_b  # green wall on the right


def test_cpu_backend_renders_cornell() -> None:
    _has_cornell_walls(_render("cpu"))


@_needs_torch
def test_gpu_backend_renders_cornell() -> None:
    # The whole point of the unification: the torch backend now has per-object
    # materials, so it can render the Cornell box the scalar GPU path could not.
    _has_cornell_walls(_render("gpu"))


@_needs_torch
def test_backends_agree_on_cornell() -> None:
    # Different RNG streams, so not pixel-identical, but the region averages of
    # the same physics must match closely.
    cpu = _render("cpu")
    gpu = _render("gpu")
    for x0, x1 in [(0.0, 0.2), (0.8, 1.0), (0.4, 0.6)]:
        assert np.allclose(_region(cpu, x0, x1), _region(gpu, x0, x1), atol=0.1)


def test_render_returns_float64_numpy_for_gpu_backend() -> None:
    # Even on the torch path, render() hands back a plain numpy image so every
    # downstream consumer (tonemap, PNG writer) is unchanged.
    backend = "gpu" if _TORCH else "cpu"
    image = _render(backend)
    assert isinstance(image, np.ndarray)
    assert image.dtype == np.float64
