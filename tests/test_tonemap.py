from __future__ import annotations

import numpy as np
import pytest

from myraytracer.tonemap import to_srgb, tone_map


@pytest.mark.parametrize("method", ["aces", "reinhard", "none"])
def test_tone_map_maps_zero_to_zero(method: str) -> None:
    pixels = np.zeros((1, 1, 3))

    result = tone_map(pixels, method)

    assert np.all(result == 0.0)


@pytest.mark.parametrize("method", ["aces", "reinhard"])
def test_tone_map_is_monotonic(method: str) -> None:
    values = np.linspace(0.0, 20.0, 200).reshape(-1, 1, 1)
    pixels = np.repeat(values, 3, axis=2)

    result = tone_map(pixels, method)

    channel = result[:, 0, 0]
    assert np.all(np.diff(channel) >= 0.0)


@pytest.mark.parametrize("method", ["aces", "reinhard"])
def test_tone_map_compresses_bright_value_below_linear(method: str) -> None:
    pixels = np.full((1, 1, 3), 10.0)

    result = tone_map(pixels, method)

    assert np.all(result < 10.0)
    assert np.all(result <= 1.0)


def test_tone_map_none_passes_through_unchanged() -> None:
    pixels = np.array([[[0.2, 5.0, -1.0]]])

    result = tone_map(pixels, "none")

    assert np.array_equal(result, pixels)


def test_tone_map_unknown_method_raises() -> None:
    pixels = np.zeros((1, 1, 3))

    with pytest.raises(ValueError):
        tone_map(pixels, "bogus")


def test_to_srgb_matches_known_reference_point() -> None:
    pixels = np.full((1, 1, 3), 0.5)

    result = to_srgb(pixels)

    assert result[0, 0, 0] == pytest.approx(0.735, abs=0.001)


def test_to_srgb_maps_endpoints() -> None:
    pixels = np.array([[[0.0, 1.0, 0.0]]])

    result = to_srgb(pixels)

    assert result[0, 0, 0] == pytest.approx(0.0)
    assert result[0, 0, 1] == pytest.approx(1.0)


def test_to_srgb_clamps_out_of_range_values() -> None:
    pixels = np.array([[[1.5, -0.5, 0.5]]])

    result = to_srgb(pixels)

    assert result[0, 0, 0] == pytest.approx(1.0)
    assert result[0, 0, 1] == pytest.approx(0.0)
