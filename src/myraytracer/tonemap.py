from __future__ import annotations

import numpy as np

# ACES filmic tone-mapping curve, fitted approximation by Krzysztof
# Narkowicz ("ACES Filmic Tone Mapping Curve", 2016,
# https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/).
_ACES_A = 2.51
_ACES_B = 0.03
_ACES_C = 2.43
_ACES_D = 0.59
_ACES_E = 0.14


def _aces(pixels: np.ndarray) -> np.ndarray:
    numerator = pixels * (_ACES_A * pixels + _ACES_B)
    denominator = pixels * (_ACES_C * pixels + _ACES_D) + _ACES_E
    return np.clip(numerator / denominator, 0.0, 1.0)


def _reinhard(pixels: np.ndarray) -> np.ndarray:
    return pixels / (1.0 + pixels)


def tone_map(pixels: np.ndarray, method: str = "aces") -> np.ndarray:
    """Compress an HDR linear-light `(H, W, 3)` array into displayable range.

    `method` selects the curve: `"aces"` applies the Narkowicz (2016) ACES
    filmic approximation, `"reinhard"` applies `x / (1 + x)`, and `"none"`
    passes the input through unchanged. Operates on linear values, before
    gamma correction via `to_srgb`.
    """
    if method == "aces":
        return _aces(pixels)
    if method == "reinhard":
        return _reinhard(pixels)
    if method == "none":
        return pixels
    raise ValueError(f"unknown tone map method: {method!r}")


def to_srgb(pixels: np.ndarray) -> np.ndarray:
    """Convert linear-light values to gamma-encoded sRGB (IEC 61966-2-1).

    Applies the piecewise sRGB transfer function: a linear segment near
    zero and a power curve elsewhere. Input is clamped to `[0, 1]` first.
    """
    clamped = np.clip(pixels, 0.0, 1.0)
    return np.where(
        clamped <= 0.0031308,
        clamped * 12.92,
        1.055 * np.power(clamped, 1.0 / 2.4) - 0.055,
    )
