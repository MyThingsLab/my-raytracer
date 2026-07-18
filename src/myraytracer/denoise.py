from __future__ import annotations

import numpy as np

# 5-tap B3 spline (binomial [1, 4, 6, 4, 1] / 16), the separable 1-D kernel
# used by the a-trous ("with holes") wavelet transform of Dammertz, Sewtz,
# Hanika & Lensch, "Edge-Avoiding A-Trous Wavelet Transform for Fast Global
# Illumination Filtering", HPG 2010. Dilating its taps by 2**iteration
# between passes grows the filter's support exponentially while reusing a
# constant number of samples per pixel per pass.
_KERNEL_1D = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
_TAPS = np.arange(-2, 3)


def _edge_stopping_weight(center: np.ndarray, neighbor: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian edge-stopping weight exp(-||center - neighbor||^2 / sigma^2),
    collapsed over the last (channel) axis to one weight per pixel. Shared
    by the normal-, depth- and albedo-edge-stopping functions below -- only
    the guide buffer and sigma differ between them."""
    diff_sq = np.sum((center - neighbor) ** 2, axis=-1, keepdims=True)
    return np.exp(-diff_sq / (sigma * sigma))


def _normal_weight(center: np.ndarray, neighbor: np.ndarray, sigma: float) -> np.ndarray:
    """w_n: edge-stopping weight from first-hit surface normals -- kills the
    filter across a geometric silhouette or crease even when color/depth
    agree either side of it."""
    return _edge_stopping_weight(center, neighbor, sigma)


def _depth_weight(center: np.ndarray, neighbor: np.ndarray, sigma: float) -> np.ndarray:
    """w_z: edge-stopping weight from first-hit depth -- kills the filter
    across a depth discontinuity (e.g. an object silhouetted against a
    farther background)."""
    return _edge_stopping_weight(center, neighbor, sigma)


def _albedo_weight(center: np.ndarray, neighbor: np.ndarray, sigma: float) -> np.ndarray:
    """w_rt: edge-stopping weight from first-hit surface albedo -- kills the
    filter across a material boundary that shares normal and depth (e.g. a
    checkerboard on a single flat plane)."""
    return _edge_stopping_weight(center, neighbor, sigma)


def _shift(buffer: np.ndarray, dy: int, dx: int) -> np.ndarray:
    # Clamp-to-edge shift: reading out of bounds at the image border repeats
    # the border pixel instead of wrapping or zero-padding, so the filter
    # doesn't darken/blur the border with fabricated neighbors.
    height, width = buffer.shape[:2]
    rows = np.clip(np.arange(height) + dy, 0, height - 1)
    cols = np.clip(np.arange(width) + dx, 0, width - 1)
    return buffer[rows[:, None], cols[None, :]]


def _atrous_pass(
    color: np.ndarray,
    normal: np.ndarray,
    albedo: np.ndarray,
    depth: np.ndarray,
    *,
    step: int,
    sigma_normal: float,
    sigma_depth: float,
    sigma_albedo: float,
) -> np.ndarray:
    accum = np.zeros_like(color)
    weight_sum = np.zeros(color.shape[:2] + (1,), dtype=np.float64)

    for dy, kernel_y in zip(_TAPS * step, _KERNEL_1D, strict=True):
        for dx, kernel_x in zip(_TAPS * step, _KERNEL_1D, strict=True):
            neighbor_color = _shift(color, dy, dx)
            neighbor_normal = _shift(normal, dy, dx)
            neighbor_albedo = _shift(albedo, dy, dx)
            neighbor_depth = _shift(depth, dy, dx)

            weight = (
                kernel_y
                * kernel_x
                * _normal_weight(normal, neighbor_normal, sigma_normal)
                * _depth_weight(depth, neighbor_depth, sigma_depth)
                * _albedo_weight(albedo, neighbor_albedo, sigma_albedo)
            )

            accum += neighbor_color * weight
            weight_sum += weight

    return accum / weight_sum


def denoise(
    color: np.ndarray,
    normal: np.ndarray,
    albedo: np.ndarray,
    depth: np.ndarray,
    *,
    iterations: int = 5,
    sigma_normal: float = 0.3,
    sigma_depth: float = 1.0,
    sigma_albedo: float = 0.1,
) -> np.ndarray:
    """Edge-avoiding a-trous wavelet denoiser (Dammertz, Sewtz, Hanika &
    Lensch, "Edge-Avoiding A-Trous Wavelet Transform for Fast Global
    Illumination Filtering", HPG 2010; the spatial filter later reused as
    SVGF's per-frame pass in Schied et al. 2017).

    Runs `iterations` passes of a 5x5 separable B3-spline blur over `color`
    (H,W,3), each pass dilating its taps by 2**i so the filter's support
    grows exponentially without growing the per-pixel sample count. Every
    tap is weighted by the kernel times three edge-stopping functions --
    `_normal_weight`, `_depth_weight`, and `_albedo_weight`, each
    exp(-||center - neighbor||^2 / sigma^2) on the corresponding guide
    buffer -- so the filter smooths within a surface but stops at geometric
    silhouettes, depth discontinuities, and material boundaries. In a flat,
    single-material region all three weights stay near 1 and the filter
    reduces to the plain binomial blur, which averages zero-mean Monte
    Carlo noise without shifting the mean.

    `normal` and `albedo` are (H,W,3); `depth` is (H,W,1).
    """
    result = color.astype(np.float64)
    normal = normal.astype(np.float64)
    albedo = albedo.astype(np.float64)
    depth = depth.astype(np.float64)

    for i in range(iterations):
        result = _atrous_pass(
            result,
            normal,
            albedo,
            depth,
            step=1 << i,
            sigma_normal=sigma_normal,
            sigma_depth=sigma_depth,
            sigma_albedo=sigma_albedo,
        )

    return result
