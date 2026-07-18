from __future__ import annotations

import numpy as np

from myraytracer.camera import Camera
from myraytracer.scene import Scene
from myraytracer.tracer import render_pixel


def render(
    scene: Scene,
    camera: Camera,
    *,
    width: int,
    height: int,
    spp: int,
    max_depth: int,
    seed: int,
) -> np.ndarray:
    # A single Generator seeded once and shared across all pixels, rather
    # than reseeding per-pixel, keeps the whole image reproducible for a
    # fixed seed while still giving every pixel an independent draw
    # sequence -- per-pixel reseeding would correlate samples across
    # pixels that happen to share a seed-derived state.
    rng = np.random.Generator(np.random.PCG64(seed))
    pixels = np.zeros((height, width, 3), dtype=np.float64)

    for py in range(height):
        for px in range(width):
            color = render_pixel(
                camera, scene, px, py, width, height, rng, spp=spp, max_depth=max_depth
            )
            pixels[py, px] = (color.x, color.y, color.z)

    return pixels
