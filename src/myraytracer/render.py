from __future__ import annotations

import math

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


def render_gbuffers(
    scene: Scene, camera: Camera, *, width: int, height: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render per-pixel first-hit guide buffers for denoising (a-trous/SVGF
    style edge-stopping): world-space normal (H,W,3), surface albedo
    (H,W,3), and hit distance / depth (H,W,1). Sibling to `render`, additive
    and opt-in -- the plain path-traced `render` path never calls this.

    Guide buffers are geometric (first-hit only, no path tracing), so a
    single ray through the pixel center is enough -- unlike `render`, there
    is no Monte Carlo noise to average over with `spp` samples.
    """
    normal = np.zeros((height, width, 3), dtype=np.float64)
    albedo = np.zeros((height, width, 3), dtype=np.float64)
    depth = np.zeros((height, width, 1), dtype=np.float64)

    for py in range(height):
        for px in range(width):
            ray = camera.ray_for_pixel(px + 0.5, py + 0.5, width, height)
            hit = scene.nearest_hit(ray, t_min=1e-4, t_max=math.inf)
            if hit is None:
                continue
            normal[py, px] = (hit.normal.x, hit.normal.y, hit.normal.z)
            albedo[py, px] = (
                hit.material.albedo.x,
                hit.material.albedo.y,
                hit.material.albedo.z,
            )
            depth[py, px, 0] = hit.t

    return normal, albedo, depth
