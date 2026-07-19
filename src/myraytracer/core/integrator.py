from __future__ import annotations

import math
from typing import Any

from myraytracer.core.backend import Array, Backend
from myraytracer.core.camera import Camera
from myraytracer.core.linalg import cross, dot, length, normalize
from myraytracer.core.sampling import sample_cosine_hemisphere
from myraytracer.core.scene import Scene, SceneHit

# Below this many remaining bounces, indirect rays are Russian-roulette
# terminated instead of always continuing -- bounded path length, unbiased in
# expectation. Mirrors the scalar tracer this replaces.
_ROULETTE_DEPTH = 2
_ROULETTE_SURVIVAL = 0.9
_T_MIN = 1e-4


def _visibility(
    scene: Scene, origin: Array, direction: Array, dist: Array, backend: Backend
) -> Array:
    # 1.0 where the light is unoccluded, 0.0 where a closer surface blocks it.
    # The far bound is the per-ray light distance, so the light itself never
    # counts as its own occluder.
    occluded = scene.nearest_hit(origin, direction, _T_MIN, dist - _T_MIN, backend).hit
    return backend.where(occluded, backend.zeros_like(dist), backend.ones_like(dist))


def _direct_lighting(scene: Scene, hit: SceneHit, generator: Any, backend: Backend) -> Array:
    # Next-event estimation: explicit shadow-ray connections to point lights and
    # (one sample each) to emissive area lights, exactly as the scalar tracer.
    radiance = backend.zeros_like(hit.point)
    albedo = hit.albedo

    for light in scene.lights:
        position = backend.asarray(light.position)
        intensity = backend.asarray(light.intensity)
        to_light = position - hit.point
        dist = length(to_light)
        light_dir = to_light / dist[..., None]
        cos_theta = backend.clip(dot(hit.normal, light_dir), lo=0.0)
        visibility = _visibility(scene, hit.point, light_dir, dist, backend)
        falloff = cos_theta / (dist * dist)
        radiance = radiance + albedo * intensity * (falloff * visibility)[..., None]

    for quad in scene.area_lights():
        corner = backend.asarray(quad.corner)
        edge1 = backend.asarray(quad.edge1)
        edge2 = backend.asarray(quad.edge2)
        emission = backend.asarray(quad.material.emission)

        n = hit.point.shape[0]
        u = backend.random(generator, n)
        v = backend.random(generator, n)
        light_point = corner + edge1 * u[..., None] + edge2 * v[..., None]
        light_normal = normalize(cross(edge1, edge2))

        to_light = light_point - hit.point
        dist = length(to_light)
        light_dir = to_light / dist[..., None]
        cos_surface = backend.clip(dot(hit.normal, light_dir), lo=0.0)
        cos_light = backend.clip(dot(light_normal, -light_dir), lo=0.0)
        visibility = _visibility(scene, hit.point, light_dir, dist, backend)

        area = length(cross(edge1, edge2))
        falloff = cos_surface * cos_light * area / (dist * dist)
        radiance = radiance + albedo * emission * (falloff * visibility)[..., None]

    return radiance


def integrate(
    scene: Scene,
    ray_origin: Array,
    ray_dir: Array,
    *,
    max_depth: int,
    generator: Any,
    backend: Backend,
) -> Array:
    """Estimate the radiance along a batch of rays as an iterative wavefront.

    One sample per ray: at each bounce, accumulate emission + direct lighting
    weighted by the path throughput, then scatter along a cosine-weighted
    bounce until the depth budget runs out (with Russian-roulette termination
    for the deepest bounces). This is the batched, backend-agnostic unification
    of the scalar recursive tracer and gpu.pathtracer.
    """
    throughput = backend.ones_like(ray_origin)
    radiance = backend.zeros_like(ray_origin)
    origin = ray_origin
    direction = ray_dir
    remaining = max_depth
    up = backend.asarray((0.0, 0.0, 1.0))

    while True:
        hit = scene.nearest_hit(origin, direction, _T_MIN, math.inf, backend)
        hit_mask = hit.hit[..., None]

        radiance = radiance + throughput * hit.emission * hit_mask
        direct = _direct_lighting(scene, hit, generator, backend)
        radiance = radiance + throughput * direct * hit_mask
        # A ray that missed carries nothing further.
        throughput = backend.where(hit_mask, throughput, backend.zeros_like(throughput))

        if remaining <= 0:
            break

        if remaining < _ROULETTE_DEPTH:
            n = origin.shape[0]
            survive = (backend.random(generator, n) <= _ROULETTE_SURVIVAL)[..., None]
            throughput = backend.where(
                survive, throughput / _ROULETTE_SURVIVAL, backend.zeros_like(throughput)
            )

        # Missed rays have a zero normal; sampling would divide by zero, so give
        # them a dummy unit normal -- their throughput is already zero.
        safe_normal = backend.where(hit_mask, hit.normal, up)
        bounce = sample_cosine_hemisphere(safe_normal, generator, backend)
        throughput = throughput * hit.albedo
        origin = backend.where(hit_mask, hit.point, origin)
        direction = backend.where(hit_mask, bounce, direction)
        remaining -= 1

    return radiance


def render(
    scene: Scene,
    camera: Camera,
    *,
    width: int,
    height: int,
    spp: int,
    max_depth: int,
    seed: int,
    backend: Backend,
) -> Array:
    """Render a scene to an (H, W, 3) radiance image on the given backend.

    A single generator seeded once drives both the sub-pixel jitter and the
    path sampling, so the image is reproducible for a fixed seed. Samples are
    accumulated one full-frame wavefront at a time to bound memory.
    """
    generator = backend.rng(seed)
    rows, cols = backend.meshgrid(backend.arange(height), backend.arange(width))
    base_x = cols.reshape(-1)
    base_y = rows.reshape(-1)
    n_pixels = width * height

    accumulator = None
    for _ in range(spp):
        jitter_x = base_x + (backend.random(generator, n_pixels) - 0.5)
        jitter_y = base_y + (backend.random(generator, n_pixels) - 0.5)
        rays = camera.rays(jitter_x, jitter_y, width=width, height=height, backend=backend)
        sample = integrate(
            scene,
            rays.origin,
            rays.direction,
            max_depth=max_depth,
            generator=generator,
            backend=backend,
        )
        accumulator = sample if accumulator is None else accumulator + sample

    return (accumulator / spp).reshape(height, width, 3)
