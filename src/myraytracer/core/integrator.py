from __future__ import annotations

import math
from typing import Any

from myraytracer.core.backend import Array, Backend
from myraytracer.core.bsdf import evaluate as bsdf_evaluate
from myraytracer.core.bsdf import sample as bsdf_sample
from myraytracer.core.camera import Camera
from myraytracer.core.linalg import cross, dot, length, normalize
from myraytracer.core.scene import Scene, SceneHit

# Below this many remaining bounces, indirect rays are Russian-roulette
# terminated instead of always continuing -- bounded path length, unbiased in
# expectation. Mirrors the scalar tracer this replaces.
_ROULETTE_DEPTH = 2
_ROULETTE_SURVIVAL = 0.9
_T_MIN = 1e-4
_INV_PI = 1.0 / math.pi
_EPS = 1e-9


def _visibility(
    scene: Scene, origin: Array, direction: Array, dist: Array, backend: Backend
) -> Array:
    # 1.0 where the light is unoccluded, 0.0 where a closer surface blocks it.
    # The far bound is the per-ray light distance, so the light itself never
    # counts as its own occluder.
    occluded = scene.nearest_hit(origin, direction, _T_MIN, dist - _T_MIN, backend).hit
    return backend.where(occluded, backend.zeros_like(dist), backend.ones_like(dist))


def _light_sampling_pdf(
    scene: Scene, hit_point: Array, direction: Array, prev_point: Array, backend: Backend
) -> Array:
    # Solid-angle pdf that area-light sampling *would* have assigned to reaching
    # the surface at `hit_point` along `direction` from `prev_point`. Summed
    # over the area lights the point lies on (normally one), it is the light
    # side of the MIS balance heuristic for emission picked up by a BSDF-sampled
    # bounce ray. Zero for points on no area light.
    total = backend.zeros_like(hit_point[..., 0])
    dist = length(hit_point - prev_point)
    for quad in scene.area_lights():
        corner = backend.asarray(quad.corner)
        edge1 = backend.asarray(quad.edge1)
        edge2 = backend.asarray(quad.edge2)
        normal = normalize(cross(edge1, edge2))
        area = length(cross(edge1, edge2))

        rel = hit_point - corner
        e1_len_sq = dot(edge1, edge1)
        e2_len_sq = dot(edge2, edge2)
        e1_dot_e2 = dot(edge1, edge2)
        denom = e1_len_sq * e2_len_sq - e1_dot_e2 * e1_dot_e2
        d1 = dot(rel, edge1)
        d2 = dot(rel, edge2)
        alpha = (d1 * e2_len_sq - d2 * e1_dot_e2) / denom
        beta = (d2 * e1_len_sq - d1 * e1_dot_e2) / denom

        coplanar = abs(dot(normal, rel)) < 1e-3
        inside = (alpha >= 0.0) & (alpha <= 1.0) & (beta >= 0.0) & (beta <= 1.0)
        cos_light = abs(dot(normal, direction))
        on_light = coplanar & inside & (cos_light > _EPS)

        pdf = dist * dist / (backend.clip(cos_light, lo=_EPS) * area)
        total = total + backend.where(on_light, pdf, backend.zeros_like(pdf))
    return total


def _bsdf(hit: SceneHit, view: Array, direction: Array, backend: Backend):
    # BSDF value*cos and pdf for a light direction, using the hit's material.
    return bsdf_evaluate(
        view,
        direction,
        hit.normal,
        hit.albedo,
        hit.metallic,
        hit.roughness,
        hit.transmission,
        backend,
    )


def _direct_lighting(
    scene: Scene, hit: SceneHit, view: Array, generator: Any, backend: Backend, mis: bool
) -> Array:
    # Next-event estimation: explicit shadow-ray connections to point lights and
    # (one sample each) to emissive area lights, shading with the surface's BSDF
    # (Lambertian or GGX). Point lights are delta sources (BSDF sampling can
    # never hit them), so they are pure NEE; area lights carry the light-side
    # MIS weight from the BSDF's own sampling pdf.
    radiance = backend.zeros_like(hit.point)

    for light in scene.lights:
        position = backend.asarray(light.position)
        intensity = backend.asarray(light.intensity)
        to_light = position - hit.point
        dist = length(to_light)
        light_dir = to_light / dist[..., None]
        f_cos, _ = _bsdf(hit, view, light_dir, backend)
        visibility = _visibility(scene, hit.point, light_dir, dist, backend)
        radiance = radiance + f_cos * intensity * (visibility / (dist * dist))[..., None]

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
        area = length(cross(edge1, edge2))

        to_light = light_point - hit.point
        dist = length(to_light)
        light_dir = to_light / dist[..., None]
        cos_light = backend.clip(dot(light_normal, -light_dir), lo=0.0)
        visibility = _visibility(scene, hit.point, light_dir, dist, backend)

        f_cos, pdf_bsdf = _bsdf(hit, view, light_dir, backend)
        geometry = cos_light * area / (dist * dist)
        contribution = f_cos * emission * (geometry * visibility)[..., None]

        if mis:
            # Balance heuristic between this light sample and the BSDF sample
            # that could have produced the same direction (both solid-angle).
            pdf_light = dist * dist / (backend.clip(cos_light, lo=_EPS) * area)
            weight = pdf_light / (pdf_light + pdf_bsdf + _EPS)
            contribution = contribution * weight[..., None]

        radiance = radiance + contribution

    return radiance


def integrate(
    scene: Scene,
    ray_origin: Array,
    ray_dir: Array,
    *,
    max_depth: int,
    generator: Any,
    backend: Backend,
    mis: bool = True,
) -> Array:
    """Estimate radiance along a batch of rays as an iterative wavefront.

    One sample per ray: at each hit, accumulate emission (BSDF-sampling term)
    and next-event direct lighting (light-sampling term), then scatter along a
    cosine-weighted bounce until the depth budget runs out (with Russian-
    roulette termination for the deepest bounces). When `mis` is set, the two
    ways an area light can be reached -- a BSDF bounce landing on it, and an
    explicit NEE connection -- are combined with the balance heuristic instead
    of double-counted, cutting variance without bias. With `mis=False` the
    estimator degrades to pure NEE (light-sampling only), which is the
    unbiased reference the MIS result must match in expectation.
    """
    throughput = backend.ones_like(ray_origin)
    radiance = backend.zeros_like(ray_origin)
    origin = ray_origin
    direction = ray_dir
    prev_point = ray_origin
    prev_bsdf_pdf = backend.zeros_like(ray_origin[..., 0])
    remaining = max_depth
    first = True
    up = backend.asarray((0.0, 0.0, 1.0))

    while True:
        hit = scene.nearest_hit(origin, direction, _T_MIN, math.inf, backend)
        hit_mask = hit.hit[..., None]
        view = -direction  # points from the surface back toward the viewer

        # Emission (BSDF-sampling term). The camera ray sees lights fully; a
        # bounce ray's emission is MIS-weighted against the NEE it competes with.
        if first:
            emission_weight = backend.ones_like(prev_bsdf_pdf)
        elif not mis:
            emission_weight = backend.zeros_like(prev_bsdf_pdf)
        else:
            pdf_light = _light_sampling_pdf(scene, hit.point, direction, prev_point, backend)
            emission_weight = prev_bsdf_pdf / (prev_bsdf_pdf + pdf_light + _EPS)
        radiance = radiance + throughput * hit.emission * (emission_weight[..., None] * hit_mask)

        direct = _direct_lighting(scene, hit, view, generator, backend, mis)
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
        safe_geo_normal = backend.where(hit_mask, hit.geo_normal, up)
        n = origin.shape[0]
        u1 = backend.random(generator, n)
        u2 = backend.random(generator, n)
        bounce, weight, pdf = bsdf_sample(
            view,
            safe_normal,
            safe_geo_normal,
            hit.albedo,
            hit.metallic,
            hit.roughness,
            hit.transmission,
            hit.ior,
            u1,
            u2,
            backend,
        )
        # The sampling pdf of this bounce, carried forward so the next hit's
        # emission can be MIS-weighted against it.
        prev_bsdf_pdf = pdf
        prev_point = hit.point
        throughput = throughput * weight
        origin = backend.where(hit_mask, hit.point, origin)
        direction = backend.where(hit_mask, bounce, direction)
        remaining -= 1
        first = False

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
    mis: bool = True,
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
            mis=mis,
        )
        accumulator = sample if accumulator is None else accumulator + sample

    return (accumulator / spp).reshape(height, width, 3)
