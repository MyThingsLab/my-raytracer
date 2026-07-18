from __future__ import annotations

import math

import numpy as np

from myraytracer.camera import Camera
from myraytracer.ray import Ray
from myraytracer.scene import Scene
from myraytracer.vec import Vec3

# Below this remaining depth, indirect bounces are Russian-roulette
# terminated instead of always recursing, keeping worst-case recursion
# bounded while staying unbiased in expectation.
_ROULETTE_DEPTH = 2
_ROULETTE_SURVIVAL = 0.9


def _multiply(a: Vec3, b: Vec3) -> Vec3:
    return Vec3(a.x * b.x, a.y * b.y, a.z * b.z)


def _orthonormal_basis(normal: Vec3) -> tuple[Vec3, Vec3]:
    # Pick a seed axis not nearly parallel to normal so the cross product
    # below isn't near-zero-length.
    seed = Vec3(0, 1, 0) if abs(normal.x) > 0.9 else Vec3(1, 0, 0)
    tangent = seed.cross(normal).normalized()
    bitangent = normal.cross(tangent)
    return tangent, bitangent


def _sample_cosine_hemisphere(normal: Vec3, rng: np.random.Generator) -> Vec3:
    u1 = rng.random()
    u2 = rng.random()
    radius = math.sqrt(u1)
    theta = 2 * math.pi * u2
    x = radius * math.cos(theta)
    y = radius * math.sin(theta)
    z = math.sqrt(max(0.0, 1.0 - u1))

    tangent, bitangent = _orthonormal_basis(normal)
    return (tangent * x + bitangent * y + normal * z).normalized()


def trace_ray(ray: Ray, scene: Scene, rng: np.random.Generator, *, max_depth: int) -> Vec3:
    hit = scene.nearest_hit(ray, t_min=1e-4, t_max=math.inf)
    if hit is None:
        return Vec3(0, 0, 0)

    radiance = hit.material.emission

    for light in scene.lights:
        to_light = light.position - hit.point
        distance = to_light.length()
        light_dir = to_light * (1.0 / distance)
        cos_theta = hit.normal.dot(light_dir)
        if cos_theta <= 0.0:
            continue

        shadow_ray = Ray(origin=hit.point, direction=light_dir)
        if scene.nearest_hit(shadow_ray, t_min=1e-4, t_max=distance - 1e-4) is not None:
            continue

        # Inverse-square falloff: a point light's irradiance at the hit
        # point drops with distance^2, matching physical point-source
        # radiometry.
        falloff = cos_theta / (distance * distance)
        radiance = radiance + _multiply(hit.material.albedo, light.intensity) * falloff

    if max_depth <= 0:
        return radiance

    weight = 1.0
    if max_depth < _ROULETTE_DEPTH:
        if rng.random() > _ROULETTE_SURVIVAL:
            return radiance
        weight = 1.0 / _ROULETTE_SURVIVAL

    bounce_dir = _sample_cosine_hemisphere(hit.normal, rng)
    bounce_ray = Ray(origin=hit.point, direction=bounce_dir)
    indirect = trace_ray(bounce_ray, scene, rng, max_depth=max_depth - 1)
    # Cosine-weighted hemisphere sampling has pdf(w) = cos(theta) / pi,
    # which exactly cancels the cos(theta) factor in the rendering
    # equation's integrand, so no explicit cosine or 1/pdf factor appears
    # here beyond the albedo and the roulette weight.
    return radiance + _multiply(hit.material.albedo, indirect) * weight


def render_pixel(
    camera: Camera,
    scene: Scene,
    px: float,
    py: float,
    width: int,
    height: int,
    rng: np.random.Generator,
    *,
    spp: int,
    max_depth: int,
) -> Vec3:
    accumulator = Vec3(0, 0, 0)
    for _ in range(spp):
        jittered_px = px + rng.random() - 0.5
        jittered_py = py + rng.random() - 0.5
        ray = camera.ray_for_pixel(jittered_px, jittered_py, width, height)
        accumulator = accumulator + trace_ray(ray, scene, rng, max_depth=max_depth)
    return accumulator * (1.0 / spp)
