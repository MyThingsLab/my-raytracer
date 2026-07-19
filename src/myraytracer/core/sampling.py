from __future__ import annotations

import math
from typing import Any

from myraytracer.core.backend import Array, Backend
from myraytracer.core.linalg import cross, normalize


def orthonormal_basis(normal: Array, backend: Backend) -> tuple[Array, Array]:
    # Per-row tangent frame orthogonal to `normal` (N, 3). The seed axis is
    # chosen per row so the cross product is never near-parallel to normal.
    seed_y = backend.asarray((0.0, 1.0, 0.0))
    seed_x = backend.asarray((1.0, 0.0, 0.0))
    use_y = (abs(normal[..., 0]) > 0.9)[..., None]
    seed = backend.where(use_y, seed_y, seed_x)
    tangent = normalize(cross(seed, normal))
    bitangent = cross(normal, tangent)
    return tangent, bitangent


def cosine_hemisphere_from_uv(normal: Array, u1: Array, u2: Array, backend: Backend) -> Array:
    # Malley's method for a given uniform (u1, u2): a disk sample lifted onto
    # the hemisphere, giving pdf(w) = cos(theta) / pi. Factored out of
    # `sample_cosine_hemisphere` so the BSDF can reuse the same random pair
    # across its diffuse and specular lobes -- keeping the pure-diffuse path
    # bit-identical to before.
    radius = u1**0.5
    theta = 2 * math.pi * u2
    x = radius * backend.xp.cos(theta)
    y = radius * backend.xp.sin(theta)
    z = backend.clip(1.0 - u1, lo=0.0) ** 0.5

    tangent, bitangent = orthonormal_basis(normal, backend)
    direction = tangent * x[..., None] + bitangent * y[..., None] + normal * z[..., None]
    return normalize(direction)


def sample_cosine_hemisphere(normal: Array, generator: Any, backend: Backend) -> Array:
    n = normal.shape[0]
    u1 = backend.random(generator, n)
    u2 = backend.random(generator, n)
    return cosine_hemisphere_from_uv(normal, u1, u2, backend)
