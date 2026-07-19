from __future__ import annotations

import math

import torch

from myraytracer.gpu.scene import Scene
from myraytracer.gpu.shading import direct_lighting
from myraytracer.gpu.vec import cross, normalize

# Below this remaining depth, indirect bounces are Russian-roulette
# terminated instead of always continuing, keeping the number of bounces
# bounded in expectation while staying unbiased (mirrors v0's tracer.py).
_ROULETTE_DEPTH = 2
_ROULETTE_MIN_SURVIVAL = 0.05
_T_MIN = 1e-4


def _orthonormal_basis(normal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Per-row tangent/bitangent frame orthogonal to `normal` (N, 3); the
    # seed axis is picked per row so the cross product below is never
    # near-zero-length.
    seed_y = torch.tensor([0.0, 1.0, 0.0], dtype=normal.dtype, device=normal.device)
    seed_x = torch.tensor([1.0, 0.0, 0.0], dtype=normal.dtype, device=normal.device)
    use_y = (normal[..., 0].abs() > 0.9).unsqueeze(-1)
    seed = torch.where(use_y, seed_y, seed_x)
    tangent = normalize(cross(seed, normal))
    bitangent = cross(normal, tangent)
    return tangent, bitangent


def _sample_cosine_hemisphere(normal: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    # Malley's method: a uniform disk sample projected up onto the
    # hemisphere, giving pdf(w) = cos(theta) / pi -- see trace()'s comment
    # on why that lets the cosine term drop out of the throughput update.
    n = normal.shape[0]
    u1 = torch.rand(n, generator=rng, dtype=normal.dtype, device=normal.device)
    u2 = torch.rand(n, generator=rng, dtype=normal.dtype, device=normal.device)
    radius = torch.sqrt(u1)
    theta = 2 * math.pi * u2
    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)
    z = torch.sqrt(torch.clamp(1.0 - u1, min=0.0))

    tangent, bitangent = _orthonormal_basis(normal)
    direction = tangent * x.unsqueeze(-1) + bitangent * y.unsqueeze(-1) + normal * z.unsqueeze(-1)
    return normalize(direction)


def trace(
    scene: Scene,
    ray_o: torch.Tensor,
    ray_d: torch.Tensor,
    *,
    max_depth: int,
    rng: torch.Generator,
) -> torch.Tensor:
    """Batched wavefront path trace over N rays: `Tensor(N, 3)` accumulated radiance.

    Bounces are streamed as batched tensor passes over the whole live ray
    set (Laine, Karras, Aila, "Megakernels Considered Harmful", HPG 2013)
    rather than recursing per ray, so the only Python-level loop is over
    `max_depth`. Each iteration carries a running `throughput` (N, 3) --
    the fraction of a hit's radiance that will still reach the camera after
    the bounces already taken -- and a per-ray `alive` (N,) mask that
    retires rays which miss geometry or lose Russian roulette. A dead ray
    is never removed from the batch: its throughput and current
    origin/direction are frozen in place with `torch.where` so every
    tensor keeps shape (N, ...) for the rest of the trace, and its
    contribution to `radiance` is masked to zero from that point on.
    """
    n = ray_o.shape[0]
    device = ray_o.device
    dtype = ray_o.dtype

    radiance = torch.zeros(n, 3, dtype=dtype, device=device)
    throughput = torch.ones(n, 3, dtype=dtype, device=device)
    alive = torch.ones(n, dtype=torch.bool, device=device)

    cur_o = ray_o
    cur_d = ray_d

    albedo = scene.albedo.to(dtype=dtype, device=device).expand(n, -1)
    emission = scene.emission.to(dtype=dtype, device=device).expand(n, -1)
    # Placeholder normal for dead rays: their sampled bounce direction is
    # discarded via `hit_mask` below, but a real (non-zero) vector is still
    # needed to keep the basis construction and normalize() finite.
    fallback_normal = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device).expand(n, -1)

    for step in range(max_depth + 1):
        remaining = max_depth - step

        hit = scene.nearest_hit(cur_o, cur_d, t_min=_T_MIN, t_max=math.inf)
        alive = alive & hit.hit
        hit_mask = alive.unsqueeze(-1)

        direct = direct_lighting(hit, albedo, scene)
        # `direct` already carries this hit's albedo (see shading.py), so
        # only `throughput` -- the product of every *earlier* bounce's
        # albedo -- multiplies in here. A dead ray contributes nothing,
        # masked rather than sliced out so every tensor stays shape (N, ...).
        radiance = radiance + torch.where(
            hit_mask, throughput * (emission + direct), torch.zeros_like(radiance)
        )

        if remaining <= 0 or not bool(torch.any(alive)):
            break

        if remaining < _ROULETTE_DEPTH:
            survival = torch.clamp(throughput.amax(dim=-1), min=_ROULETTE_MIN_SURVIVAL, max=1.0)
            roll = torch.rand(n, generator=rng, dtype=dtype, device=device)
            alive = alive & (roll < survival)
            hit_mask = alive.unsqueeze(-1)
            # Reweight surviving throughput by 1/survival so terminating
            # early stays unbiased in expectation.
            throughput = torch.where(hit_mask, throughput / survival.unsqueeze(-1), throughput)

        safe_normal = torch.where(hit_mask, hit.normal, fallback_normal)
        bounce_dir = _sample_cosine_hemisphere(safe_normal, rng)

        throughput = torch.where(hit_mask, throughput * albedo, throughput)
        cur_o = torch.where(hit_mask, hit.point, cur_o)
        cur_d = torch.where(hit_mask, bounce_dir, cur_d)

    return radiance
