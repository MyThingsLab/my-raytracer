from __future__ import annotations

import math

from myraytracer.core.backend import Array, Backend
from myraytracer.core.linalg import dot, normalize
from myraytracer.core.sampling import cosine_hemisphere_from_uv, orthonormal_basis

# A per-ray BSDF that is either Lambertian diffuse (metallic <= 0.5) or a GGX
# microfacet conductor (metallic > 0.5). Both lobes are always evaluated over
# the whole ray batch and selected with `where`, so a mixed-material scene is
# handled branchlessly. The diffuse lobe reuses the integrator's cosine sample
# unchanged, so a pure-diffuse render is bit-identical to before GGX.

_INV_PI = 1.0 / math.pi
_EPS = 1e-6
_MIN_ALPHA = 1e-4


def _alpha(roughness: Array) -> Array:
    return roughness * roughness


def _ggx_d(cos_h: Array, alpha: Array, backend: Backend) -> Array:
    # Trowbridge-Reitz (GGX) normal distribution, zero below the horizon.
    a2 = alpha * alpha
    denom = cos_h * cos_h * (a2 - 1.0) + 1.0
    d = a2 / (math.pi * denom * denom)
    return backend.where(cos_h > 0.0, d, backend.zeros_like(d))


def _smith_g1(cos_v: Array, alpha: Array, backend: Backend) -> Array:
    # Smith masking-shadowing term for one direction (separable G = G1*G1).
    a2 = alpha * alpha
    positive = cos_v > 0.0
    safe = backend.clip(cos_v, lo=_EPS)
    g1 = 2.0 * safe / (safe + (a2 + (1.0 - a2) * safe * safe) ** 0.5)
    return backend.where(positive, g1, backend.zeros_like(g1))


def _fresnel_schlick(cos_theta: Array, f0: Array, backend: Backend) -> Array:
    # Schlick approximation; f0 is the per-channel reflectance at normal
    # incidence (the conductor's albedo here).
    m = backend.clip(1.0 - cos_theta, lo=0.0, hi=1.0)
    return f0 + (1.0 - f0) * (m**5)[..., None]


def _diffuse_sample(normal: Array, albedo: Array, u1: Array, u2: Array, backend: Backend):
    wi = cosine_hemisphere_from_uv(normal, u1, u2, backend)
    pdf = backend.clip(dot(normal, wi), lo=0.0) * _INV_PI
    return wi, albedo, pdf


def _metal_sample(
    view: Array, normal: Array, albedo: Array, alpha: Array, u1: Array, u2: Array, backend: Backend
):
    # Sample a microfacet normal h from the GGX distribution, reflect the view
    # about it, and return the throughput weight f*cos/pdf (which collapses to
    # F*G*(wo.h)/((n.wo)(n.h))) and the solid-angle pdf.
    cos_h = ((1.0 - u1) / (1.0 + (alpha * alpha - 1.0) * u1)) ** 0.5
    sin_h = backend.clip(1.0 - cos_h * cos_h, lo=0.0) ** 0.5
    phi = 2.0 * math.pi * u2
    tangent, bitangent = orthonormal_basis(normal, backend)
    h = (
        tangent * (sin_h * backend.xp.cos(phi))[..., None]
        + bitangent * (sin_h * backend.xp.sin(phi))[..., None]
        + normal * cos_h[..., None]
    )
    h = normalize(h)

    wi = normalize(2.0 * dot(view, h)[..., None] * h - view)
    n_wo = dot(normal, view)
    n_wi = dot(normal, wi)
    n_h = dot(normal, h)
    wo_h = dot(view, h)
    valid = (n_wi > 0.0) & (wo_h > 0.0) & (n_wo > 0.0)

    g = _smith_g1(n_wo, alpha, backend) * _smith_g1(n_wi, alpha, backend)
    fresnel = _fresnel_schlick(wo_h, albedo, backend)
    weight = fresnel * (g * wo_h / (backend.clip(n_wo, lo=_EPS) * backend.clip(n_h, lo=_EPS)))[
        ..., None
    ]
    pdf = _ggx_d(n_h, alpha, backend) * n_h / (4.0 * backend.clip(wo_h, lo=_EPS))

    weight = backend.where(valid[..., None], weight, backend.zeros_like(weight))
    pdf = backend.where(valid, pdf, backend.full_like(pdf, _EPS))
    return wi, weight, pdf


def sample(
    view: Array,
    normal: Array,
    albedo: Array,
    metallic: Array,
    roughness: Array,
    u1: Array,
    u2: Array,
    backend: Backend,
):
    """Sample an outgoing direction. Returns (wi, weight, pdf), where weight is
    the Monte Carlo throughput factor f*cos/pdf and pdf is the solid-angle
    density. `view` points from the surface toward the viewer."""
    diff_wi, diff_weight, diff_pdf = _diffuse_sample(normal, albedo, u1, u2, backend)
    metal_wi, metal_weight, metal_pdf = _metal_sample(
        view, normal, albedo, _alpha(roughness), u1, u2, backend
    )
    is_metal = (metallic > 0.5)[..., None]
    wi = backend.where(is_metal, metal_wi, diff_wi)
    weight = backend.where(is_metal, metal_weight, diff_weight)
    pdf = backend.where(metallic > 0.5, metal_pdf, diff_pdf)
    return wi, weight, pdf


def evaluate(
    view: Array,
    wi: Array,
    normal: Array,
    albedo: Array,
    metallic: Array,
    roughness: Array,
    backend: Backend,
):
    """Evaluate the BSDF for a fixed incident direction `wi`. Returns
    (f_cos, pdf): the BSDF value times cos(n, wi) (per channel) and the
    solid-angle sampling pdf of `wi` -- both needed for next-event estimation
    and its MIS weight."""
    n_wi = backend.clip(dot(normal, wi), lo=0.0)

    diff_f_cos = albedo * _INV_PI * n_wi[..., None]
    diff_pdf = n_wi * _INV_PI

    alpha = _alpha(roughness)
    half = normalize(view + wi)
    n_wo = dot(normal, view)
    n_h = dot(normal, half)
    wo_h = dot(view, half)
    valid = (dot(normal, wi) > 0.0) & (n_wo > 0.0)
    d = _ggx_d(n_h, alpha, backend)
    g = _smith_g1(n_wo, alpha, backend) * _smith_g1(dot(normal, wi), alpha, backend)
    fresnel = _fresnel_schlick(wo_h, albedo, backend)
    f = fresnel * (d * g / (4.0 * backend.clip(n_wo, lo=_EPS) * backend.clip(n_wi, lo=_EPS)))[
        ..., None
    ]
    metal_f_cos = f * n_wi[..., None]
    metal_pdf = d * n_h / (4.0 * backend.clip(wo_h, lo=_EPS))
    metal_f_cos = backend.where(valid[..., None], metal_f_cos, backend.zeros_like(metal_f_cos))
    metal_pdf = backend.where(valid, metal_pdf, backend.zeros_like(metal_pdf))

    is_metal = (metallic > 0.5)[..., None]
    f_cos = backend.where(is_metal, metal_f_cos, diff_f_cos)
    pdf = backend.where(metallic > 0.5, metal_pdf, diff_pdf)
    return f_cos, pdf
