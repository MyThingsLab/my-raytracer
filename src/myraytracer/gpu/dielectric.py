from __future__ import annotations

import torch

from myraytracer.gpu.geometry import HitBatch
from myraytracer.gpu.vec import dot


def refract(
    incident: torch.Tensor, normal: torch.Tensor, eta: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched Snell refraction of `incident` through a surface, vectorized.

    `incident` is the unit ray direction (pointing toward the surface) and
    `normal` must already be face-forwarded against it, i.e.
    `dot(incident, normal) <= 0`. `eta` = n_from / n_to, the ratio of the
    refractive index of the medium the ray is leaving to that of the medium
    it is entering, broadcastable to `incident`'s row count. Returns the
    refracted direction (unit vector; meaningless where `tir_mask` is True --
    callers must reflect instead) and a `(N,)` boolean mask marking rows past
    the critical angle (total internal reflection).
    """
    cos_theta_i = torch.clamp(-dot(incident, normal), min=-1.0, max=1.0)
    eta = torch.broadcast_to(eta, cos_theta_i.shape)

    # Clamp before sqrt so TIR rows (sin2_theta_t > 1) still yield a finite
    # `refracted` value instead of NaN, matching the sphere_hit pattern of
    # keeping unused branches finite for torch.where masking downstream.
    sin2_theta_t = torch.clamp(eta * eta * (1.0 - cos_theta_i * cos_theta_i), min=0.0)
    tir_mask = sin2_theta_t > 1.0
    cos_theta_t = torch.sqrt(torch.clamp(1.0 - sin2_theta_t, min=0.0))

    refracted = eta.unsqueeze(-1) * incident + (
        eta * cos_theta_i - cos_theta_t
    ).unsqueeze(-1) * normal

    return refracted, tir_mask


def fresnel_schlick(cos_theta: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    """Schlick's approximation to unpolarized Fresnel reflectance, batched.

    `cos_theta` is the cosine of the angle between the incident ray and the
    face-forwarded surface normal (>= 0), and `eta` = n_from / n_to. Returns
    reflectance in [0, 1], rising smoothly from the normal-incidence value
    `((eta - 1) / (eta + 1)) ** 2` at `cos_theta == 1` to `1` at grazing
    incidence.
    """
    r0 = ((eta - 1.0) / (eta + 1.0)) ** 2
    # Clamp guards against cos_theta drifting slightly outside [0, 1] from
    # upstream floating-point error; pow's integer exponent keeps the
    # gradient well-defined at the clamped boundaries.
    grazing = torch.clamp(1.0 - cos_theta, min=0.0, max=1.0)
    return r0 + (1.0 - r0) * torch.pow(grazing, 5)


def dielectric_scatter(
    ray_d: torch.Tensor,
    hit: HitBatch,
    ior: torch.Tensor,
    rng: torch.Generator,
) -> torch.Tensor:
    """Sample a dielectric (glass) scatter direction per ray, vectorized.

    `ray_d` is the unit incident ray direction and `hit.normal` is the
    surface's geometric normal, *not* pre-face-forwarded: the sign of
    `dot(ray_d, hit.normal)` determines whether each ray is entering
    (negative, normal points against the ray) or exiting (positive) the
    dielectric. `ior` is the material's index of refraction relative to
    vacuum; `eta` is `1 / ior` when entering and `ior` when exiting. Each row
    reflects or refracts by sampling its Fresnel/Schlick reflectance
    probability against the seeded `rng` (same device as `ray_d`), always
    reflecting where refraction would total-internally-reflect. Deterministic
    given a seeded, cloned `rng` state. Returns unit scatter directions,
    shape `(N, 3)`.
    """
    n = ray_d.shape[0]
    entering = dot(ray_d, hit.normal) < 0.0

    eta = torch.where(entering, 1.0 / ior, ior)
    face_normal = torch.where(entering.unsqueeze(-1), hit.normal, -hit.normal)

    refracted, tir_mask = refract(ray_d, face_normal, eta)
    cos_theta_i = torch.clamp(-dot(ray_d, face_normal), min=0.0, max=1.0)
    reflectance = fresnel_schlick(cos_theta_i, eta)

    reflected = ray_d - 2.0 * dot(ray_d, face_normal).unsqueeze(-1) * face_normal

    u = torch.rand(n, generator=rng, dtype=ray_d.dtype, device=ray_d.device)
    reflect_mask = tir_mask | (u < reflectance)

    return torch.where(reflect_mask.unsqueeze(-1), reflected, refracted)
