from __future__ import annotations

from dataclasses import dataclass, field

import torch

from myraytracer.gpu.geometry import HitBatch, Plane, Sphere, plane_hit, sphere_hit


@dataclass(frozen=True)
class PointLight:
    position: torch.Tensor  # (3,)
    intensity: torch.Tensor  # (3,)

    def __post_init__(self) -> None:
        if bool(torch.any(self.intensity < 0.0)):
            raise ValueError("light intensity components must be non-negative")


@dataclass
class Scene:
    objects: list[Sphere | Plane]
    lights: list[PointLight]
    # Uniform Lambertian albedo shared by every object in the scene -- v1
    # has no per-object material system yet, so this is the single
    # differentiable surface parameter `render` shades against.
    albedo: torch.Tensor = field(default_factory=lambda: torch.tensor([1.0, 1.0, 1.0]))

    def nearest_hit(
        self,
        ray_origin: torch.Tensor,
        ray_dir: torch.Tensor,
        t_min: float,
        t_max: float | torch.Tensor,
    ) -> HitBatch:
        n = ray_origin.shape[0]
        device = ray_origin.device
        dtype = ray_origin.dtype

        acc_t = torch.as_tensor(t_max, dtype=dtype, device=device).expand(n).clone()
        acc_hit = torch.zeros(n, dtype=torch.bool, device=device)
        acc_point = torch.zeros_like(ray_origin)
        acc_normal = torch.zeros_like(ray_origin)

        for obj in self.objects:
            hit_fn = sphere_hit if isinstance(obj, Sphere) else plane_hit
            result = hit_fn(obj, ray_origin, ray_dir, t_min=t_min, t_max=t_max)

            # Masked min over t across primitives: a candidate only replaces
            # the running nearest hit when it actually hit and is closer,
            # so gradients flow only through the primitive that owns the
            # nearest hit for each ray.
            closer = result.hit & (result.t < acc_t)
            acc_t = torch.where(closer, result.t, acc_t)
            acc_hit = acc_hit | closer
            mask = closer.unsqueeze(-1)
            acc_point = torch.where(mask, result.point, acc_point)
            acc_normal = torch.where(mask, result.normal, acc_normal)

        return HitBatch(hit=acc_hit, t=acc_t, point=acc_point, normal=acc_normal)
