from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from myraytracer.gpu.vec import cross, normalize


@dataclass(frozen=True)
class Camera:
    origin: torch.Tensor  # (3,)
    look_at: torch.Tensor  # (3,)
    up: torch.Tensor  # (3,)
    vfov_degrees: float
    aspect_ratio: float

    def rays(
        self, width: int, height: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        origin = self.origin.to(device)
        look_at = self.look_at.to(device)
        up = self.up.to(device)
        dtype = origin.dtype

        forward = normalize((look_at - origin).unsqueeze(0)).squeeze(0)
        right = normalize(cross(forward, up).unsqueeze(0)).squeeze(0)
        true_up = cross(right, forward)

        half_height = math.tan(math.radians(self.vfov_degrees) / 2)
        half_width = half_height * self.aspect_ratio

        py, px = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        px = px.reshape(-1)
        py = py.reshape(-1)

        u = (2 * px / width - 1) * half_width
        v = (1 - 2 * py / height) * half_height

        directions = forward + right * u.unsqueeze(-1) + true_up * v.unsqueeze(-1)
        directions = normalize(directions)
        origins = origin.expand_as(directions)

        return origins, directions
