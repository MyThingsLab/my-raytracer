from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Ray:
    origin: torch.Tensor  # (..., 3)
    direction: torch.Tensor  # (..., 3), assumed pre-normalized by the caller

    def at(self, t: torch.Tensor) -> torch.Tensor:
        return self.origin + self.direction * t.unsqueeze(-1)
