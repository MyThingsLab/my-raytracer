from __future__ import annotations

from dataclasses import dataclass

from myraytracer.core.backend import Array


@dataclass(frozen=True)
class RayBatch:
    """A batch of rays as parallel origin/direction arrays, shaped (..., 3).

    Directions are assumed pre-normalized by the caller (as in the scalar
    Ray). Backend-agnostic: origin/direction are both numpy arrays or both
    torch tensors.
    """

    origin: Array
    direction: Array

    def at(self, t: Array) -> Array:
        # t has the batch shape (...); the trailing axis is added so it
        # broadcasts against the (..., 3) direction.
        return self.origin + self.direction * t[..., None]
