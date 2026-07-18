from __future__ import annotations

import torch


def dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(dim=-1)


def cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.cross(a, b, dim=-1)


def length(v: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(v, dim=-1)


def normalize(v: torch.Tensor) -> torch.Tensor:
    # Raise on zero-length rows rather than silently dividing by zero,
    # mirroring v0's Vec3.normalized() -- a degenerate row should surface
    # as an error, not propagate as inf/nan.
    norm = length(v)
    if bool(torch.any(norm == 0.0)):
        raise ValueError("cannot normalize a zero-length vector")
    return v / norm.unsqueeze(-1)


def device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
