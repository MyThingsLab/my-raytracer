from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# An array is either a numpy ndarray (CPU path) or a torch Tensor (GPU /
# differentiable path). torch is imported lazily so the numpy-only install
# never pays for it.
Array = Any


def _torch_module() -> Any:
    import torch

    return torch


@dataclass(frozen=True)
class Backend:
    """Thin array-API shim over one array library (numpy or torch).

    Holds the array module plus the handful of ops whose call signatures
    differ between numpy and torch (cross, and the constructors that need a
    torch device/dtype). Batched core code -- linalg, ray, camera, and later
    geometry/integrators -- is written once against this seam and runs on
    either backend; differentiability is simply a property of the torch one.
    """

    name: str
    xp: Any
    device: Any = None
    dtype: Any = None

    @property
    def is_torch(self) -> bool:
        return self.name == "torch"

    def asarray(self, data: Any) -> Array:
        if self.is_torch:
            return self.xp.as_tensor(data, dtype=self.dtype, device=self.device)
        return self.xp.asarray(data, dtype=self.dtype)

    def arange(self, n: int) -> Array:
        if self.is_torch:
            return self.xp.arange(n, dtype=self.dtype, device=self.device)
        return self.xp.arange(n, dtype=self.dtype)

    def meshgrid(self, a: Array, b: Array) -> tuple[Array, Array]:
        return self.xp.meshgrid(a, b, indexing="ij")

    def cross(self, a: Array, b: Array) -> Array:
        if self.is_torch:
            return self.xp.linalg.cross(a, b, dim=-1)
        return self.xp.cross(a, b, axis=-1)

    def broadcast_to(self, x: Array, shape: tuple[int, ...]) -> Array:
        return self.xp.broadcast_to(x, shape)

    def where(self, cond: Array, a: Array, b: Array) -> Array:
        return self.xp.where(cond, a, b)

    def clip(self, x: Array, lo: float | None = None, hi: float | None = None) -> Array:
        # numpy spells it clip(x, lo, hi); torch spells it clamp(x, min=, max=).
        if self.is_torch:
            return self.xp.clamp(x, min=lo, max=hi)
        return self.xp.clip(x, lo, hi)

    def zeros_like(self, x: Array) -> Array:
        return self.xp.zeros_like(x)

    def ones_like(self, x: Array) -> Array:
        return self.xp.ones_like(x)

    def full_like(self, x: Array, value: float) -> Array:
        return self.xp.full_like(x, value)

    def rng(self, seed: int) -> Any:
        # An explicit generator object so a render is reproducible from a seed
        # without touching global RNG state.
        if self.is_torch:
            generator = self.xp.Generator(device=self.device)
            generator.manual_seed(int(seed))
            return generator
        return self.xp.random.Generator(self.xp.random.PCG64(seed))

    def random(self, generator: Any, n: int) -> Array:
        # `n` uniform [0, 1) draws as a (n,) array on this backend.
        if self.is_torch:
            return self.xp.rand(n, generator=generator, device=self.device, dtype=self.dtype)
        return generator.random(n)


NUMPY = Backend(name="numpy", xp=np)


def torch_backend(*, device: Any = None, dtype: Any = None) -> Backend:
    torch = _torch_module()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = torch.float32
    return Backend(name="torch", xp=torch, device=device, dtype=dtype)


def get_backend(kind: str = "cpu", **kwargs: Any) -> Backend:
    key = kind.lower()
    if key in ("cpu", "numpy"):
        return NUMPY
    if key in ("gpu", "torch", "cuda"):
        return torch_backend(**kwargs)
    raise ValueError(f"unknown backend {kind!r} (expected 'cpu'/'numpy' or 'gpu'/'torch')")


def backend_of(x: Array) -> Backend:
    # Dispatch by array type so batched ops can be called as `op(a, b)` without
    # threading a Backend argument through every call site. numpy is checked
    # first, so a numpy-only install never imports torch here.
    if isinstance(x, np.ndarray):
        return NUMPY
    torch = _torch_module()
    if isinstance(x, torch.Tensor):
        return Backend(name="torch", xp=torch, device=x.device, dtype=x.dtype)
    raise TypeError(f"not a supported array type: {type(x).__name__}")
