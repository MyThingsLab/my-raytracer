from __future__ import annotations

from myraytracer.core.backend import Array, backend_of

# Batched vector algebra over the trailing axis: every function accepts arrays
# shaped (..., 3) -- a bare (3,) vector, a batch (N, 3) of rays, or any
# broadcastable stack -- and reduces/operates on that last axis, on whichever
# backend the inputs live on. This replaces both the scalar Vec3 methods and
# the torch-only myraytracer.gpu.vec helpers with one implementation.


def dot(a: Array, b: Array) -> Array:
    return (a * b).sum(-1)


def cross(a: Array, b: Array) -> Array:
    return backend_of(a).cross(a, b)


def length(v: Array) -> Array:
    return dot(v, v) ** 0.5


def normalize(v: Array) -> Array:
    # Raise on any zero-length row rather than dividing by zero: a degenerate
    # vector should surface as an error, not propagate as inf/nan -- mirroring
    # both Vec3.normalized() and gpu.vec.normalize().
    norm = length(v)
    if bool((norm == 0).any()):
        raise ValueError("cannot normalize a zero-length vector")
    return v / norm[..., None]
