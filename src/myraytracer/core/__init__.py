from __future__ import annotations

from myraytracer.core.backend import (
    NUMPY,
    Backend,
    backend_of,
    get_backend,
    torch_backend,
)
from myraytracer.core.camera import Camera
from myraytracer.core.geometry import (
    Hit,
    Mesh,
    Plane,
    Quad,
    Sphere,
    hit_primitive,
    mesh_hit,
)
from myraytracer.core.integrator import integrate
from myraytracer.core.integrator import render as render_image
from myraytracer.core.linalg import cross, dot, length, normalize
from myraytracer.core.material import Material
from myraytracer.core.mesh import load_obj
from myraytracer.core.ray import RayBatch
from myraytracer.core.sampling import sample_cosine_hemisphere
from myraytracer.core.scene import PointLight, Scene, SceneHit

__all__ = [
    "NUMPY",
    "Backend",
    "Camera",
    "Hit",
    "Material",
    "Mesh",
    "Plane",
    "PointLight",
    "Quad",
    "RayBatch",
    "Scene",
    "SceneHit",
    "Sphere",
    "backend_of",
    "cross",
    "dot",
    "get_backend",
    "hit_primitive",
    "integrate",
    "length",
    "load_obj",
    "mesh_hit",
    "normalize",
    "render_image",
    "sample_cosine_hemisphere",
    "torch_backend",
]
