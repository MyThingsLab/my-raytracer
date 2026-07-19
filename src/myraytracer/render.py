from __future__ import annotations

import math

import numpy as np

from myraytracer.camera import Camera
from myraytracer.core import Camera as CoreCamera
from myraytracer.core import Material as CoreMaterial
from myraytracer.core import Plane as CorePlane
from myraytracer.core import PointLight as CorePointLight
from myraytracer.core import Quad as CoreQuad
from myraytracer.core import Scene as CoreScene
from myraytracer.core import Sphere as CoreSphere
from myraytracer.core import get_backend, render_image
from myraytracer.geometry import Plane, Quad, Sphere
from myraytracer.scene import Scene
from myraytracer.vec import Vec3

# Bridge from the scalar scene description (Vec3-based, produced by
# `load_scene`) to the backend-agnostic `core` scene the batched integrator
# consumes. Kept here as a thin transitional adapter -- a later phase makes
# scene loading emit `core` types directly and drops these.


def _tuple(vec: Vec3) -> tuple[float, float, float]:
    return (vec.x, vec.y, vec.z)


def _core_material(material) -> CoreMaterial:
    return CoreMaterial(albedo=_tuple(material.albedo), emission=_tuple(material.emission))


def _core_object(obj):
    if isinstance(obj, Sphere):
        return CoreSphere(
            center=_tuple(obj.center), radius=obj.radius, material=_core_material(obj.material)
        )
    if isinstance(obj, Plane):
        return CorePlane(
            point=_tuple(obj.point),
            normal=_tuple(obj.normal),
            material=_core_material(obj.material),
        )
    if isinstance(obj, Quad):
        return CoreQuad(
            corner=_tuple(obj.corner),
            edge1=_tuple(obj.edge1),
            edge2=_tuple(obj.edge2),
            material=_core_material(obj.material),
        )
    raise TypeError(f"unsupported object type: {type(obj).__name__}")


def _core_scene(scene: Scene) -> CoreScene:
    return CoreScene(
        objects=[_core_object(obj) for obj in scene.objects],
        lights=[
            CorePointLight(position=_tuple(light.position), intensity=_tuple(light.intensity))
            for light in scene.lights
        ],
    )


def _core_camera(camera: Camera) -> CoreCamera:
    return CoreCamera(
        origin=_tuple(camera.origin),
        look_at=_tuple(camera.look_at),
        up=_tuple(camera.up),
        vfov_degrees=camera.vfov_degrees,
        aspect_ratio=camera.aspect_ratio,
    )


def render(
    scene: Scene,
    camera: Camera,
    *,
    width: int,
    height: int,
    spp: int,
    max_depth: int,
    seed: int,
    backend: str = "cpu",
) -> np.ndarray:
    image = render_image(
        _core_scene(scene),
        _core_camera(camera),
        width=width,
        height=height,
        spp=spp,
        max_depth=max_depth,
        seed=seed,
        backend=get_backend(backend),
    )
    if hasattr(image, "detach"):  # a torch tensor -> bring it back to numpy
        image = image.detach().cpu().numpy()
    return np.asarray(image, dtype=np.float64)


def render_gbuffers(
    scene: Scene, camera: Camera, *, width: int, height: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render per-pixel first-hit guide buffers for denoising (a-trous/SVGF
    style edge-stopping): world-space normal (H,W,3), surface albedo
    (H,W,3), and hit distance / depth (H,W,1). Sibling to `render`, additive
    and opt-in -- the plain path-traced `render` path never calls this.

    Guide buffers are geometric (first-hit only, no path tracing), so a
    single ray through the pixel center is enough -- unlike `render`, there
    is no Monte Carlo noise to average over with `spp` samples.
    """
    normal = np.zeros((height, width, 3), dtype=np.float64)
    albedo = np.zeros((height, width, 3), dtype=np.float64)
    depth = np.zeros((height, width, 1), dtype=np.float64)

    for py in range(height):
        for px in range(width):
            ray = camera.ray_for_pixel(px + 0.5, py + 0.5, width, height)
            hit = scene.nearest_hit(ray, t_min=1e-4, t_max=math.inf)
            if hit is None:
                continue
            normal[py, px] = (hit.normal.x, hit.normal.y, hit.normal.z)
            albedo[py, px] = (
                hit.material.albedo.x,
                hit.material.albedo.y,
                hit.material.albedo.z,
            )
            depth[py, px, 0] = hit.t

    return normal, albedo, depth
