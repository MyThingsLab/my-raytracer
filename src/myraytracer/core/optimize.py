from __future__ import annotations

from dataclasses import replace

from myraytracer.core.backend import Array, Backend
from myraytracer.core.camera import Camera
from myraytracer.core.integrator import render
from myraytracer.core.scene import Scene
from myraytracer.core.splat import Gaussians, rasterize


def fit_albedo(
    scene: Scene,
    camera: Camera,
    target: Array,
    *,
    object_index: int,
    width: int,
    height: int,
    spp: int,
    max_depth: int,
    seed: int,
    backend: Backend,
    lr: float = 0.05,
    steps: int = 200,
) -> Array:
    """Recover one object's Lambertian albedo by gradient descent against `target`.

    Requires the torch backend: the scene is re-rendered through the same
    unified wavefront integrator every step, with `scene.objects[object_index]`'s
    albedo as the only free parameter (everything else -- geometry, lights, the
    rest of the scene's materials -- stays fixed). Proves the array-API core is
    differentiable end-to-end, not just the deprecated `gpu` path this replaces.
    """
    if not backend.is_torch:
        raise ValueError("fit_albedo requires the torch backend (needs autograd)")
    torch = backend.xp

    obj = scene.objects[object_index]
    albedo = backend.asarray(obj.material.albedo).detach().clone().requires_grad_(True)
    fit_object = replace(obj, material=replace(obj.material, albedo=albedo))
    fit_objects = list(scene.objects)
    fit_objects[object_index] = fit_object
    fit_scene = replace(scene, objects=fit_objects)

    optimizer = torch.optim.Adam([albedo], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        image = render(
            fit_scene,
            camera,
            width=width,
            height=height,
            spp=spp,
            max_depth=max_depth,
            seed=seed,
            backend=backend,
        )
        loss = ((image - target) ** 2).mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            albedo.clamp_(0.0, 1.0)

    return albedo.detach()


def fit_gaussians(
    gaussians: Gaussians,
    camera: Camera,
    target: Array,
    *,
    width: int,
    height: int,
    backend: Backend,
    lr: float = 0.02,
    steps: int = 200,
) -> Gaussians:
    """Fit a set of 3D Gaussians to `target` by gradient descent through `rasterize`.

    Requires the torch backend: all five Gaussian arrays become leaves, the
    scene is re-splatted every step, and Adam minimizes MSE against the target
    image. `opacity` and `color` are clamped to [0, 1] after each step;
    `mean`, `scale`, and `quat` are left free (`rasterize` re-normalizes the
    quaternion and only reads `scale` squared, so their sign/norm is
    unconstrained). Returns a detached copy of the fitted Gaussians.
    """
    if not backend.is_torch:
        raise ValueError("fit_gaussians requires the torch backend (needs autograd)")
    torch = backend.xp

    def _leaf(array: Array) -> Array:
        return backend.asarray(array).detach().clone().requires_grad_(True)

    fit = Gaussians(
        mean=_leaf(gaussians.mean),
        scale=_leaf(gaussians.scale),
        quat=_leaf(gaussians.quat),
        opacity=_leaf(gaussians.opacity),
        color=_leaf(gaussians.color),
    )
    params = [fit.mean, fit.scale, fit.quat, fit.opacity, fit.color]
    optimizer = torch.optim.Adam(params, lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        image = rasterize(fit, camera, width, height, backend=backend)
        loss = ((image - target) ** 2).mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            fit.opacity.clamp_(0.0, 1.0)
            fit.color.clamp_(0.0, 1.0)

    return Gaussians(
        mean=fit.mean.detach(),
        scale=fit.scale.detach(),
        quat=fit.quat.detach(),
        opacity=fit.opacity.detach(),
        color=fit.color.detach(),
    )
