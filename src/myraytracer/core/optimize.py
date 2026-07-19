from __future__ import annotations

from dataclasses import replace

from myraytracer.core.backend import Array, Backend
from myraytracer.core.camera import Camera
from myraytracer.core.integrator import render
from myraytracer.core.scene import Scene


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
