from __future__ import annotations

import torch

from myraytracer.gpu.geometry import Plane
from myraytracer.gpu.pathtracer import trace
from myraytracer.gpu.scene import PointLight, Scene


def _plane_scene(albedo: torch.Tensor | None = None, emission: torch.Tensor | None = None) -> Scene:
    plane = Plane(point=torch.tensor([0.0, 0.0, -5.0]), normal=torch.tensor([0.0, 0.0, 1.0]))
    light = PointLight(
        position=torch.tensor([0.0, 0.0, -2.0]), intensity=torch.tensor([10.0, 10.0, 10.0])
    )
    kwargs = {}
    if albedo is not None:
        kwargs["albedo"] = albedo
    if emission is not None:
        kwargs["emission"] = emission
    return Scene(objects=[plane], lights=[light], **kwargs)


def test_trace_matches_analytic_direct_lighting_at_max_depth_1() -> None:
    # Single infinite plane: any bounce ray leaves through the normal's
    # hemisphere (away from the plane) and can never hit the same plane
    # again, so in expectation this must match v0's closed-form direct
    # term exactly, the same scene as test_tracer.py's analytic check.
    albedo = torch.tensor([0.8, 0.5, 0.2])
    scene = _plane_scene(albedo=albedo)
    n = 4096
    ray_o = torch.tensor([[0.0, 0.0, 0.0]]).expand(n, -1)
    ray_d = torch.tensor([[0.0, 0.0, -1.0]]).expand(n, -1)
    rng = torch.Generator().manual_seed(0)

    result = trace(scene, ray_o, ray_d, max_depth=1, rng=rng)
    mean = result.mean(dim=0)

    distance = 3.0
    cos_theta = 1.0
    expected = albedo * (10.0 * cos_theta / distance**2)

    assert torch.allclose(mean, expected, rtol=0.05, atol=0.02)


def test_trace_adds_emission_at_max_depth_0() -> None:
    albedo = torch.tensor([0.8, 0.5, 0.2])
    emission = torch.tensor([0.1, 0.1, 0.1])
    scene = _plane_scene(albedo=albedo, emission=emission)
    ray_o = torch.tensor([[0.0, 0.0, 0.0]])
    ray_d = torch.tensor([[0.0, 0.0, -1.0]])
    rng = torch.Generator().manual_seed(0)

    result = trace(scene, ray_o, ray_d, max_depth=0, rng=rng)

    distance = 3.0
    cos_theta = 1.0
    expected = emission + albedo * (10.0 * cos_theta / distance**2)

    assert torch.allclose(result[0], expected)


def _corner_scene(albedo: torch.Tensor | None = None) -> Scene:
    # Floor and ceiling facing each other: a ray hitting the floor
    # cosine-samples a bounce direction whose y-component is always >= 0
    # (the hemisphere opens toward +y around the floor's up-facing normal),
    # so it is guaranteed -- not just likely -- to reach the (unbounded)
    # ceiling plane, letting the ceiling's direct lighting bleed back
    # through the floor's indirect term.
    floor = Plane(point=torch.tensor([0.0, -1.0, 0.0]), normal=torch.tensor([0.0, 1.0, 0.0]))
    ceiling = Plane(point=torch.tensor([0.0, 2.0, 0.0]), normal=torch.tensor([0.0, -1.0, 0.0]))
    light = PointLight(
        position=torch.tensor([0.0, 0.5, -3.0]), intensity=torch.tensor([15.0, 15.0, 15.0])
    )
    kwargs = {}
    if albedo is not None:
        kwargs["albedo"] = albedo
    return Scene(objects=[floor, ceiling], lights=[light], **kwargs)


def _corner_ray(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    origin = torch.tensor([[0.0, 0.0, 0.0]]).expand(n, -1)
    direction = torch.nn.functional.normalize(
        torch.tensor([[0.0, -0.3, -1.0]]).expand(n, -1), dim=-1
    )
    return origin, direction


def test_two_surface_scene_shows_indirect_contribution_at_max_depth_2() -> None:
    scene = _corner_scene(albedo=torch.tensor([0.9, 0.9, 0.9]))
    n = 4096
    ray_o, ray_d = _corner_ray(n)

    direct_only = trace(scene, ray_o, ray_d, max_depth=0, rng=torch.Generator().manual_seed(1))
    with_bounces = trace(scene, ray_o, ray_d, max_depth=2, rng=torch.Generator().manual_seed(1))

    direct_mean = direct_only.mean(dim=0)
    bounced_mean = with_bounces.mean(dim=0)

    # The ceiling's direct illumination reliably bounces back off the floor
    # (see _corner_scene), so allowing bounces must raise every channel's
    # mean radiance well above the direct-only baseline.
    assert bool(torch.all(bounced_mean > direct_mean * 1.1))


def test_trace_is_deterministic_given_same_seed() -> None:
    scene = _corner_scene()
    n = 64
    ray_o, ray_d = _corner_ray(n)

    result_a = trace(scene, ray_o, ray_d, max_depth=3, rng=torch.Generator().manual_seed(42))
    result_b = trace(scene, ray_o, ray_d, max_depth=3, rng=torch.Generator().manual_seed(42))

    assert torch.equal(result_a, result_b)


def test_trace_gradcheck_wrt_albedo_through_two_bounces() -> None:
    floor = Plane(
        point=torch.tensor([0.0, -1.0, 0.0], dtype=torch.float64),
        normal=torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
    )
    ceiling = Plane(
        point=torch.tensor([0.0, 2.0, 0.0], dtype=torch.float64),
        normal=torch.tensor([0.0, -1.0, 0.0], dtype=torch.float64),
    )
    light = PointLight(
        position=torch.tensor([0.0, 0.5, -3.0], dtype=torch.float64),
        intensity=torch.tensor([15.0, 15.0, 15.0], dtype=torch.float64),
    )
    ray_o = torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=torch.float64)
    ray_d = torch.nn.functional.normalize(
        torch.tensor([[0.0, -0.3, -1.0], [0.0, -0.3, -1.0]], dtype=torch.float64), dim=-1
    )
    albedo = torch.tensor([0.5, 0.4, 0.3], dtype=torch.float64, requires_grad=True)

    def trace_fn(albedo: torch.Tensor) -> torch.Tensor:
        scene = Scene(objects=[floor, ceiling], lights=[light], albedo=albedo)
        # Reseeded with a fixed value on every call so gradcheck's finite
        # differences see identical random draws at each perturbed point --
        # the bounce direction depends only on rng and the hit normal, never
        # on albedo, so this keeps the ray/hit topology fixed across calls.
        rng = torch.Generator().manual_seed(7)
        return trace(scene, ray_o, ray_d, max_depth=2, rng=rng)

    assert torch.autograd.gradcheck(trace_fn, (albedo,))
