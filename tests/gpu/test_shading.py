from __future__ import annotations

import torch

from myraytracer.gpu.geometry import HitBatch, Plane, Sphere
from myraytracer.gpu.scene import PointLight, Scene
from myraytracer.gpu.shading import direct_lighting


def test_direct_lighting_matches_analytic_lambertian_value() -> None:
    # Mirrors v0's test_tracer.py analytic case: a plane at z=-5 lit by a
    # point light directly along its normal at z=-2 (distance 3).
    albedo = torch.tensor([[0.8, 0.5, 0.2]])
    plane = Plane(point=torch.tensor([0.0, 0.0, -5.0]), normal=torch.tensor([0.0, 0.0, 1.0]))
    light = PointLight(
        position=torch.tensor([0.0, 0.0, -2.0]), intensity=torch.tensor([10.0, 10.0, 10.0])
    )
    scene = Scene(objects=[plane], lights=[light])
    hit = HitBatch(
        hit=torch.tensor([True]),
        t=torch.tensor([3.0]),
        point=torch.tensor([[0.0, 0.0, -5.0]]),
        normal=torch.tensor([[0.0, 0.0, 1.0]]),
    )

    result = direct_lighting(hit, albedo, scene)

    distance = 3.0
    cos_theta = 1.0
    expected = albedo * (10.0 * cos_theta / distance**2)

    assert torch.allclose(result, expected)


def test_blocking_primitive_zeroes_light_contribution_and_backward_is_finite() -> None:
    albedo = torch.tensor([[1.0, 1.0, 1.0]], requires_grad=True)
    plane = Plane(point=torch.tensor([0.0, 0.0, -5.0]), normal=torch.tensor([0.0, 0.0, 1.0]))
    blocker = Sphere(center=torch.tensor([0.0, 0.0, -3.5]), radius=torch.tensor(0.3))
    light = PointLight(
        position=torch.tensor([0.0, 0.0, -2.0]), intensity=torch.tensor([10.0, 10.0, 10.0])
    )
    scene = Scene(objects=[plane, blocker], lights=[light])
    hit = HitBatch(
        hit=torch.tensor([True]),
        t=torch.tensor([3.0]),
        point=torch.tensor([[0.0, 0.0, -5.0]]),
        normal=torch.tensor([[0.0, 0.0, 1.0]]),
    )

    result = direct_lighting(hit, albedo, scene)
    result.sum().backward()

    assert torch.allclose(result, torch.zeros_like(result))
    assert albedo.grad is not None
    assert bool(torch.all(torch.isfinite(albedo.grad)))


def test_direct_lighting_gradcheck_wrt_albedo_and_intensity() -> None:
    plane = Plane(
        point=torch.tensor([0.0, 0.0, -5.0], dtype=torch.float64),
        normal=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
    )
    hit = HitBatch(
        hit=torch.tensor([True]),
        t=torch.tensor([3.0], dtype=torch.float64),
        point=torch.tensor([[0.0, 0.0, -5.0]], dtype=torch.float64),
        normal=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64),
    )
    albedo = torch.tensor([[0.8, 0.5, 0.2]], dtype=torch.float64, requires_grad=True)
    intensity = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64, requires_grad=True)
    position = torch.tensor([0.0, 0.0, -2.0], dtype=torch.float64)

    def lighting_fn(albedo: torch.Tensor, intensity: torch.Tensor) -> torch.Tensor:
        light = PointLight(position=position, intensity=intensity)
        scene = Scene(objects=[plane], lights=[light])
        return direct_lighting(hit, albedo, scene)

    assert torch.autograd.gradcheck(lighting_fn, (albedo, intensity))
