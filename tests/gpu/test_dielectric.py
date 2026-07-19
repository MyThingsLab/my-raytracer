from __future__ import annotations

import math

import pytest
import torch

from myraytracer.gpu.dielectric import dielectric_scatter, fresnel_schlick, refract
from myraytracer.gpu.geometry import HitBatch


def test_fresnel_schlick_matches_analytic_value_at_normal_incidence() -> None:
    eta = torch.tensor([1.0 / 1.5, 1.5, 2.0])
    cos_theta = torch.ones(3)

    result = fresnel_schlick(cos_theta, eta)

    expected = ((eta - 1.0) / (eta + 1.0)) ** 2
    assert torch.allclose(result, expected)


def test_fresnel_schlick_approaches_one_at_grazing_incidence() -> None:
    cos_theta = torch.tensor([0.0])
    eta = torch.tensor([1.5])

    result = fresnel_schlick(cos_theta, eta)

    assert result.item() == pytest.approx(1.0)


def test_fresnel_schlick_gradcheck_wrt_cos_theta() -> None:
    cos_theta = torch.tensor([0.3, 0.7], dtype=torch.float64, requires_grad=True)
    eta = torch.tensor([1.5, 0.6667], dtype=torch.float64)

    def schlick_fn(cos_theta: torch.Tensor) -> torch.Tensor:
        return fresnel_schlick(cos_theta, eta)

    assert torch.autograd.gradcheck(schlick_fn, (cos_theta,))


def test_refract_obeys_snell_law_for_known_angle() -> None:
    # 30 degree incidence, air (n=1.0) into glass (n=1.5): eta = n_from/n_to.
    theta_i = math.radians(30.0)
    incident = torch.tensor([[math.sin(theta_i), 0.0, -math.cos(theta_i)]])
    normal = torch.tensor([[0.0, 0.0, 1.0]])
    eta = torch.tensor([1.0 / 1.5])

    refracted, tir_mask = refract(incident, normal, eta)

    sin_theta_t = (1.0 / 1.5) * math.sin(theta_i)
    theta_t = math.asin(sin_theta_t)
    expected = torch.tensor([[math.sin(theta_t), 0.0, -math.cos(theta_t)]])

    assert not bool(tir_mask[0])
    assert torch.allclose(refracted, expected, atol=1e-6)


def test_refract_normal_incidence_passes_through_unbent() -> None:
    incident = torch.tensor([[0.0, 0.0, -1.0]])
    normal = torch.tensor([[0.0, 0.0, 1.0]])
    eta = torch.tensor([1.0 / 1.5])

    refracted, tir_mask = refract(incident, normal, eta)

    assert not bool(tir_mask[0])
    assert torch.allclose(refracted, incident, atol=1e-6)


def test_refract_past_critical_angle_sets_tir_mask() -> None:
    # Glass (n=1.5) into air (n=1.0): critical angle is asin(1/1.5) ~= 41.8
    # degrees, so 60 degrees of incidence must total-internally-reflect.
    theta_i = math.radians(60.0)
    incident = torch.tensor([[math.sin(theta_i), 0.0, -math.cos(theta_i)]])
    normal = torch.tensor([[0.0, 0.0, 1.0]])
    eta = torch.tensor([1.5])

    _, tir_mask = refract(incident, normal, eta)

    assert bool(tir_mask[0])


def test_refract_batch_mixes_tir_and_transmission_rows() -> None:
    theta_shallow = math.radians(10.0)
    theta_steep = math.radians(60.0)
    incident = torch.tensor(
        [
            [math.sin(theta_shallow), 0.0, -math.cos(theta_shallow)],
            [math.sin(theta_steep), 0.0, -math.cos(theta_steep)],
        ]
    )
    normal = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    eta = torch.tensor([1.5, 1.5])

    _, tir_mask = refract(incident, normal, eta)

    assert not bool(tir_mask[0])
    assert bool(tir_mask[1])


def _glancing_hit(n: int) -> HitBatch:
    return HitBatch(
        hit=torch.ones(n, dtype=torch.bool),
        t=torch.ones(n),
        point=torch.zeros(n, 3),
        normal=torch.tensor([[0.0, 0.0, 1.0]]).expand(n, 3),
    )


def test_dielectric_scatter_forces_reflection_past_critical_angle() -> None:
    # Ray already inside the glass (exiting), well past the critical angle,
    # must reflect regardless of the Fresnel sample drawn.
    theta_i = math.radians(60.0)
    ray_d = torch.tensor([[math.sin(theta_i), 0.0, math.cos(theta_i)]])
    hit = _glancing_hit(1)
    ior = torch.tensor(1.5)
    rng = torch.Generator().manual_seed(0)

    scatter = dielectric_scatter(ray_d, hit, ior, rng)

    face_normal = -hit.normal
    expected_reflect = ray_d - 2.0 * (ray_d * face_normal).sum(-1, keepdim=True) * face_normal
    assert torch.allclose(scatter, expected_reflect, atol=1e-6)


def test_dielectric_scatter_returns_unit_vectors() -> None:
    ray_d = torch.tensor([[0.3, 0.0, -math.sqrt(1.0 - 0.3**2)]])
    hit = _glancing_hit(1)
    ior = torch.tensor(1.5)
    rng = torch.Generator().manual_seed(1)

    scatter = dielectric_scatter(ray_d, hit, ior, rng)

    assert torch.linalg.vector_norm(scatter, dim=-1).item() == pytest.approx(1.0, abs=1e-5)


def test_dielectric_scatter_is_deterministic_given_same_seed() -> None:
    ray_d = torch.tensor(
        [[0.3, 0.0, -0.95], [0.1, 0.2, -0.97], [-0.2, 0.1, -0.97], [0.0, 0.0, -1.0]]
    )
    ray_d = ray_d / torch.linalg.vector_norm(ray_d, dim=-1, keepdim=True)
    hit = _glancing_hit(4)
    ior = torch.tensor(1.5)

    rng_a = torch.Generator().manual_seed(42)
    result_a = dielectric_scatter(ray_d, hit, ior, rng_a)

    rng_b = torch.Generator().manual_seed(42)
    result_b = dielectric_scatter(ray_d, hit, ior, rng_b)

    assert torch.equal(result_a, result_b)


def test_refract_entering_then_exiting_is_parallel_to_original_ray() -> None:
    # Snell's law reversibility (the parallel-slab theorem): a ray refracted
    # into glass and then back out through a parallel face must exit
    # traveling parallel to its original direction.
    theta_i = math.radians(20.0)
    entering_ray = torch.tensor([[math.sin(theta_i), 0.0, -math.cos(theta_i)]])
    entry_normal = torch.tensor([[0.0, 0.0, 1.0]])

    inside_ray, entering_tir = refract(entering_ray, entry_normal, torch.tensor([1.0 / 1.5]))
    # Exit face's outward normal is the entry face's, mirrored: the exiting
    # ray's face-forwarded normal (against `inside_ray`) is `entry_normal`.
    exit_ray, exiting_tir = refract(inside_ray, entry_normal, torch.tensor([1.5]))

    assert not bool(entering_tir[0])
    assert not bool(exiting_tir[0])
    assert torch.allclose(exit_ray, entering_ray, atol=1e-6)
