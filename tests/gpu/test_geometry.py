from __future__ import annotations

import math

import pytest
import torch

from myraytracer.gpu.geometry import Plane, Sphere, plane_hit, sphere_hit


def test_sphere_hit_returns_nearest_root() -> None:
    sphere = Sphere(center=torch.tensor([0.0, 0.0, -5.0]), radius=torch.tensor(1.0))
    origin = torch.tensor([[0.0, 0.0, 0.0]])
    direction = torch.tensor([[0.0, 0.0, -1.0]])

    result = sphere_hit(sphere, origin, direction, t_min=0.001, t_max=math.inf)

    assert bool(result.hit[0])
    assert result.t[0].item() == pytest.approx(4.0)
    assert torch.allclose(result.point[0], torch.tensor([0.0, 0.0, -4.0]))
    assert torch.allclose(result.normal[0], torch.tensor([0.0, 0.0, 1.0]))


def test_sphere_miss_returns_false() -> None:
    sphere = Sphere(center=torch.tensor([0.0, 0.0, -5.0]), radius=torch.tensor(1.0))
    origin = torch.tensor([[0.0, 0.0, 0.0]])
    direction = torch.tensor([[0.0, 1.0, 0.0]])

    result = sphere_hit(sphere, origin, direction, t_min=0.001, t_max=math.inf)

    assert not bool(result.hit[0])


def test_sphere_tangent_hit() -> None:
    sphere = Sphere(center=torch.tensor([0.0, 0.0, -5.0]), radius=torch.tensor(1.0))
    origin = torch.tensor([[1.0, 0.0, 0.0]])
    direction = torch.tensor([[0.0, 0.0, -1.0]])

    result = sphere_hit(sphere, origin, direction, t_min=0.001, t_max=math.inf)

    assert bool(result.hit[0])
    assert result.t[0].item() == pytest.approx(5.0)
    assert torch.allclose(result.point[0], torch.tensor([1.0, 0.0, -5.0]))
    assert torch.allclose(result.normal[0], torch.tensor([1.0, 0.0, 0.0]), atol=1e-6)


def test_sphere_hit_outside_t_range_is_false() -> None:
    sphere = Sphere(center=torch.tensor([0.0, 0.0, -5.0]), radius=torch.tensor(1.0))
    origin = torch.tensor([[0.0, 0.0, 0.0]])
    direction = torch.tensor([[0.0, 0.0, -1.0]])

    result = sphere_hit(sphere, origin, direction, t_min=0.001, t_max=3.0)

    assert not bool(result.hit[0])


def test_plane_hit_returns_expected_t() -> None:
    plane = Plane(point=torch.tensor([0.0, 0.0, -5.0]), normal=torch.tensor([0.0, 0.0, 1.0]))
    origin = torch.tensor([[0.0, 0.0, 0.0]])
    direction = torch.tensor([[0.0, 0.0, -1.0]])

    result = plane_hit(plane, origin, direction, t_min=0.001, t_max=math.inf)

    assert bool(result.hit[0])
    assert result.t[0].item() == pytest.approx(5.0)
    assert torch.allclose(result.point[0], torch.tensor([0.0, 0.0, -5.0]))
    assert torch.allclose(result.normal[0], torch.tensor([0.0, 0.0, 1.0]))


def test_plane_miss_when_parallel_to_ray() -> None:
    plane = Plane(point=torch.tensor([0.0, 0.0, -5.0]), normal=torch.tensor([0.0, 0.0, 1.0]))
    origin = torch.tensor([[0.0, 0.0, 0.0]])
    direction = torch.tensor([[1.0, 0.0, 0.0]])

    result = plane_hit(plane, origin, direction, t_min=0.001, t_max=math.inf)

    assert not bool(result.hit[0])


def test_plane_miss_when_intersection_outside_t_range() -> None:
    plane = Plane(point=torch.tensor([0.0, 0.0, -5.0]), normal=torch.tensor([0.0, 0.0, 1.0]))
    origin = torch.tensor([[0.0, 0.0, 0.0]])
    direction = torch.tensor([[0.0, 0.0, 1.0]])

    result = plane_hit(plane, origin, direction, t_min=0.001, t_max=math.inf)

    assert not bool(result.hit[0])


def test_sphere_hit_batch_of_eight_mixed_hits_and_misses() -> None:
    sphere = Sphere(center=torch.tensor([0.0, 0.0, -5.0]), radius=torch.tensor(1.0))
    origins = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # straight hit, t=4
            [0.0, 5.0, 0.0],  # miss, points away
            [1.0, 0.0, 0.0],  # tangent hit, t=5
            [0.0, 0.0, 0.0],  # hit but outside t_max
            [0.0, 0.0, 0.0],  # miss, orthogonal direction
            [2.0, 0.0, 0.0],  # miss, misses sphere entirely
            [0.0, 0.0, 0.0],  # hit, t=4
            [0.0, 0.0, 10.0],  # hit from behind through origin side
        ]
    )
    directions = torch.tensor(
        [
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    t_maxes = [math.inf, math.inf, math.inf, 3.0, math.inf, math.inf, math.inf, math.inf]
    expected_hit = [True, False, True, False, False, False, True, True]
    expected_t = [4.0, None, 5.0, None, None, None, 4.0, 14.0]

    for row in range(8):
        result = sphere_hit(
            sphere,
            origins[row : row + 1],
            directions[row : row + 1],
            t_min=0.001,
            t_max=t_maxes[row],
        )
        assert bool(result.hit[0]) is expected_hit[row], row
        if expected_t[row] is not None:
            assert result.t[0].item() == pytest.approx(expected_t[row]), row


def test_sphere_gradcheck_wrt_center_and_radius() -> None:
    center = torch.tensor([0.0, 0.0, -5.0], dtype=torch.float64, requires_grad=True)
    radius = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    origin = torch.tensor(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=torch.float64, requires_grad=False
    )
    direction = torch.tensor(
        [[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]], dtype=torch.float64, requires_grad=False
    )

    def hit_fn(center: torch.Tensor, radius: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sphere = Sphere(center=center, radius=radius)
        result = sphere_hit(sphere, origin, direction, t_min=0.001, t_max=math.inf)
        return result.t, result.point

    assert torch.autograd.gradcheck(hit_fn, (center, radius))


def test_plane_gradcheck_wrt_point_and_normal() -> None:
    point = torch.tensor([0.0, 0.0, -5.0], dtype=torch.float64, requires_grad=True)
    normal = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64, requires_grad=True)
    origin = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.5, 0.0]], dtype=torch.float64, requires_grad=False
    )
    direction = torch.tensor(
        [[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]], dtype=torch.float64, requires_grad=False
    )

    def hit_fn(point: torch.Tensor, normal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        plane = Plane(point=point, normal=normal)
        result = plane_hit(plane, origin, direction, t_min=0.001, t_max=math.inf)
        return result.t, result.point

    assert torch.autograd.gradcheck(hit_fn, (point, normal))


def test_sphere_no_hit_row_does_not_raise_or_nan_other_fields() -> None:
    sphere = Sphere(center=torch.tensor([0.0, 0.0, -5.0]), radius=torch.tensor(1.0))
    origin = torch.tensor([[0.0, 5.0, 0.0]])
    direction = torch.tensor([[0.0, 1.0, 0.0]])

    result = sphere_hit(sphere, origin, direction, t_min=0.001, t_max=math.inf)

    assert not bool(result.hit[0])
    assert bool(torch.isfinite(result.t).all())
    assert bool(torch.isfinite(result.point).all())
    assert bool(torch.isfinite(result.normal).all())


def test_plane_no_hit_row_does_not_raise_or_nan_other_fields() -> None:
    plane = Plane(point=torch.tensor([0.0, 0.0, -5.0]), normal=torch.tensor([0.0, 0.0, 1.0]))
    origin = torch.tensor([[0.0, 0.0, 0.0]])
    direction = torch.tensor([[1.0, 0.0, 0.0]])

    result = plane_hit(plane, origin, direction, t_min=0.001, t_max=math.inf)

    assert not bool(result.hit[0])
    assert bool(torch.isfinite(result.t).all())
    assert bool(torch.isfinite(result.point).all())
    assert bool(torch.isfinite(result.normal).all())
