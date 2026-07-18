from __future__ import annotations

import torch

from myraytracer.gpu.ray import Ray


def test_at_single_ray() -> None:
    ray = Ray(origin=torch.tensor([0.0, 0.0, 0.0]), direction=torch.tensor([0.0, 0.0, -1.0]))

    result = ray.at(torch.tensor(2.0))

    assert torch.allclose(result, torch.tensor([0.0, 0.0, -2.0]))


def test_at_batch() -> None:
    ray = Ray(
        origin=torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        direction=torch.tensor([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]),
    )

    result = ray.at(torch.tensor([2.0, 3.0]))

    assert result.shape == (2, 3)
    assert torch.allclose(result[0], torch.tensor([0.0, 0.0, -2.0]))
    assert torch.allclose(result[1], torch.tensor([4.0, 1.0, 1.0]))


def test_at_broadcasts_origin_against_batched_direction() -> None:
    ray = Ray(
        origin=torch.tensor([0.0, 0.0, 0.0]),
        direction=torch.tensor([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]),
    )

    result = ray.at(torch.tensor([1.0, 5.0]))

    assert result.shape == (2, 3)
    assert torch.allclose(result[0], torch.tensor([0.0, 0.0, -1.0]))
    assert torch.allclose(result[1], torch.tensor([5.0, 0.0, 0.0]))
