from __future__ import annotations

import math

import pytest
import torch

from myraytracer.gpu.geometry import Plane, Sphere
from myraytracer.gpu.scene import PointLight, Scene


def test_point_light_rejects_negative_intensity_component() -> None:
    with pytest.raises(ValueError):
        PointLight(
            position=torch.tensor([0.0, 0.0, 0.0]),
            intensity=torch.tensor([-1.0, 0.0, 0.0]),
        )


def test_nearest_hit_returns_closer_plane_in_front_of_sphere() -> None:
    sphere = Sphere(center=torch.tensor([0.0, 0.0, -10.0]), radius=torch.tensor(1.0))
    plane = Plane(point=torch.tensor([0.0, 0.0, -5.0]), normal=torch.tensor([0.0, 0.0, 1.0]))
    scene = Scene(objects=[sphere, plane], lights=[])
    origin = torch.tensor([[0.0, 0.0, 0.0]])
    direction = torch.tensor([[0.0, 0.0, -1.0]])

    result = scene.nearest_hit(origin, direction, t_min=0.001, t_max=math.inf)

    assert bool(result.hit[0])
    assert result.t[0].item() == pytest.approx(5.0)


def test_nearest_hit_on_empty_scene_is_false() -> None:
    scene = Scene(objects=[], lights=[])
    origin = torch.tensor([[0.0, 0.0, 0.0]])
    direction = torch.tensor([[0.0, 0.0, -1.0]])

    result = scene.nearest_hit(origin, direction, t_min=0.001, t_max=math.inf)

    assert not bool(result.hit[0])


def test_nearest_hit_outside_t_range_is_false() -> None:
    sphere = Sphere(center=torch.tensor([0.0, 0.0, -5.0]), radius=torch.tensor(1.0))
    scene = Scene(objects=[sphere], lights=[])
    origin = torch.tensor([[0.0, 0.0, 0.0]])
    direction = torch.tensor([[0.0, 0.0, -1.0]])

    result = scene.nearest_hit(origin, direction, t_min=0.001, t_max=3.0)

    assert not bool(result.hit[0])
