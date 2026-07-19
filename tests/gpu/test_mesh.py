from __future__ import annotations

import math
import pathlib

import pytest
import torch

from myraytracer.gpu.geometry import Mesh, Sphere, mesh_hit, triangle_hit
from myraytracer.gpu.mesh import load_obj
from myraytracer.gpu.scene import Scene


def _triangle() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Counter-clockwise as seen from +z, so cross(edge1, edge2) points
    # toward +z -- i.e. toward a camera at the origin looking down -z.
    v0 = torch.tensor([[-1.0, -1.0, -5.0]])
    v1 = torch.tensor([[1.0, -1.0, -5.0]])
    v2 = torch.tensor([[0.0, 1.0, -5.0]])
    return v0, v1, v2


def test_triangle_hit_returns_expected_t() -> None:
    v0, v1, v2 = _triangle()
    origin = torch.tensor([[0.0, 0.0, 0.0]])
    direction = torch.tensor([[0.0, 0.0, -1.0]])

    result = triangle_hit(v0, v1, v2, origin, direction, t_min=0.001, t_max=math.inf)

    assert bool(result.hit[0])
    assert result.t[0].item() == pytest.approx(5.0)
    assert torch.allclose(result.point[0], torch.tensor([0.0, 0.0, -5.0]))
    assert torch.allclose(result.normal[0], torch.tensor([0.0, 0.0, 1.0]), atol=1e-6)


def test_triangle_miss_outside_bounds_returns_false() -> None:
    v0, v1, v2 = _triangle()
    origin = torch.tensor([[5.0, 5.0, 0.0]])
    direction = torch.tensor([[0.0, 0.0, -1.0]])

    result = triangle_hit(v0, v1, v2, origin, direction, t_min=0.001, t_max=math.inf)

    assert not bool(result.hit[0])


def test_triangle_backface_hit_faces_forward_like_sphere_hit() -> None:
    # Hitting the triangle from behind its geometric normal must still
    # return a normal that opposes the ray, matching sphere_hit's
    # face-forward convention for a ray from inside the sphere.
    v0, v1, v2 = _triangle()
    origin = torch.tensor([[0.0, 0.0, -10.0]])
    direction = torch.tensor([[0.0, 0.0, 1.0]])

    result = triangle_hit(v0, v1, v2, origin, direction, t_min=0.001, t_max=math.inf)

    assert bool(result.hit[0])
    assert result.t[0].item() == pytest.approx(5.0)
    assert torch.allclose(result.normal[0], torch.tensor([0.0, 0.0, -1.0]), atol=1e-6)
    assert (result.normal[0] * direction[0]).sum().item() < 0.0


def _write_quad_obj(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "quad.obj"
    path.write_text(
        "# a unit quad, triangulated as a fan\n"
        "\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 1 1 0\n"
        "v 0 1 0\n"
        "\n"
        "f 1 2 3 4\n"
    )
    return path


def test_load_obj_triangulates_quad_into_two_triangles(tmp_path: pathlib.Path) -> None:
    path = _write_quad_obj(tmp_path)

    triangles = load_obj(path, device=torch.device("cpu"))

    assert triangles.shape == (2, 3, 3)
    expected_first = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    expected_second = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    assert torch.allclose(triangles[0], expected_first)
    assert torch.allclose(triangles[1], expected_second)


def test_mesh_in_scene_occludes_sphere_behind_it() -> None:
    sphere = Sphere(center=torch.tensor([0.0, 0.0, -10.0]), radius=torch.tensor(1.0))
    v0 = torch.tensor([-10.0, -10.0, -5.0])
    v1 = torch.tensor([10.0, -10.0, -5.0])
    v2 = torch.tensor([0.0, 10.0, -5.0])
    mesh = Mesh(vertices=torch.stack([v0, v1, v2]).unsqueeze(0))
    scene = Scene(objects=[sphere, mesh], lights=[])
    origin = torch.tensor([[0.0, 0.0, 0.0]])
    direction = torch.tensor([[0.0, 0.0, -1.0]])

    result = scene.nearest_hit(origin, direction, t_min=0.001, t_max=math.inf)

    assert bool(result.hit[0])
    assert result.t[0].item() == pytest.approx(5.0)


def test_mesh_hit_matches_triangle_hit_directly() -> None:
    v0, v1, v2 = _triangle()
    mesh = Mesh(vertices=torch.stack([v0[0], v1[0], v2[0]]).unsqueeze(0))
    origin = torch.tensor([[0.0, 0.0, 0.0]])
    direction = torch.tensor([[0.0, 0.0, -1.0]])

    mesh_result = mesh_hit(mesh, origin, direction, t_min=0.001, t_max=math.inf)
    triangle_result = triangle_hit(v0, v1, v2, origin, direction, t_min=0.001, t_max=math.inf)

    assert bool(mesh_result.hit[0]) == bool(triangle_result.hit[0])
    assert mesh_result.t[0].item() == pytest.approx(triangle_result.t[0].item())


def test_triangle_gradcheck_wrt_vertices() -> None:
    v0 = torch.tensor([[-1.0, -1.0, -5.0]], dtype=torch.float64, requires_grad=True)
    v1 = torch.tensor([[1.0, -1.0, -5.0]], dtype=torch.float64, requires_grad=True)
    v2 = torch.tensor([[0.0, 1.0, -5.0]], dtype=torch.float64, requires_grad=True)
    origin = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64, requires_grad=False)
    direction = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float64, requires_grad=False)

    def hit_fn(
        v0: torch.Tensor, v1: torch.Tensor, v2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        result = triangle_hit(v0, v1, v2, origin, direction, t_min=0.001, t_max=math.inf)
        return result.t, result.point

    assert torch.autograd.gradcheck(hit_fn, (v0, v1, v2))
