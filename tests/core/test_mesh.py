from __future__ import annotations

import math

import numpy as np

from myraytracer.core.backend import NUMPY, Backend
from myraytracer.core.camera import Camera
from myraytracer.core.geometry import Mesh, Quad, mesh_hit
from myraytracer.core.integrator import render as render_image
from myraytracer.core.material import Material
from myraytracer.core.mesh import load_obj
from myraytracer.core.scene import PointLight, Scene

_MAT = Material(albedo=(0.7, 0.7, 0.7))
_T_MIN = 1e-4

# Two triangles tiling the axis-aligned quad z=-3, x/y in [-1, 1].
_QUAD_MESH = np.array(
    [
        [[-1.0, -1.0, -3.0], [1.0, -1.0, -3.0], [1.0, 1.0, -3.0]],
        [[-1.0, -1.0, -3.0], [1.0, 1.0, -3.0], [-1.0, 1.0, -3.0]],
    ]
)


def _rays(backend: Backend, width: int = 8, height: int = 8):
    camera = Camera(
        origin=(0.0, 0.0, 0.0),
        look_at=(0.0, 0.0, -1.0),
        up=(0.0, 1.0, 0.0),
        vfov_degrees=70.0,
        aspect_ratio=1.0,
    )
    rays = camera.grid_rays(width, height, backend)
    return rays.origin, rays.direction


def test_single_triangle_hit(backend: Backend) -> None:
    tri = np.array([[[-1.0, -1.0, -2.0], [1.0, -1.0, -2.0], [0.0, 1.0, -2.0]]])
    mesh = Mesh(vertices=tri, material=_MAT)
    ro = backend.asarray([[0.0, 0.0, 0.0], [5.0, 5.0, 0.0]])
    rd = backend.asarray([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]])
    hit = mesh_hit(mesh, ro, rd, _T_MIN, math.inf, backend)

    assert np.asarray(hit.hit).tolist() == [True, False]
    assert np.allclose(np.asarray(hit.t)[0], 2.0, atol=1e-5)
    assert np.allclose(np.asarray(hit.normal)[0], [0.0, 0.0, 1.0], atol=1e-5)


def test_mesh_matches_gpu_reference() -> None:
    # Validate the backend-agnostic port against the trusted torch gpu.mesh_hit
    # on the same rays and triangles.
    import torch

    from myraytracer.core.backend import torch_backend
    from myraytracer.gpu.geometry import Mesh as GpuMesh
    from myraytracer.gpu.geometry import mesh_hit as gpu_mesh_hit

    backend = torch_backend(device=torch.device("cpu"))
    ro, rd = _rays(backend)
    core = mesh_hit(Mesh(vertices=_QUAD_MESH, material=_MAT), ro, rd, _T_MIN, math.inf, backend)

    gpu_verts = torch.tensor(_QUAD_MESH, dtype=torch.float32)
    gpu = gpu_mesh_hit(GpuMesh(vertices=gpu_verts), ro, rd, t_min=_T_MIN, t_max=math.inf)

    assert np.array_equal(np.asarray(core.hit), np.asarray(gpu.hit))
    mask = np.asarray(core.hit)
    assert np.allclose(np.asarray(core.t)[mask], np.asarray(gpu.t)[mask], atol=1e-4)
    assert np.allclose(np.asarray(core.point)[mask], np.asarray(gpu.point)[mask], atol=1e-4)


def test_mesh_agrees_across_backends(backend: Backend) -> None:
    mesh = Mesh(vertices=_QUAD_MESH, material=_MAT)
    ro_np, rd_np = _rays(NUMPY)
    ref = mesh_hit(mesh, ro_np, rd_np, _T_MIN, math.inf, NUMPY)
    ro, rd = _rays(backend)
    other = mesh_hit(mesh, ro, rd, _T_MIN, math.inf, backend)

    assert np.array_equal(np.asarray(ref.hit), np.asarray(other.hit))
    mask = np.asarray(ref.hit)
    assert np.allclose(np.asarray(ref.t)[mask], np.asarray(other.t)[mask], atol=1e-4)


def test_mesh_quad_equivalent_render(backend: Backend) -> None:
    # A two-triangle mesh and the equivalent Quad, lit identically, must render
    # to the same image (same geometry, deterministic seed).
    light = PointLight(position=(0.0, 0.0, 1.0), intensity=(5.0, 5.0, 5.0))
    camera = Camera(
        origin=(0.0, 0.0, 0.0),
        look_at=(0.0, 0.0, -1.0),
        up=(0.0, 1.0, 0.0),
        vfov_degrees=70.0,
        aspect_ratio=1.0,
    )
    kwargs = dict(width=12, height=12, spp=2, max_depth=1, seed=0, backend=backend)

    mesh_scene = Scene(objects=[Mesh(vertices=_QUAD_MESH, material=_MAT)], lights=[light])
    quad_scene = Scene(
        objects=[
            Quad(
                corner=(-1.0, -1.0, -3.0),
                edge1=(2.0, 0.0, 0.0),
                edge2=(0.0, 2.0, 0.0),
                material=_MAT,
            )
        ],
        lights=[light],
    )
    mesh_img = np.asarray(render_image(mesh_scene, camera, **kwargs))
    quad_img = np.asarray(render_image(quad_scene, camera, **kwargs))
    assert np.allclose(mesh_img, quad_img, atol=1e-4)


def test_load_obj_parses_triangles(tmp_path) -> None:
    obj = tmp_path / "quad.obj"
    obj.write_text(
        "# a unit quad as two triangles\n"
        "v -1 -1 0\nv 1 -1 0\nv 1 1 0\nv -1 1 0\n"
        "f 1 2 3\nf 1 3 4\n"
    )
    vertices = load_obj(obj)
    assert vertices.shape == (2, 3, 3)
    assert np.allclose(vertices[0, 0], [-1.0, -1.0, 0.0])
    assert np.allclose(vertices[1, 2], [-1.0, 1.0, 0.0])


def test_load_obj_fan_triangulates_quad_face(tmp_path) -> None:
    obj = tmp_path / "ngon.obj"
    obj.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n")
    vertices = load_obj(obj)
    assert vertices.shape == (2, 3, 3)  # a 4-gon fans into 2 triangles
