from __future__ import annotations

import json
import pathlib

import pytest

from myraytracer.camera import Camera
from myraytracer.geometry import Plane, Sphere
from myraytracer.light import PointLight
from myraytracer.sceneio import load_scene
from myraytracer.vec import Vec3


def _basic_scene_dict() -> dict:
    return {
        "camera": {
            "origin": [0, 0, 0],
            "look_at": [0, 0, -1],
            "up": [0, 1, 0],
            "vfov_degrees": 90,
            "aspect_ratio": 1,
        },
        "objects": [
            {
                "type": "sphere",
                "center": [0, 0, -5],
                "radius": 1,
                "material": {"albedo": [0.8, 0.5, 0.2]},
            },
            {
                "type": "plane",
                "point": [0, -1, 0],
                "normal": [0, 1, 0],
                "material": {"albedo": [0.2, 0.2, 0.2]},
            },
        ],
        "lights": [{"position": [0, 5, -2], "intensity": [10, 10, 10]}],
    }


def _write_scene(tmp_path: pathlib.Path, data: dict) -> pathlib.Path:
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(data))
    return path


def test_load_scene_happy_path_matches_expected_dataclasses(tmp_path: pathlib.Path) -> None:
    path = _write_scene(tmp_path, _basic_scene_dict())

    scene, camera = load_scene(path)

    assert camera == Camera(
        origin=Vec3(0, 0, 0),
        look_at=Vec3(0, 0, -1),
        up=Vec3(0, 1, 0),
        vfov_degrees=90,
        aspect_ratio=1,
    )
    assert len(scene.objects) == 2
    sphere, plane = scene.objects
    assert isinstance(sphere, Sphere)
    assert sphere.center == Vec3(0, 0, -5)
    assert sphere.radius == 1
    assert sphere.material.albedo == Vec3(0.8, 0.5, 0.2)
    assert isinstance(plane, Plane)
    assert plane.point == Vec3(0, -1, 0)
    assert plane.normal == Vec3(0, 1, 0)
    assert len(scene.lights) == 1
    assert scene.lights[0] == PointLight(position=Vec3(0, 5, -2), intensity=Vec3(10, 10, 10))


def test_load_scene_rejects_non_positive_sphere_radius(tmp_path: pathlib.Path) -> None:
    data = _basic_scene_dict()
    data["objects"][0]["radius"] = 0
    path = _write_scene(tmp_path, data)

    with pytest.raises(ValueError, match="radius"):
        load_scene(path)


def test_load_scene_rejects_near_zero_plane_normal(tmp_path: pathlib.Path) -> None:
    data = _basic_scene_dict()
    data["objects"][1]["normal"] = [0, 0, 0]
    path = _write_scene(tmp_path, data)

    with pytest.raises(ValueError, match="normal"):
        load_scene(path)


def test_load_scene_rejects_negative_light_intensity(tmp_path: pathlib.Path) -> None:
    data = _basic_scene_dict()
    data["lights"][0]["intensity"] = [-1, 10, 10]
    path = _write_scene(tmp_path, data)

    with pytest.raises(ValueError, match="intensity"):
        load_scene(path)


def test_load_scene_rejects_missing_required_camera_field(tmp_path: pathlib.Path) -> None:
    data = _basic_scene_dict()
    del data["camera"]["vfov_degrees"]
    path = _write_scene(tmp_path, data)

    with pytest.raises(ValueError, match="vfov_degrees"):
        load_scene(path)
