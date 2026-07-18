from __future__ import annotations

import json
import pathlib
from typing import Any

from myraytracer.camera import Camera
from myraytracer.geometry import Plane, Sphere
from myraytracer.light import PointLight
from myraytracer.material import Material
from myraytracer.scene import Scene
from myraytracer.vec import Vec3

# On-disk scene format, minimal JSON covering one sphere + one plane + one
# point light:
#
# {
#   "camera": {
#     "origin": [x, y, z],
#     "look_at": [x, y, z],
#     "up": [x, y, z],
#     "vfov_degrees": 90,
#     "aspect_ratio": 1
#   },
#   "objects": [
#     {"type": "sphere", "center": [x, y, z], "radius": 1,
#      "material": {"albedo": [r, g, b], "emission": [r, g, b]}},
#     {"type": "plane", "point": [x, y, z], "normal": [x, y, z],
#      "material": {"albedo": [r, g, b]}}
#   ],
#   "lights": [
#     {"position": [x, y, z], "intensity": [r, g, b]}
#   ]
# }
#
# `material.emission` is optional and defaults to (0, 0, 0), matching
# Material's own default.


def _vec3(data: Any, field_name: str) -> Vec3:
    if not isinstance(data, list) or len(data) != 3:
        raise ValueError(f"{field_name} must be a 3-element array")
    try:
        return Vec3(float(data[0]), float(data[1]), float(data[2]))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain three numbers") from exc


def _require(data: dict[str, Any], key: str, context: str) -> Any:
    if key not in data:
        raise ValueError(f"{context} is missing required field '{key}'")
    return data[key]


def _parse_camera(data: dict[str, Any]) -> Camera:
    try:
        return Camera(
            origin=_vec3(_require(data, "origin", "camera"), "camera.origin"),
            look_at=_vec3(_require(data, "look_at", "camera"), "camera.look_at"),
            up=_vec3(_require(data, "up", "camera"), "camera.up"),
            vfov_degrees=float(_require(data, "vfov_degrees", "camera")),
            aspect_ratio=float(_require(data, "aspect_ratio", "camera")),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid camera: {exc}") from exc


def _parse_material(data: dict[str, Any] | None, context: str) -> Material:
    if data is None:
        raise ValueError(f"{context} is missing required field 'material'")
    albedo = _vec3(_require(data, "albedo", f"{context}.material"), f"{context}.material.albedo")
    if "emission" in data:
        emission = _vec3(data["emission"], f"{context}.material.emission")
        return Material(albedo=albedo, emission=emission)
    return Material(albedo=albedo)


def _parse_object(data: dict[str, Any]) -> Sphere | Plane:
    object_type = data.get("type")
    if object_type == "sphere":
        try:
            return Sphere(
                center=_vec3(_require(data, "center", "sphere"), "sphere.center"),
                radius=float(_require(data, "radius", "sphere")),
                material=_parse_material(data.get("material"), "sphere"),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid sphere: {exc}") from exc
    if object_type == "plane":
        try:
            return Plane(
                point=_vec3(_require(data, "point", "plane"), "plane.point"),
                normal=_vec3(_require(data, "normal", "plane"), "plane.normal"),
                material=_parse_material(data.get("material"), "plane"),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid plane: {exc}") from exc
    raise ValueError(f"unknown object type: {object_type!r}")


def _parse_light(data: dict[str, Any]) -> PointLight:
    try:
        return PointLight(
            position=_vec3(_require(data, "position", "light"), "light.position"),
            intensity=_vec3(_require(data, "intensity", "light"), "light.intensity"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid light: {exc}") from exc


def load_scene(path: pathlib.Path) -> tuple[Scene, Camera]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("scene file must contain a JSON object")

    camera = _parse_camera(_require(raw, "camera", "scene"))
    objects = [_parse_object(entry) for entry in _require(raw, "objects", "scene")]
    lights = [_parse_light(entry) for entry in _require(raw, "lights", "scene")]

    return Scene(objects=objects, lights=lights), camera
