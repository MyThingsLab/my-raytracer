from __future__ import annotations

import json
import pathlib
import subprocess
import sys

FIXTURE_SCENE = {
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


def test_render_command_writes_ppm_and_exits_zero(tmp_path: pathlib.Path) -> None:
    scene_path = tmp_path / "scene.json"
    scene_path.write_text(json.dumps(FIXTURE_SCENE))
    out_path = tmp_path / "render.ppm"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "myraytracer.cli",
            "render",
            "--scene",
            str(scene_path),
            "--width",
            "4",
            "--height",
            "4",
            "--spp",
            "1",
            "--max-depth",
            "1",
            "--out",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        cwd=pathlib.Path(__file__).resolve().parent.parent / "src",
    )

    assert result.returncode == 0, result.stderr
    assert out_path.exists()
    assert out_path.read_bytes().startswith(b"P6\n4 4\n255\n")
    assert "4x4" in result.stdout


def test_render_command_writes_png_for_png_extension(tmp_path: pathlib.Path) -> None:
    scene_path = tmp_path / "scene.json"
    scene_path.write_text(json.dumps(FIXTURE_SCENE))
    out_path = tmp_path / "render.png"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "myraytracer.cli",
            "render",
            "--scene",
            str(scene_path),
            "--width",
            "4",
            "--height",
            "4",
            "--spp",
            "1",
            "--max-depth",
            "1",
            "--out",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        cwd=pathlib.Path(__file__).resolve().parent.parent / "src",
    )

    assert result.returncode == 0, result.stderr
    assert out_path.exists()
    assert out_path.read_bytes().startswith(bytes([137, 80, 78, 71, 13, 10, 26, 10]))
