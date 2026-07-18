from __future__ import annotations

import pathlib

import numpy as np

from myraytracer.image import write_ppm


def test_write_ppm_round_trips_header_and_pixel_bytes(tmp_path: pathlib.Path) -> None:
    pixels = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.5, 0.5, 0.5]],
        ]
    )
    path = tmp_path / "out.ppm"

    write_ppm(pixels, path)

    data = path.read_bytes()
    header = b"P6\n2 2\n255\n"
    assert data.startswith(header)

    body = data[len(header) :]
    assert body[0:3] == bytes([255, 0, 0])
    assert body[3:6] == bytes([0, 255, 0])
    assert body[6:9] == bytes([0, 0, 255])
    assert body[9:12] == bytes([127, 127, 127])
    assert len(body) == 2 * 2 * 3


def test_write_ppm_clamps_out_of_range_values(tmp_path: pathlib.Path) -> None:
    pixels = np.array([[[1.5, -0.5, 0.0]]])
    path = tmp_path / "clamped.ppm"

    write_ppm(pixels, path)

    data = path.read_bytes()
    header = b"P6\n1 1\n255\n"
    body = data[len(header) :]
    assert body == bytes([255, 0, 0])
