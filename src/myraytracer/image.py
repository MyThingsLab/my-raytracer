from __future__ import annotations

import pathlib

import numpy as np


def write_ppm(pixels: np.ndarray, path: pathlib.Path) -> None:
    height, width, _ = pixels.shape
    clamped = np.clip(pixels, 0.0, 1.0)
    scaled = (clamped * 255).astype(np.uint8)

    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    path.write_bytes(header + scaled.tobytes())
