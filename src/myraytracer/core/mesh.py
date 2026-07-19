from __future__ import annotations

from pathlib import Path

import numpy as np


def load_obj(path: str | Path) -> np.ndarray:
    """Parse a Wavefront OBJ file into a batch of triangle vertex positions.

    Reads `v` (vertex) and `f` (face) lines only -- normals/UVs, comments, and
    blank lines are ignored -- and triangulates polygonal faces as a fan.
    Returns a (M, 3, 3) numpy array: triangle index, vertex-in-triangle
    (v0/v1/v2), xyz. Backend-agnostic (host array); a `Mesh` converts it to the
    render backend at hit time.
    """
    positions: list[list[float]] = []
    triangles: list[list[int]] = []

    for line in Path(path).read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        keyword, *args = stripped.split()

        if keyword == "v":
            positions.append([float(value) for value in args[:3]])
        elif keyword == "f":
            # Only the leading vertex index of each "v/vt/vn" token matters; a
            # fan triangulation turns an N-gon into N-2 triangles.
            indices = [_vertex_index(token, len(positions)) for token in args]
            for i in range(1, len(indices) - 1):
                triangles.append([indices[0], indices[i], indices[i + 1]])

    vertices = np.asarray(positions, dtype=np.float64)
    faces = np.asarray(triangles, dtype=np.intp)
    return vertices[faces]


def _vertex_index(token: str, vertex_count: int) -> int:
    # OBJ vertex indices are 1-based; negative indices count back from the end.
    raw = int(token.split("/")[0])
    return raw - 1 if raw > 0 else vertex_count + raw
