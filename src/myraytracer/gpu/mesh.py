from __future__ import annotations

from pathlib import Path

import torch


def load_obj(path: str | Path, *, device: torch.device) -> torch.Tensor:
    """Parse a Wavefront OBJ file into a batch of triangle vertex positions.

    Reads `v` (vertex) and `f` (face) lines only -- normals/UVs, comments,
    and blank lines are ignored -- and triangulates polygonal faces as a
    fan. Returns a (M, 3, 3) tensor: triangle index, vertex-in-triangle
    (v0/v1/v2), xyz, on `device`.
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
            # Each face token may be "v", "v/vt", or "v/vt/vn"; only the
            # leading vertex index matters here. A fan triangulation turns
            # an N-gon into N-2 triangles sharing the first vertex.
            indices = [_vertex_index(token, len(positions)) for token in args]
            for i in range(1, len(indices) - 1):
                triangles.append([indices[0], indices[i], indices[i + 1]])

    vertices = torch.tensor(positions, dtype=torch.float32, device=device)
    faces = torch.tensor(triangles, dtype=torch.long, device=device)
    return vertices[faces]


def _vertex_index(token: str, vertex_count: int) -> int:
    # OBJ vertex indices are 1-based; negative indices count back from the
    # current end of the vertex list.
    raw = int(token.split("/")[0])
    return raw - 1 if raw > 0 else vertex_count + raw
