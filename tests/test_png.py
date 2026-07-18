from __future__ import annotations

import pathlib
import struct
import zlib

import numpy as np

from myraytracer.png import write_png


def _read_chunks(data: bytes) -> dict[bytes, bytes]:
    chunks: dict[bytes, bytes] = {}
    offset = 8
    while offset < len(data):
        (length,) = struct.unpack(">I", data[offset : offset + 4])
        chunk_type = data[offset + 4 : offset + 8]
        chunk_data = data[offset + 8 : offset + 8 + length]
        chunks[chunk_type] = chunk_data
        offset += 12 + length
    return chunks


def test_write_png_signature_and_ihdr(tmp_path: pathlib.Path) -> None:
    pixels = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [127, 127, 127]],
        ],
        dtype=np.uint8,
    )
    path = tmp_path / "out.png"

    write_png(pixels, path)

    data = path.read_bytes()
    assert data.startswith(bytes([137, 80, 78, 71, 13, 10, 26, 10]))

    chunks = _read_chunks(data)
    width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(
        ">IIBBBBB", chunks[b"IHDR"]
    )
    assert (width, height) == (2, 2)
    assert bit_depth == 8
    assert color_type == 2
    assert (compression, filter_method, interlace) == (0, 0, 0)


def test_write_png_idat_round_trips_pixel_bytes(tmp_path: pathlib.Path) -> None:
    pixels = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [127, 127, 127]],
        ],
        dtype=np.uint8,
    )
    path = tmp_path / "out.png"

    write_png(pixels, path)

    chunks = _read_chunks(path.read_bytes())
    raw = zlib.decompress(chunks[b"IDAT"])

    stride = 1 + 2 * 3
    assert len(raw) == 2 * stride
    row0 = raw[0:stride]
    row1 = raw[stride : 2 * stride]

    assert row0[0] == 0  # filter type "none"
    assert row0[1:4] == bytes([255, 0, 0])
    assert row0[4:7] == bytes([0, 255, 0])
    assert row1[0] == 0
    assert row1[1:4] == bytes([0, 0, 255])
    assert row1[4:7] == bytes([127, 127, 127])


def test_write_png_ends_with_iend_chunk(tmp_path: pathlib.Path) -> None:
    pixels = np.zeros((1, 1, 3), dtype=np.uint8)
    path = tmp_path / "out.png"

    write_png(pixels, path)

    chunks = _read_chunks(path.read_bytes())
    assert chunks[b"IEND"] == b""
