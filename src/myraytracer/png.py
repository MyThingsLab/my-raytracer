from __future__ import annotations

import pathlib
import struct
import zlib

import numpy as np

_SIGNATURE = bytes([137, 80, 78, 71, 13, 10, 26, 10])
_COLOR_TYPE_TRUECOLOR = 2
_BIT_DEPTH = 8


def _chunk(chunk_type: bytes, data: bytes) -> bytes:
    length = struct.pack(">I", len(data))
    crc = struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    return length + chunk_type + data + crc


def write_png(pixels_uint8: np.ndarray, path: pathlib.Path) -> None:
    """Write an `(H, W, 3)` uint8 array as a truecolor PNG.

    Hand-rolled from stdlib `zlib` and `struct` (no imaging dependency), in
    the same from-scratch spirit as `image.write_ppm`: an 8-byte signature
    followed by IHDR, a single zlib-compressed IDAT (each scanline prefixed
    with a "none" filter byte), and IEND.
    """
    height, width, _ = pixels_uint8.shape

    ihdr = struct.pack(
        ">IIBBBBB",
        width,
        height,
        _BIT_DEPTH,
        _COLOR_TYPE_TRUECOLOR,
        0,
        0,
        0,
    )

    filter_bytes = np.zeros((height, 1), dtype=np.uint8)
    scanlines = np.concatenate([filter_bytes, pixels_uint8.reshape(height, width * 3)], axis=1)
    raw = scanlines.astype(np.uint8).tobytes()
    idat = zlib.compress(raw)

    body = (
        _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", idat)
        + _chunk(b"IEND", b"")
    )
    path.write_bytes(_SIGNATURE + body)
