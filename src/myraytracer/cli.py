from __future__ import annotations

import argparse
import pathlib
import time

from myraytracer.image import write_ppm
from myraytracer.render import render
from myraytracer.sceneio import load_scene


def _render_command(args: argparse.Namespace) -> None:
    scene, camera = load_scene(args.scene)

    start = time.perf_counter()
    pixels = render(
        scene,
        camera,
        width=args.width,
        height=args.height,
        spp=args.spp,
        max_depth=args.max_depth,
        seed=args.seed,
    )
    elapsed = time.perf_counter() - start

    write_ppm(pixels, args.out)
    print(
        f"rendered {args.width}x{args.height} at {args.spp} spp in {elapsed:.2f}s -> {args.out}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="myraytracer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    render_parser = subparsers.add_parser("render", help="render a scene to a PPM file")
    render_parser.add_argument("--scene", type=pathlib.Path, required=True)
    render_parser.add_argument("--width", type=int, default=64)
    render_parser.add_argument("--height", type=int, default=64)
    render_parser.add_argument("--spp", type=int, default=16)
    render_parser.add_argument("--max-depth", type=int, default=4)
    render_parser.add_argument("--seed", type=int, default=0)
    render_parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("render.ppm"))
    render_parser.set_defaults(func=_render_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
