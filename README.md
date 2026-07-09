# my-raytracer

[![CI](https://github.com/MyThingsLab/my-raytracer/actions/workflows/ci.yml/badge.svg)](https://github.com/MyThingsLab/my-raytracer/actions/workflows/ci.yml) [![codecov](https://codecov.io/gh/MyThingsLab/my-raytracer/branch/main/graph/badge.svg)](https://codecov.io/gh/MyThingsLab/my-raytracer) ![Python](https://img.shields.io/badge/python-3.11%2B-blue) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A Monte Carlo path tracer: spheres + planes, Lambertian diffuse materials,
one pinhole camera, PPM output, deterministic given a fixed RNG seed.

This is **not** a `My[X]` fleet tool — no Engine call, no `CLAUDE.md`
tool-seams, no backlog label of its own. It's a plain target codebase: the
fleet's stress test for "can the harness build a genuinely hard piece of
software from an issue backlog," originally proposed as "my-renderer"
(see `MyThingsLab/mythings-core`'s `docs/tools/README.md`, MyCoder section).

## Scope (v0)

- Vectors/rays/camera, sphere + plane primitives, Lambertian BRDF.
- Cosine-weighted hemisphere Monte Carlo integration, Russian-roulette
  bounce termination, point/area lights.
- `numpy` for vector math; PPM output (stdlib, no PNG).
- Out of scope: triangle meshes, textures, refraction, GPU acceleration,
  denoising.

## Install (development)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## License

MIT — see [`LICENSE`](LICENSE).
