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
- Out of scope: triangle meshes, textures, refraction, denoising.

## Scope (v1 — `myraytracer.gpu`)

A second, parallel implementation of the same physics in PyTorch
(`src/myraytracer/gpu/`), batched over rays/pixels and differentiable —
GPU-accelerated when CUDA is available, CPU fallback otherwise (CI has no
GPU, so it only ever exercises the CPU path). The v0 `numpy` module is
untouched; nothing in v0 imports from `gpu/` or vice versa.

- Batched, autograd-safe vector/ray/geometry ops; a `Scene` gradient can flow
  from a rendered pixel back to material albedo, light intensity, and
  geometry parameters for the *smooth* (non-occluded) terms.
- Occlusion/visibility is a hard boolean mask in v1 — gradients do **not**
  flow through shadow boundaries (a detached mask, not the harder
  reparameterized/edge-sampling visibility gradient from the differentiable-
  rendering literature). Documented as a known limitation, not a bug.
- v1 direct lighting only (no recursive Monte Carlo path tracing yet) — a
  full differentiable multi-bounce integrator is future work.
- Proven with an actual inverse-rendering test: gradient descent recovers a
  known albedo from a target rendered image.
- Out of scope (for now): differentiable visibility/soft shadows, multi-bounce
  GI, triangle meshes.

## Install (development)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"       # v0 (numpy) only
pip install -e ".[dev,gpu]"   # + v1 (myraytracer.gpu, torch)
pytest
```

## License

MIT — see [`LICENSE`](LICENSE).
