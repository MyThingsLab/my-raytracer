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
- `render()` (camera/pixel path) is direct lighting only; `pathtracer.trace()`
  adds full multi-bounce global illumination as a batched wavefront path
  tracer (Laine, Karras, Aila, *"Megakernels Considered Harmful"*, HPG 2013)
  — bounces are streamed as tensor passes over the whole live ray set, with a
  per-ray `alive` mask and running throughput, rather than per-ray recursion.
- Proven with an actual inverse-rendering test: gradient descent recovers a
  known albedo from a target rendered image.
- Out of scope (for now): differentiable visibility/soft shadows, triangle
  meshes.

## Example: Cornell box

`examples/cornell_box.json` is the canonical Cornell box scene: five
axis-aligned `Quad` walls (red left, green right, white floor/ceiling/back —
the camera-side wall is omitted so the camera can see in), a small emissive
`Quad` recessed into the ceiling as an area light, and a white `Sphere` inside
as a shadow-casting occluder. Concrete bounds and the exact wall/light
placement are documented in the scene file's own `"_comment"` field (JSON has
no native comments, and `load_scene` tolerates unknown top-level keys).

```bash
myraytracer render --scene examples/cornell_box.json \
  --width 128 --height 128 --spp 64 --max-depth 4 --seed 0 \
  --out cornell_box.ppm
```

This finishes in a single session and produces a `.ppm` image with the
Cornell box's recognizable shape and color-bled walls. It is not meant to be
photorealistic or fully converged at this sample count.

## Install (development)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"       # v0 (numpy) only
pip install -e ".[dev,gpu]"   # + v1 (myraytracer.gpu, torch)
pytest
```

## License

MIT — see [`LICENSE`](LICENSE).
