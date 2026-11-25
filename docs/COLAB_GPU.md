# Colab GPU validation (NVIDIA T4)

> **Documentation map:** [README](../README.md) · [Getting Started](GETTING_STARTED.md) · [Tutorials](TUTORIALS.md) · [Crease Format](CREASE_FORMAT.md) · [Simulation](SIMULATION.md) · [Evaluation](EVALUATION.md) · [Fitness](FITNESS.md) · [API Reference](API_REFERENCE.md) · [Architecture](ARCHITECTURE.md) · [Troubleshooting](TROUBLESHOOTING.md) · [Glossary](GLOSSARY.md) · [Integrations](INTEGRATIONS.md) · [Optimization (ES)](OPTIMIZATION_ES.md) · [Search space](SEARCH_SPACE.md) · [Populations](POPULATIONS.md) · [RL](RL.md) · [Contributing](CONTRIBUTING.md)

This document describes how to validate the MJX-Warp backend and benchmark both backends on Google Colab's T4 GPU. **No GPU results are claimed in this repo** — run `examples/colab_gpu_validation.py` in Colab and record the output.

## Setup in Colab

```python
!pip install 'origami-engine-mjx[all]'
# or from a checkout:
!pip install -e '.[all]'
```

Colab's T4 has CUDA; `warp-lang` will initialise `cuda:0` automatically.

## Validation checklist

Run `examples/colab_gpu_validation.py`, which prints (and checks):

1. `available_backends()`, `WarpBackend().is_available()` must be `True`, and `device_info()`.
2. `jax.devices()` / `jax.default_backend()` — confirms JAX sees the GPU.
3. MJX-Warp model conversion (`mjw.put_model`) plus a short `mjw.step` run; `final_centroid` must be finite.
4. The same run on MJX-JAX/GPU for a numerical comparison.
5. Population throughput at sizes 32 / 128 / 256 with `evaluate_population` (one warm-up run, then timed), including a `batch_size`-ignored note: both backends currently loop per individual.
6. GPU memory — `jax.devices()[0].memory_stats()` and `nvidia-smi --query-gpu=memory.used,memory.total`.

## What to record

- Compile/warm-up time per backend (the script runs one warm-up population pass before timing).
- Simulation wall time for the 50 ms single run (Warp and JAX).
- Population evaluations per second at each population size.
- GPU memory high-water mark (JAX memory stats + `nvidia-smi`).
- Numerical comparison of `final_centroid` vs classic MuJoCo on the same model (same initial conditions, duration).

Paste results into a Colab notebook cell or a `BENCHMARKS.md` and link from the PR.

## Benchmark steps (Colab T4)

The validation script `examples/colab_gpu_validation.py` performs these steps in order:

1. **Environment** — print `available_backends()`, `WarpBackend().is_available()`, `device_info()`, and `jax.devices()`.
2. **Warp single sim** — `OrigamiEngine(backend="warp").run(creases, duration=0.05, fold_duration=0.02, vibrate=True)`; assert finite `final_centroid`.
3. **JAX on GPU single sim** — `OrigamiEngine(backend="jax").run(...)` with the same parameters, for numerical comparison.
4. **Population throughput** — `evaluate_population` at sizes 32 / 128 / 256 (one warm-up pass, then timed) for each available backend.
5. **GPU memory** — `jax.devices()[0].memory_stats()` and `nvidia-smi`.
6. **Report** — print all timings, device info, and any errors.

To run manually in a Colab notebook:

```python
!pip install 'origami-engine-mjx[all]'
!git clone https://github.com/ihsan31415/origami-engine-mjx.git /content/origami-engine-mjx
%cd /content/origami-engine-mjx
!python examples/colab_gpu_validation.py
```

## Expected output (no numbers claimed)

The script prints a structured report. On a correctly configured T4:

- `available backends: ['jax', 'warp']` and `warp is_available: True`
- `jax devices: [CudaDevice(id=0)]` (or similar)
- Finite `final_centroid` for both the Warp and JAX runs
- Per-population-size wall times and throughput (record these in your PR)

## Troubleshooting GPU issues

- **`warp is_available: False`** — Warp or `mujoco-warp` not installed, or no CUDA device. Check `nvidia-smi` in the Colab runtime (`Runtime → Change runtime type → T4 GPU`).
- **JAX uses CPU on GPU machine** — set `JAX_PLATFORMS=cuda` (or ensure `jax[cuda12]` is installed). Check `jax.devices()`.
- **Out of memory at large population sizes** — reduce `n` in the validation script (the loop is sequential, so this means per-individual model size; large Miura meshes are the main cost).
- **Numerical mismatch vs classic MuJoCo** — MJX uses float32; small differences are expected (see [BACKENDS.md](BACKENDS.md)).

## See also

- [BACKENDS.md](BACKENDS.md) — backend semantics and feature table
- [examples/README.md](../examples/README.md) — runnable scripts
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — general runtime issues
## Measured: batched Warp evolution (T4, 2026-11-25)

Runtime: Colab T4 (15 GiB), `mujoco-warp==4.0.1`, `warp-lang==1.19.0`, JAX 0.11.1 + CUDA 13 plugin; package branch `perf/colab-warp-all-creases`. Pattern: 2×2 Miura, 6 creases, 12 free variables (4 angles + 8 vertex coordinates). `duration=0.05 s`, `fold=0.02 s`, `vibrate=False`.

| Worlds | Wall time | Candidates/s | Graph launch | Host readback |
|---:|---:|---:|---:|---:|
| 8  | 0.24 s | 33  | 0.0039 s | 0.031 s |
| 32 | 0.52 s | 62  | 0.0039 s | 0.039 s |
| 128| 1.77 s | 72  | 0.0041 s | 0.074 s |

Sequential single-candidate scoring of 8 genomes took 0.49 s (16 cand/s). Batched and single scores agreed to 5e-9 (max abs difference). Per-call model rebuild cost ~0.6 s (`put_model`); caching the batched model/graph removes it. The notebook's end-to-end run: 3 generations × 32 candidates = 96 simulations in ~32 s, 3/3 successful generations, per-generation JSON checkpoints written.

Setup pitfalls: install only one JAX CUDA plugin (Colab ships jax-cuda13; a `[jax]` reinstall adds jax-cuda12 and triggers `PJRT_Api already exists for device type cuda` — uninstall `jax-cuda12-*`). `mujoco-warp 4.0` needs `make_data(nworld=...)` and only fields with an `(*, …)` array spec may be passed as `batch_sizes` to `put_model`. Use `colab ssh` + `nohup` for runs longer than ~30 s; `colab exec` drops the connection during long kernel compiles.
