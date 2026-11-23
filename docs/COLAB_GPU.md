# Colab GPU validation (NVIDIA T4)

> **Documentation map:** [README](../README.md) · [Getting Started](GETTING_STARTED.md) · [Tutorials](TUTORIALS.md) · [Crease Format](CREASE_FORMAT.md) · [Simulation](SIMULATION.md) · [Evaluation](EVALUATION.md) · [API Reference](API_REFERENCE.md) · [Architecture](ARCHITECTURE.md) · [Troubleshooting](TROUBLESHOOTING.md) · [Glossary](GLOSSARY.md) · [Contributing](CONTRIBUTING.md)

This document describes how to validate the MJX-Warp backend and benchmark both backends on Google Colab's T4 GPU. **No GPU results are claimed in this repo** — run `examples/colab_gpu_validation.py` in Colab and record the output.

## Setup in Colab

```python
!pip install 'origami-engine-mjx[all]'
# or from a checkout:
!pip install -e '.[all]'
```

Colab's T4 has CUDA; `warp-lang` will initialise `cuda:0` automatically.

## Validation checklist

Run `examples/colab_gpu_validation.py`, which checks:

1. Warp backend availability (`WarpBackend().is_available()` must be `True`).
2. MJX-Warp model conversion (`mjw.make_model`) for the reference Miura pattern.
3. State init + individual `mjw.step` calls (finite `qpos`).
4. Actuation ramp + table vibration drive.
5. Contact/constraint handling (paper geoms collide with table; equality constraints hold).
6. Batched population evaluation at sizes 32 / 128 / 256 / 512 / 1024.
7. MJX-JAX on GPU (`jax.devices()` shows `gpu`) for comparison.
8. GPU memory consumption (`nvidia-smi` or `jax.devices()[0].memory_stats()`).
9. Steady-state throughput (steps/s, populations/s), excluding JIT/compile warm-up.

## What to record

- Compile/warm-up time per backend.
- Simulation steps per second at steady state.
- Population evaluations per second at each batch size.
- GPU memory high-water mark.
- Numerical comparison vs classic MuJoCo on the same model (same initial conditions, duration).

Paste results into a Colab notebook cell or a `BENCHMARKS.md` and link from the PR.

## Benchmark steps (Colab T4)

The validation script `examples/colab_gpu_validation.py` performs these steps in order:

1. **Environment** — print `available_backends()`, `WarpBackend().is_available()`, device info, and `jax.devices()`.
2. **Warp single sim** — `OrigamiEngine(backend="warp").run(creases, duration=0.05, fold_duration=0.02, vibrate=True)`; assert finite `final_centroid`.
3. **JAX on GPU single sim** — `OrigamiEngine(backend="jax").run(...)` with the same parameters, for numerical comparison.
4. **Population evaluation** — `evaluate_population` at each batch size, time each run.
5. **Report** — print all timings, device info, and any errors.

To run manually in a Colab notebook:

```python
!pip install 'origami-engine-mjx[all]'
!git clone https://github.com/ihsan31415/origami-engine-mjx.git /content/origami-engine-mjx
%cd /content/origami-engine-mjx
!python examples/colab_gpu_validation.py
```

## Expected output (no numbers claimed)

The script prints a structured report. On a correctly configured T4:

- `warp is_available: True`
- `jax devices: [CudaDevice(id=0)]` (or similar)
- Finite centroids for all runs
- Per-batch-size timings (record these in your PR)

## Troubleshooting GPU issues

- **`warp is_available: False`** — Warp or `mujoco-warp` not installed, or no CUDA device. Check `nvidia-smi` in the Colab runtime (`Runtime → Change runtime type → T4 GPU`).
- **JAX uses CPU on GPU machine** — set `JAX_PLATFORMS=cuda` (or ensure `jax[cuda12]` is installed). Check `jax.devices()`.
- **Out of memory at large batch sizes** — reduce `batch_size` or `n` in the validation script.
- **Numerical mismatch vs classic MuJoCo** — MJX uses float32; small differences are expected (see [BACKENDS.md](BACKENDS.md)).

## See also

- [BACKENDS.md](BACKENDS.md) — backend semantics and feature table
- [examples/README.md](../examples/README.md) — runnable scripts
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — general runtime issues