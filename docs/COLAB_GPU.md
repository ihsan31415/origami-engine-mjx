# Colab GPU validation (NVIDIA T4)

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
