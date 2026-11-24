# Backends: MJX-JAX vs MJX-Warp

> **Documentation map:** [README](../README.md) · [Getting Started](GETTING_STARTED.md) · [Tutorials](TUTORIALS.md) · [Crease Format](CREASE_FORMAT.md) · [Simulation](SIMULATION.md) · [Evaluation](EVALUATION.md) · [Fitness](FITNESS.md) · [API Reference](API_REFERENCE.md) · [Architecture](ARCHITECTURE.md) · [Troubleshooting](TROUBLESHOOTING.md) · [Glossary](GLOSSARY.md) · [Integrations](INTEGRATIONS.md) · [Optimization (ES)](OPTIMIZATION_ES.md) · [Search space](SEARCH_SPACE.md) · [Populations](POPULATIONS.md) · [RL](RL.md) · [Contributing](CONTRIBUTING.md)

## MJX-JAX (`backend="jax"`)

- **Engine**: `mujoco.mjx` running on JAX/XLA.
- **Hardware**: CPU (local dev, no CUDA needed), NVIDIA GPU, or TPU via JAX device backends.
- **Transformations**: fully compatible with `jax.jit`, `jax.vmap`, `jax.grad`, `jax.lax.scan`.
- **State**: immutable JAX pytree; `data.replace(...)` pattern.
- **Batching**: `jax.vmap` is possible in principle when individuals share the same model topology, but it is **not implemented** in this package. Individuals with different crease patterns have different topologies, so `run_batch` loops per individual and calls `run` eagerly (no JIT, no vmap, `batch_size` ignored).
- **Performance on CPU**: mesh-contact models are slow (~70 s/step for the 16-face Miura on CPU; ~1.4 s/step for a 2-face model). JIT compilation of `mjx.step` on this model is prohibitively slow, so the backend runs eager by default (`jit=False`).
- **Install**: `pip install 'origami-engine-mjx[jax]'`

## MJX-Warp (`backend="warp"`)

- **Engine**: `mujoco_warp` running on NVIDIA Warp (CUDA kernels).
- **Hardware**: NVIDIA CUDA GPU only. Not usable on CPU or Intel graphics.
- **Transformations**: Warp state is **not** a JAX pytree — do not apply `jax.vmap`/`jax.jit` to Warp state.
- **Batching**: **not implemented yet.** `run_batch` loops over individuals and calls `run` in Python; `batch_size` is accepted but ignored. Native batched Warp execution (`mjw.step` over batched state) remains a Colab-GPU integration target.
- **Install**: `pip install 'origami-engine-mjx[warp]'` (pulls `mujoco-warp` + `warp-lang`).
- **Status**: implemented and import-guarded; **GPU execution must be validated on Colab T4** (see [COLAB_GPU.md](COLAB_GPU.md)). No GPU test results are claimed here.

## Backend selection

```python
OrigamiEngine(backend="jax")  # explicit
OrigamiEngine(backend="warp")  # explicit; raises RuntimeError if no CUDA GPU
OrigamiEngine(backend="auto")  # warp if CUDA available, else jax
```

The engine never silently switches backends after construction. If you request `"warp"` without a GPU, you get a clear `RuntimeError`, not a fallback.

## Performance

Measured locally on CPU (`jax` 0.12.1, `mujoco` 4.0.1): the 16-face reference Miura mesh model costs roughly **70 s per 1 ms step**, while a tiny 2-face model costs roughly **1.4 s per step**. Cost scales with `duration / 0.001` steps. JIT-compiling `mjx.step` for the mesh-contact model is prohibitively slow, so the JAX backend runs eager by default.

Use a GPU (JAX or Warp) for full-scale workloads. **No GPU numbers are claimed in this repository**; run [`examples/colab_gpu_validation.py`](../examples/colab_gpu_validation.py) on Colab T4 and record your own results — see [COLAB_GPU.md](COLAB_GPU.md#benchmark-steps-colab-t4).

## Known feature differences / limitations

- **Rendering**: neither MJX backend renders directly. Use classic MuJoCo (`MjModel`/`MjData` + `mujoco.Renderer`) for video, copying state via `mjx.get_data`.
- **Model features**: MJX supports the features used here (hinge/slide/free joints, equality `connect`, position actuators, mesh geoms, `implicitfast`), verified by successful `mjx.put_model` + stepping on CPU. Warp feature support is a subset — validate on GPU.
- **Numerical differences**: MJX uses float32 XLA kernels; classic MuJoCo uses float64 C++. Expect small numerical differences; investigate rather than forcing identical outputs.

## Feature/limitation comparison table

| Feature | Classic `origami_engine` (MuJoCo C++) | MJX-JAX | MJX-Warp |
|---|---|---|---|
| State semantics | In-place (`mujoco.mj_step`) | Immutable pytree (`mjx.step` returns new state) | Warp struct (in-place via `.numpy()` views) |
| `jax.jit` / `jax.vmap` / `jax.grad` | No | Yes (CPU/GPU/TPU) | No (Warp-native batching instead) |
| Rendering | Yes (`mujoco.Renderer`) | No (copy back via `mjx.get_data`) | No (copy back via `mjx.get_data`) |
| CPU execution | Fast | Slow for mesh-contact models (~70 s/step) | N/A (CUDA only) |
| GPU execution | No | Yes (via JAX backend) | Yes (CUDA only) |
| Population batching | `ProcessPoolExecutor` (CPU) | Sequential per-individual loop (no JIT/vmap) | Sequential per-individual loop (no native batching) |
| Import guard | `require_mujoco` | `require_mjx_jax` | `require_mjx_warp` |
| `workers` parameter | Yes (ProcessPool) | No | No |
| `batch_size` parameter | N/A | Accepted, currently ignored | Accepted, currently ignored |
| Differentiability | No | Yes | No |
| `jit` parameter on `.run()` | N/A | `jit=False` default (eager); `jit=True` supported | N/A (no `jit` param) |

## Backend selection logic

- `backend="auto"`: prefer `warp` if `WarpBackend().is_available()` (CUDA GPU present), otherwise `jax`.
- `backend="warp"` without CUDA: raises `RuntimeError` (no silent fallback).
- `backend="jax"`: always available if `mujoco-mjx` + `jax` installed.

## See also

- [COLAB_GPU.md](COLAB_GPU.md) — GPU setup, validation, benchmarking
- [MIGRATION.md](MIGRATION.md) — classic → MJX API mapping
- [ARCHITECTURE.md](ARCHITECTURE.md) — module responsibilities and data flow
- [SIMULATION.md](SIMULATION.md) — model structure, stepping loop, constants
- [RL.md](RL.md) — the Gymnasium/JAX environments are MJX-JAX only