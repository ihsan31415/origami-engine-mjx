# Backends: MJX-JAX vs MJX-Warp

## MJX-JAX (`backend="jax"`)

- **Engine**: `mujoco.mjx` running on JAX/XLA.
- **Hardware**: CPU (local dev, no CUDA needed), NVIDIA GPU, or TPU via JAX device backends.
- **Transformations**: fully compatible with `jax.jit`, `jax.vmap`, `jax.grad`, `jax.lax.scan`.
- **State**: immutable JAX pytree; `data.replace(...)` pattern.
- **Batching**: `jax.vmap` works when individuals share the same model topology. Individuals with different crease patterns have different topologies, so `run_batch` loops per individual and JIT-compiles each; same-topology batches can use `vmap`.
- **Performance on CPU**: mesh-contact models are slow (~70 s/step for the 16-face Miura on CPU; ~1.4 s/step for a 2-face model). JIT compilation of `mjx.step` on this model is prohibitively slow, so the backend runs eager by default (`jit=False`).
- **Install**: `pip install 'origami-engine-mjx[jax]'`

## MJX-Warp (`backend="warp"`)

- **Engine**: `mujoco.warp` running on NVIDIA Warp (CUDA kernels).
- **Hardware**: NVIDIA CUDA GPU only. Not usable on CPU or Intel graphics.
- **Transformations**: Warp state is **not** a JAX pytree — do not apply `jax.vmap`/`jax.jit` to Warp state. Use Warp-native batched execution (`mjw.step` over batched state) for population evaluation.
- **Batching**: native batched simulation on GPU; configure `batch_size` and contact/constraint capacities via the Warp model.
- **Install**: `pip install 'origami-engine-mjx[warp]'` (pulls `mujoco-warp` + `warp-lang`).
- **Status**: implemented and import-guarded; **GPU execution must be validated on Colab T4** (see `docs/COLAB_GPU.md`). No GPU test results are claimed here.

## Backend selection

```python
OrigamiEngine(backend="jax")    # explicit
OrigamiEngine(backend="warp")   # explicit; raises RuntimeError if no CUDA GPU
OrigamiEngine(backend="auto")   # warp if CUDA available, else jax
```

The engine never silently switches backends after construction. If you request `"warp"` without a GPU, you get a clear `RuntimeError`, not a fallback.

## Known feature differences / limitations

- **Rendering**: neither MJX backend renders directly. Use classic MuJoCo (`MjModel`/`MjData` + `mujoco.Renderer`) for video, copying state via `mjx.get_data`.
- **Model features**: MJX supports the features used here (hinge/slide/free joints, equality `connect`, position actuators, mesh geoms, `implicitfast`), verified by successful `mjx.put_model` + stepping on CPU. Warp feature support is a subset — validate on GPU.
- **Numerical differences**: MJX uses float32 XLA kernels; classic MuJoCo uses float64 C++. Expect small numerical differences; investigate rather than forcing identical outputs.
