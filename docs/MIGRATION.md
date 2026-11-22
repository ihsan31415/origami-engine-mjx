# Migration from `origami-engine` (classic MuJoCo)

## What changed

| Old (`origami_engine`) | New (`origami_engine_mjx`) |
|---|---|
| `dynamics.run_simulation(creases, ...)` using `mujoco.mj_step` in-place | `OrigamiEngine(backend=...).run(creases, ...)` using MJX immutable state |
| `evaluation.evaluate_individual(creases, ...)` (classic MuJoCo) | `evaluation.evaluate_individual(creases, ..., backend="jax")` |
| `model.generate_xml` imported `require_mujoco` | `model.generate_xml` is pure XML generation (no MuJoCo needed) |
| `render_individual.py` renders via `mujoco.Renderer` | unchanged (classic MuJoCo still used for rendering) |
| `evaluate_population(..., workers=N)` (ProcessPoolExecutor) | `engine.evaluate_population(..., batch_size=N)` (backend-native batching) |

## API mapping

```python
# OLD
from origami_engine import evaluate_individual, evaluate_population
score = evaluate_individual(creases, duration=2.0)
scores = evaluate_population(pop, duration=2.0, workers=4)

# NEW
from origami_engine_mjx import OrigamiEngine, evaluate_individual, evaluate_population
engine = OrigamiEngine(backend="jax")          # or "warp" on GPU
score = evaluate_individual(creases, duration=2.0, backend="jax")
scores = evaluate_population(pop, duration=2.0, backend="jax")
# or via the engine object:
score = engine.evaluate(creases, duration=2.0)
result = engine.evaluate_population(pop, duration=2.0)
```

## Behavioral notes

- **State immutability**: any custom code that mutated `data.qpos[i] = v` must use `data = data.replace(qpos=data.qpos.at[i].set(v))`.
- **Step**: `data = mjx.step(m, data)` returns a new state.
- **Controls**: pass `jax.numpy` arrays to `ctrl`, not NumPy arrays.
- **Performance**: classic MuJoCo on CPU is fast per-step; MJX-JAX on CPU is slow for mesh models (use GPU). If you need CPU speed with the old behavior, keep using `origami_engine` — this package targets MJX.
