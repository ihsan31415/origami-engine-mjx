# Migration from `origami-engine` (classic MuJoCo)

> **Documentation map:** [README](../README.md) · [Getting Started](GETTING_STARTED.md) · [Tutorials](TUTORIALS.md) · [Crease Format](CREASE_FORMAT.md) · [Simulation](SIMULATION.md) · [Evaluation](EVALUATION.md) · [API Reference](API_REFERENCE.md) · [Architecture](ARCHITECTURE.md) · [Troubleshooting](TROUBLESHOOTING.md) · [Glossary](GLOSSARY.md) · [Contributing](CONTRIBUTING.md)

This document maps API calls from the classic [`origami-engine`](https://github.com/ihsan31415/origami-engine) to `origami_engine_mjx` for users migrating their scripts and notebooks.

## Old vs. new entry points

The following table maps classic `origami_engine` functions to `origami_engine_mjx` equivalents:

| Old (`origami_engine`) | New (`origami_engine_mjx`) | Notes |
|---|---|---|
| `dynamics.run_simulation(creases, ...)` using `mujoco.mj_step` in-place | `OrigamiEngine(backend="jax").run(creases, ...)` using MJX immutable state | `duration`, `fold_duration`, `vibrate`, `amp_x`, `amp_z`, `freq_hz` map directly |
| `evaluation.evaluate_individual(creases, ...)` (classic MuJoCo) | `evaluate_individual(creases, ..., backend="jax")` | Same params; the new `backend` param selects MJX-JAX or MJX-Warp (legacy had no backend selector) |
| `evaluation.evaluate_population(pop, ...)` (classic MuJoCo) | `evaluate_population(pop, ..., backend="jax")` | `workers` has no equivalent; both MJX backends loop sequentially |
| `model.generate_xml` imported `require_mujoco` | `model.generate_xml` is pure XML generation (no MuJoCo needed) | Same function signature; MuJoCo import guard removed |
| `render_individual.py` renders via `mujoco.Renderer` | unchanged (classic MuJoCo still used for rendering) | Legacy script still works with classic `MjModel`/`MjData` |
| `evaluate_population(..., workers=N)` (ProcessPoolExecutor) | `engine.evaluate_population(..., batch_size=N)` | `workers` does not exist; `batch_size` is accepted but currently ignored (sequential per-individual loops) |
| `origami-render` CLI (entry point `origami_engine.render_individual:main`) | does not exist in `origami-engine-mjx` | See [TROUBLESHOOTING.md](TROUBLESHOOTING.md#rendering-and-video) for workarounds |

## API mapping examples

### Old script

```python
from origami_engine import evaluate_individual, evaluate_population

score = evaluate_individual(creases, duration=2.0)
scores = evaluate_population(pop, duration=2.0, workers=4)
```

### New script

```python
from origami_engine_mjx import OrigamiEngine, evaluate_individual, evaluate_population

engine = OrigamiEngine(backend="jax")  # or "warp" on GPU
score = evaluate_individual(creases, duration=2.0, backend="jax")
scores = evaluate_population(pop, duration=2.0, backend="jax")
# NOTE: legacy "workers" does not exist. batch_size is accepted but currently
# ignored by both backends; population scoring always uses +x displacement.
# or via the engine object:
score = engine.evaluate(creases, duration=2.0)
result = engine.evaluate_population(pop, duration=2.0)
```

## Behavioral notes

| Aspect | Classic `origami_engine` | `origami_engine_mjx` |
|---|---|---|
| **State mutability** | In-place (`data.qpos[i] = v`) | Immutable (`data = data.replace(qpos=data.qpos.at[i].set(v))`) |
| **Step function** | `mujoco.mj_step(model, data)` returns `data` in-place | `mjx.step(model, data)` returns new state; Warp `mjw.step` struct mutate |
| **Control arrays** | NumPy arrays OK | Must pass `jax.numpy` arrays to MJX controls; mixing causes `AttributeError` |
| **Name lookups** | `model.joint("table_x").qposadr` etc. | Use classic `MjModel` kept in the bundle; MJX model has no `.joint()` views |
| **Population `workers`** | `ProcessPoolExecutor` with `workers=N` parameter | Does not exist — both MJX backends evaluate sequentially; `batch_size` is accepted but currently ignored |
| **Render video** | `make render` → `individual_simulation.mp4` | MJX package has no renderer; use classic MuJoCo or copy state via `mjx.get_data` |
| **Constants sign convention** | `M = π/4` = valley, `V = -π/4` = mountain | Same constants inherited; documented in [CREASE_FORMAT.md](CREASE_FORMAT.md) |
| **`origami-render` entry point** | installed as `origami-render` CLI | removed from `pyproject.toml` (module doesn't exist); see [TROUBLESHOOTING.md](TROUBLESHOOTING.md) |

## `origami-render` and video rendering

The legacy `origami-engine` installed a console script `origami-render` that rendered a video of a single pattern simulation and wrote `individual_simulation.mp4` + `render_report.json` to the working directory.

**`origami-engine-mjx` does not ship a renderer.** To render video:

- Use classic MuJoCo: `python -m origami_engine.render_individual` (still available in the legacy package).
- Copy MJX state back: `mjx.get_data(model, data)` → classic `MjData` → `mujoco.Renderer`.
- The `examples/` folder includes `05_plot_pattern.py` for static PNG output (Agg backend).

## See also

- [EVALUATION.md](EVALUATION.md) — fitness functions and evaluation in MJX
- [BACKENDS.md](BACKENDS.md) — MJX-JAX vs MJX-Warp differences
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — runtime problems and install guidance
- [ARCHITECTURE.md](ARCHITECTURE.md) — module responsibilities and data flow