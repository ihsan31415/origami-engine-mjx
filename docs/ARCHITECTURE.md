# Architecture

> **Documentation map:** [README](../README.md) · [Getting Started](GETTING_STARTED.md) · [Tutorials](TUTORIALS.md) · [Crease Format](CREASE_FORMAT.md) · [Simulation](SIMULATION.md) · [Evaluation](EVALUATION.md) · [API Reference](API_REFERENCE.md) · **Architecture** · [Troubleshooting](TROUBLESHOOTING.md) · [Glossary](GLOSSARY.md) · [Integrations](INTEGRATIONS.md) · [Optimization (ES)](OPTIMIZATION_ES.md) · [RL](RL.md) · [Contributing](CONTRIBUTING.md)

`origami_engine_mjx` keeps all physics-independent logic in shared modules and isolates backend-specific execution in `origami_engine_mjx/backends/`.

## Data flow

```
creases (Crease list | dict list | JSON path)
   │  normalize_creases()                       (serialization.py — coerce + validate)
   ▼
build_pattern(creases) → Pattern                 (geometry.py — Shapely polygonize → faces → spanning tree → loop edges)
   │
   ▼
generate_xml(pat, creases)                     (model.py — → MJCF XML string + info dict (hinges, targets))
   │   ▲                                         │
   │   │ pure XML (no MuJoCo runtime needed)   │
   ▼   │                                         ▼
backends.get_backend(name).build_model(xml, info)
   │   jax:  MjModel.from_xml_string → mjx.put_model → mjx.make_data
   │   warp: MjModel.from_xml_string → mjw.make_model → mjw.make_data
   ▼
backend.run(creases, ...)                      (simulation loop: table vibration, actuator ramp, mjx.step/mjw.step)
   │
   ▼
SimulationResult(initial_centroid, final_centroid, ...) → fitness = projection on trajectory
```

## Module responsibilities

| Module | Responsibility | Backend-specific? |
|---|---|---|
| `constants.py`, `types.py` | Physical constants, dataclasses (`Crease`, `Pattern`, `SimulationResult`, `PopulationResult`) | No |
| `geometry.py`, `serialization.py`, `patterns.py`, `plotting.py` | Crease-graph geometry, JSON I/O, reference patterns, Matplotlib visualization | No |
| `model.py` | `generate_xml()` — builds MJCF XML string from `Pattern`; no MuJoCo runtime needed | No |
| `engine.py` | `OrigamiEngine` facade with `.run()`, `.evaluate()`, `.evaluate_population()` | No (delegates to backend) |
| `evaluation.py` | `evaluate_individual()`, `evaluate_population()` — top-level convenience functions | No |
| `integrations/` | ES/RL/AI **interfaces** (no algorithms): `SearchSpace`, `OrigamiEvaluator`, optimizer adapters, Gymnasium `OrigamiEnv`/`OrigamiVectorEnv`, pure-JAX `reset`/`step`, plumbing | No (thin layer over `OrigamiEngine`; JAX-optional) |
| `backend.py` | Lazy import guards for `mujoco`, `mjx_jax`, `mjx_warp`; raises clear errors | No (metadata only) |
| `backends/base.py` | `SimulationBackend` ABC: `build_model()`, `make_state()`, `run()`, `run_batch()` | Interfaces only |
| `backends/jax_backend.py` | MJX-JAX: `mjx.put_model`, `mjx.make_data`, `mjx.step` (eager by default); `jit` optional | **Yes** |
| `backends/warp_backend.py` | MJX-Warp: `mjw.make_model`, `mjw.make_data`, `mjw.step`; CUDA-only | **Yes** |

## Key MJX differences from classic MuJoCo

1. **Immutable state**: MJX `Data` is a JAX pytree. Use `data = data.replace(qpos=data.qpos.at[i].set(v))` instead of `data.qpos[i] = v`.
2. **Step returns new state**: `data = mjx.step(model, data)` instead of `mujoco.mj_step(model, data)` (in-place).
3. **JAX arrays for controls**: control targets must be `jax.numpy` arrays, not plain NumPy (mixing causes `AttributeError: 'numpy.ndarray' has no attribute 'at'` inside MJX).
4. **Name lookups**: use the classic `MjModel` (kept in the model bundle) for `model.joint("table_x").qposadr` etc.; the MJX model has no `.joint()` views.
5. **Rendering**: `mujoco.Renderer` only works with classic `MjModel`/`MjData`. The Warp/JAX backends don't render; use `mjx.get_data` to copy state back to a classic `MjData` for rendering. This repository ships **no** renderer — see [MIGRATION.md](MIGRATION.md#origami-render-and-video-rendering).

## The `Crease` → `Pattern` → Simulation pipeline

1. **Normalize**: `normalize_creases()` coerces JSON path, `Crease` list, or dict list into `list[Crease]`. Raises `ValueError`/`TypeError` on missing keys, wrong types, empty list, or `None`.

2. **Build pattern**: `build_pattern()` polygonizes the crease arrangement with Shapely:
   - Validates tiling (face areas sum to `S²` within 1e-6), no sliver faces (`area < 1e-4`), connected faces.
   - Produces a spanning tree from BFS on the largest-area face (`pat.root`).
   - Tree edges become hinge joints; loop edges become equality constraints.

3. **Generate XML**: `generate_xml()` builds MJCF:
   - One body per face, triangulated mesh, `<freejoint>` on root.
   - Each hinge has `position` actuator with ramped target over `fold_duration`.
   - Support table with kinematic vibration joints (`table_x`, `table_z).

4. **Backend model**: `JaxBackend.build_model()` / `WarpBackend.build_model()` (from the backend registry):
   - Parses MJCF with classic `MjModel.from_xml_string()` for name lookups.
   - Converts to MJX pytree (`mjx.put_model`) or Warp model (`mjw.make_model`).

5. **Run loop**: `backend.run()`:
   - Zeroes hinge qpos; computes initial centroid.
   - For each timestep: smoothstep ramps actuators, kinematically drives table vibration, calls `mjx.step`/`mjw.step`.
   - Checks for non-finite joints; returns `SimulationResult`.

## See also

- [SIMULATION.md](SIMULATION.md) — detailed model structure, stepping loop, constants
- [BACKENDS.md](BACKENDS.md) — MJX-JAX vs MJX-Warp execution details
- [CREASE_FORMAT.md](CREASE_FORMAT.md) — crease input format and validation
- [INTEGRATIONS.md](INTEGRATIONS.md) — where `origami_engine_mjx/integrations/` sits in the package
- [MIGRATION.md](MIGRATION.md) — API mapping from classic `origami-engine`
- [examples/README.md](../examples/README.md) — runnable demo scripts