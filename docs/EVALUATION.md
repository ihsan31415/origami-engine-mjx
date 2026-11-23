# Evaluation & Fitness

> **Documentation map:** [README](../README.md) · [Getting Started](GETTING_STARTED.md) · [Tutorials](TUTORIALS.md) · [Crease Format](CREASE_FORMAT.md) · [Simulation](SIMULATION.md) · **Evaluation** · [API Reference](API_REFERENCE.md) · [Architecture](ARCHITECTURE.md) · [Troubleshooting](TROUBLESHOOTING.md) · [Glossary](GLOSSARY.md) · [Contributing](CONTRIBUTING.md)

This document describes the fitness signal used throughout origami-engine-mjx and how to evaluate patterns and populations of patterns. All statements below were verified against the implementation (`engine.py`, `evaluation.py`, `backends/jax_backend.py`, `backends/warp_backend.py`).

## Forward displacement

The engine's fitness signal is the **forward displacement** of the paper centroid: how far the sheet's centre of mass travels along a requested direction while it folds.

### `compute_forward_displacement(sim_result, trajectory=(1.0, 0.0, 0.0)) -> float`

Project the centroid displacement onto the trajectory direction:

```python
from origami_engine_mjx import compute_forward_displacement

score = compute_forward_displacement(sim_result, trajectory=(1.0, 0.0, 0.0))
```

The exact computation depends on the **length of `trajectory`**:

```python
delta = sim_result.final_centroid - sim_result.initial_centroid   # (2,)

# 2-element trajectory: NOT normalized
score = dot(delta[:2], trajectory)

# 3-element trajectory: normalized, z of delta is 0
delta_3d = (delta_x, delta_y, 0)
score = dot(delta_3d, trajectory / norm(trajectory))
```

- A **2-vector** `(x, y)` is used as-is, so the score is scaled by its magnitude. `(1.0, 0.0)` gives the +x displacement; `(3.0, 4.0)` gives `3·Δx + 4·Δy`.
- A **3-vector** `(x, y, z)` is normalized internally, so only its direction matters. The sheet's centroid motion is planar (`z = 0`), so a direction with a non-zero z component reduces the score.
- A zero 2-vector returns `0.0`. A zero 3-vector divides by zero and returns `nan` (with a NumPy `RuntimeWarning`) — it does **not** raise.
- `(N, 2)` / `(N, 3)` waypoint arrays and "direction = last − first" are **not supported** by this function; passing an array with a leading axis raises a NumPy shape error.

`sim_result` is normally the `SimulationResult` from `OrigamiEngine.run()`; any object exposing `initial_centroid` and `final_centroid` works.

**Raises:** no explicit validation. Malformed shapes raise NumPy errors; non-finite centroids propagate to `nan`.

### `evaluate_individual(...) -> float | dict`

```python
from origami_engine_mjx import evaluate_individual, load_default_creases

details = evaluate_individual(
    load_default_creases(),
    duration=8.0,
    trajectory=(1.0, 0.0, 0.0),
    fold_duration=3.0,
    return_details=True,
)
```

- Returns a `float` (the forward displacement), or a `dict` when `return_details=True`.
- The details dict has **exactly** these keys: `displacement`, `initial_centroid`, `final_centroid`, `duration`, `backend`, `n_faces`, `n_loops`. There is no `trajectory` key.
- Internally: builds `OrigamiEngine(backend=backend)` → `run(...)` (same `duration`/`fold_duration`/`vibrate`/amps/`freq`) → `compute_forward_displacement`.
- Parameters: `duration` (simulated seconds, must be > 0), `trajectory` (2- or 3-vector; see above), `fold_duration` (smoothstep ramp time), `vibrate` (support-table vibration on/off), `amp_x` / `amp_z` / `freq_hz` (vibration settings), `backend` (`"jax"` or `"warp"`).
- There is **no finite-score check**: a `nan` displacement is returned unchanged.
- **Raises:** `ValueError` for `duration <= 0`, invalid crease input, or invalid pattern; `FloatingPointError` if the simulation produces non-finite joint coordinates; `RuntimeError` if the backend is unavailable.

### `evaluate_population(...) -> list[float]`

```python
from origami_engine_mjx import evaluate_population, load_default_creases

population = [load_default_creases() for _ in range(8)]
scores = evaluate_population(
    population,
    duration=4.0,
    trajectory=(1.0, 0.0, 0.0),
    fold_duration=2.0,
    backend="jax",
    error_policy="nan",
)
```

- Scores a list of crease patterns using the given backend (`"jax"` or `"warp"`).
- `error_policy="raise"` (default): after the batch, any non-finite score raises `RuntimeError("individual {i} produced non-finite score: ...")`. Any other value returns the raw scores; because both backends catch per-individual exceptions and store `nan`, this behaves as a "nan" policy.
- Results preserve input order in both modes.
- **`batch_size`** is accepted and forwarded to the backend, but both built-in `run_batch` implementations currently **ignore it**.
- **`workers`** does not exist in this package. Both backends use sequential per-individual loops; there is no process pool.

| Mode | Actual behaviour |
|---|---|
| `backend="jax"` | Per-individual Python loop; each `run(...)` uses eager `mjx.step` unless called directly with `jit=True` (the batch path does not set it). |
| `backend="warp"` | Per-individual Python loop over `mjw.step`; requires a CUDA GPU. Native batched Warp execution is not implemented. |

### Current behaviour caveats

These are verified properties of the current code, not recommended usage:

1. **`trajectory` is ignored by population evaluation.** `OrigamiEngine.evaluate_population` normalizes and validates the vector (a zero vector raises `ValueError("trajectory direction must be non-zero.")`), then calls `backend.run_batch(...)` without it. Both backends score `final_centroid[0] - initial_centroid[0]`, i.e. **+x only**. To score along another direction, evaluate individuals with `evaluate_individual(..., trajectory=...)` or call `compute_forward_displacement` on your own `SimulationResult`.
2. **`batch_size` has no effect.** Neither backend sub-batches or vmaps; it is accepted for forward compatibility.
3. **The JAX batch path is not JIT-compiled.** `JaxBackend.run_batch` calls `self.run(...)` without `jit=True`, so each individual runs eager `mjx.step`.
4. **Per-individual failures become `nan`.** `run_batch` catches every exception for an individual and stores `nan`; `error_policy="raise"` only inspects the final array.
5. **2-element trajectories are not normalized** in `compute_forward_displacement` (see above).

See the final report accompanying this documentation update for the tracked decision on items 1–2.

### Practical guidance

- `duration`: keep it small while iterating (e.g. 0.5–1.0 s); scale up for final numbers. Cost is linear in `duration` (1 ms timestep).
- Disable vibration (`vibrate=False`) for a deterministic, purely fold-driven score; vibration adds the dynamic "shake" effect used in the README media.
- Compare runs with the same `duration`, `fold_duration`, vibration settings, and scoring direction — the score is only meaningful relative to identical settings.
- A population of patterns with *different* face counts has different model topologies, which is why both backends loop instead of vmapping.

## Typical GA-style loop

```python
from origami_engine_mjx import evaluate_individual, load_default_creases
import numpy as np

population = [load_default_creases() for _ in range(8)]
scores = [],
for individual in population:
    scores.append(
        evaluate_individual(
            individual,
            duration=4.0,
            trajectory=(1.0, 0.0, 0.0),  # applied per individual
            fold_duration=2.0,
            backend="jax",
        )
    )

ranked = sorted(zip(scores, population), key=lambda t: t[0], reverse=True)
best_score, best_creases = ranked[0]
print(f"best: {best_score:.6f} m")
```

If your fitness direction is +x (the current population-scoring direction), the shorter form is equivalent:

```python
from origami_engine_mjx import evaluate_population, load_default_creases

population = [load_default_creases() for _ in range(8)]
scores = evaluate_population(
    population,
    duration=4.0,
    fold_duration=2.0,
    backend="jax",
    error_policy="nan",
)
```

## See also

- [SIMULATION.md](SIMULATION.md) — what `engine.run` does with these parameters
- [TUTORIALS.md](TUTORIALS.md) — Tutorials 3 and 6
- [API_REFERENCE.md](API_REFERENCE.md) — full signatures and raise conditions
- [`examples/01_evaluate_cpu_jax.py`](../examples/01_evaluate_cpu_jax.py) — minimal single + population run on CPU
- [`examples/03_batch_population.py`](../examples/03_batch_population.py) — population loop with timing
