# Evaluation & Fitness

> **Documentation map:** [README](../README.md) · [Getting Started](GETTING_STARTED.md) · [Tutorials](TUTORIALS.md) · [Crease Format](CREASE_FORMAT.md) · [Simulation](SIMULATION.md) · **Evaluation** · [API Reference](API_REFERENCE.md) · [Architecture](ARCHITECTURE.md) · [Troubleshooting](TROUBLESHOOTING.md) · [Glossary](GLOSSARY.md) · [Contributing](CONTRIBUTING.md)

This document describes the fitness signal used throughout the origami-engine-mjx system and how to evaluate patterns and populations of patterns.

## Forward displacement

The engine's fitness signal is the **forward displacement** of the paper centroid: how far the sheet's center of mass travels along a requested direction while it folds.

### `compute_forward_displacement(sim_result, trajectory=(1.0, 0.0, 0.0)) -> float`

Project the centroid displacement onto the trajectory direction:

```python
from origami_engine_mjx import compute_forward_displacement

score = compute_forward_displacement(sim_result, trajectory=(1.0, 0.0, 0.0))
```

```
direction = normalize(trajectory)
delta     = sim_result.final_centroid - sim_result.initial_centroid
score     = dot(delta, direction)        # 2D: uses x,y · 3D: delta embedded in 3D, z=0
```

- 2-vector `(x, y)` or 3-vector `(x, y, z)`: direction of travel; normalized internally.
- `(N, 2)` / `(N, 3)` waypoint array (N ≥ 2): direction = `last − first`, normalized.
- Returns a `float` (signed displacement in meters). Zero 2-vector → `0.0`; non-zero 2-vector → normalized; zero 3-vector → raises `ValueError`.

**Raises:**
- `ValueError` on malformed/zero/non-finite/non-finite-norm input.
- Returns `nan` for degenerate cases (e.g. computed `nan` if centroids are non-finite).

### `evaluate_individual(...) -> float | dict`

```python
from origami_engine_mjx import evaluate_individual
from origami_engine_mjx import load_default_creases

details = evaluate_individual(
    load_default_creases(),
    duration=8.0,
    trajectory=(1.0, 0.0, 0.0),  # fitness direction: +x
    fold_duration=3.0,
    return_details=True,
)
```

- Returns a `float` (the forward displacement), or a `dict` when `return_details=True` with keys: `displacement`, `initial_centroid`, `final_centroid`, `duration`, `trajectory`, `n_faces`, `n_loops`, `backend`.
- Internally: `normalize_creases` → `engine.run(...)` (same `duration`/`fold_duration`/`vibrate`/amps/`freq`) → `compute_forward_displacement`.
- Parameters: `duration` (simulated seconds, must be > 0), `trajectory` (direction or waypoints), `fold_duration` (smoothstep ramp time), `vibrate` (support-table vibration on/off), `amp_x`/`amp_z` / `freq_hz` (vibration defaults), `backend` (`"jax"` or `"warp"`).
- **Raises:**
  - `ValueError` for `duration <= 0`, invalid crease input, invalid pattern, bad trajectory.
  - `RuntimeError` for non-finite score.

### `evaluate_population(...) -> list[float]`

```python
from origami_engine_mjx import evaluate_population
from origami_engine_mjx import load_default_creases

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
- `error_policy="raise"` (default): a failing individual raises `RuntimeError("individual {i} produced non-finite score: ...")`. `error_policy="nan"`: the slot becomes `float("nan")` and evaluation continues.
- Results preserve input order in both modes.
- **`workers` note**: The MJX backends do **not** have a `workers` parameter. Batching is handled natively: each individual is evaluated individually in a per-individual loop (JAX backend), with JIT compilation per individual; Warp backend loops per-individual when topology differs, otherwise native GPU batched execution with `batch_size`.

| Mode | Description |
|---|---|
| `backend="jax"` | Each individual simulated individually; JIT-compiled per individual. |
| `backend="warp"` | Per-individual loop; native GPU batched execution when `batch_size` is set and individuals share the same topology. |

- `duration`: kept small while iterating (e.g. 0.5–1.0 s); scale up for final numbers. Cost is linear in `duration` (1 ms timestep).
- Disable vibration (`vibrate=False`) when you want a deterministic, purely fold-driven score; vibration adds the dynamic "shake" effect shown in the README videos.
- Compare runs with the same `duration`, `fold_duration`, `trajectory`, and vibration settings — the score is only meaningful relative to identical settings.

## Typical GA-style loop

```python
from origami_engine_mjx import evaluate_population, load_default_creases


def fitness(population):
    return evaluate_population(
        population,
        duration=4.0,
        trajectory=(1.0, 0.0, 0.0),
        fold_duration=2.0,
        backend="jax",
        error_policy="nan",
    )


population = [load_default_creases() for _ in range(8)]
scores = fitness(population)
ranked = sorted(zip(scores, population), key=lambda t: t[0], reverse=True)
best_score, best_creases = ranked[0]
print(f"best: {best_score:.6f} m")
```

## See also

- [SIMULATION.md](SIMULATION.md) — what `engine.run` does with these parameters
- [TUTORIALS.md](TUTORIALS.md) — Tutorials 3 and 6
- [API_REFERENCE.md](API_REFERENCE.md) — full signatures and raise conditions
- `examples/01_evaluate_reference.py` — minimal runnable demo
- `examples/04_evaluate_population.py` — population evaluation demo