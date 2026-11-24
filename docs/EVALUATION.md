# Evaluation & Fitness

> **Documentation map:** [README](../README.md) · [Getting Started](GETTING_STARTED.md) · [Tutorials](TUTORIALS.md) · [Crease Format](CREASE_FORMAT.md) · [Simulation](SIMULATION.md) · **Evaluation** · [API Reference](API_REFERENCE.md) · [Architecture](ARCHITECTURE.md) · [Troubleshooting](TROUBLESHOOTING.md) · [Glossary](GLOSSARY.md) · [Integrations](INTEGRATIONS.md) · [Optimization (ES)](OPTIMIZATION_ES.md) · [Search space](SEARCH_SPACE.md) · [Populations](POPULATIONS.md) · [RL](RL.md) · [Contributing](CONTRIBUTING.md)

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
delta = sim_result.final_centroid - sim_result.initial_centroid  # (2,)

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
scores = []
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

## Configurable spaces and per-individual paths

The `OrigamiEvaluator` in `origami_engine_mjx.integrations` accepts a
`SearchSpace` **or** a [`DesignSpace`](SEARCH_SPACE.md) with no API change. When
the space can change the crease **topology** between individuals (geometry or
presence variables), the fast batch path decodes every genome, groups them by
`topology_signature` and sends one group per backend call. Because both built-in
backends already loop per individual (and rebuild the model inside that loop),
each group is scored **exactly like one-by-one evaluation**; the grouping is
bookkeeping only and currently gives **no throughput benefit**. Each
result records which path scored it, on `EvalRecord.path` / `BatchResult.paths`:
`batched` (a same-topology batch), `bucketed` (a topology group of more than
one), `single` (one individual), `invalid` (decode failed) or `cached`.

## Fold (hinge-target) fitness and blends

Movement asks *how far did the sheet go*; the fold score asks *did each hinge
reach the angle it was designed for*. Both live in
`origami_engine_mjx.fitness`, and any entry point takes `fold_weight` /
`movement_weight` to blend them:

```python
from origami_engine_mjx import OrigamiEngine, fold_report, load_default_creases

engine = OrigamiEngine(backend="jax")
creases = load_default_creases()
result = engine.run(creases, duration=0.02)
report = fold_report(result, percent=True)
print(report["score"], report["mean_abs_error_deg"])
print(engine.evaluate(creases, duration=0.02, fold_weight=0.3))  # 70 % movement + 30 % folding
```

`fold_weight=0` is the default and gives exactly the movement score. See
[FITNESS.md](FITNESS.md) for the formulas, the hinge-actuation options and
practical advice on choosing a split.

### Which evaluation path runs

With `fold_weight = 0` (default) nothing changes: `evaluate_batch` uses the
`batched` / `bucketed` / `single` paths described above. Turning on folding or
per-hinge actuation forces the **structured** per-genome path (`single`, or
`cached` with `use_cache=True`), because the fold score needs each result's
hinge angles. It is the same simulation count — just the bookkeeping route
changes — and `EvalRecord.descriptors` always gains `fold_score` and
`fold_mean_abs_error_deg`.

| Evaluator setting | Effect on the score |
|---|---|
| `fold_weight=0` (default) | Movement only — identical to previous releases. |
| `movement_weight=70, fold_weight=30` | `0.7·movement + 0.3·fold`; `raw` is the blend, `fitness` follows `maximize`. |
| `movement_scale=4.0` | Divides movement by 4 before blending when pushes are much larger than a sheet side. |
| `actuated=[True]*8 + [False]*7` | Drives the first eight hinges; the rest are passive (still scored). |

`movement_fraction` / `fold_fraction` on the evaluator report the normalized
split (e.g. `0.7` / `0.3`).

## Logging generations (`GenerationLogger`)

The library ships no optimization algorithm, so *you* own the loop. The
`GenerationLogger` records it: per-generation statistics, the global best, and
a checkpoint of the best individual of **every** generation.

```python
from origami_engine_mjx.integrations import GenerationLogger, SearchSpace

space = SearchSpace.from_default_pattern()
with GenerationLogger("runs/v1/run.jsonl", space, keep_every_best=True) as log:
    for gen in range(n_generations):
        genomes = my_algorithm.ask()  # your search
        batch = evaluator.evaluate_batch(genomes)  # library scoring
        log.log_batch(batch, genomes, generation=gen)  # optional per-individual rows
        summary = log.end_generation(gen, batch, genomes)
        print(
            summary.generation,
            f"best={summary.best_fitness:.5f}",
            f"mean={summary.mean_fitness:.5f}",
            "NEW BEST" if summary.improved else "",
        )
print(log.best_fitness, log.best_generation)
```

`end_generation(generation, batch, genomes)` returns a `GenerationSummary`
with `best_fitness`, `best_index`, `mean_fitness`, `std_fitness`,
`worst_fitness`, `median_fitness`, `n_ok`, `n_failed`, `n_cache_hits`,
`best_genome`, `global_best_fitness`/`global_best_generation`, `wall_time` /
`gen_wall_time` (so you can time each generation and the run), `evals_so_far`
and `improved`.

Files written (all optional, all next to `path`):

| file | contents |
|---|---|
| `run.jsonl` / `run.csv` | one row per call: `kind="generation"` summaries and, if you call it, `kind="eval"` rows tagged with `generation`/`index` |
| `run.summary.json` | all generation rows plus a `best` block (genome, generation, total evals, wall time) |
| `run.best.json` | checkpoint of the generation that last improved the global best |
| `run.best_gen_0007.json` | with `keep_every_best=True`, the best of **each** generation |

Options: `maximising=False` for a loss (affects which genome is "best" —
fitness values are logged unchanged), `descriptor_names=` to pin the CSV
columns, `path=None` for in-memory only (`log.history`, `log.evals`,
`log.curve()` for `(generations, best, mean, global_best)`). Checkpoints use
`save_best`/`load_best`, so `load_best("run.best.json", space)` returns the
genome and its metadata (generation, fitness, mean/std, `maximising`).

Reading a finished run back: `load_generation_history("run.jsonl")` returns the
generation rows as dicts, and
`origami_engine_mjx.rendering.plot_history(log)` (or the loaded rows) draws
best/mean per generation plus the running global best. Use
`RunLogger(path, include_generation=True)` if you only need flat per-individual
rows with a `generation` column.

## See also

- [SIMULATION.md](SIMULATION.md) — what `engine.run` does with these parameters, trajectory recording and rendering
- [OPTIMIZATION_ES.md](OPTIMIZATION_ES.md) — using the fitness signal as a black-box objective
- [SEARCH_SPACE.md](SEARCH_SPACE.md) — configurable `DesignSpace` targets
- [TUTORIALS.md](TUTORIALS.md) — Tutorials 3 and 6
- [API_REFERENCE.md](API_REFERENCE.md) — full signatures and raise conditions
- [`examples/01_evaluate_cpu_jax.py`](../examples/01_evaluate_cpu_jax.py) — minimal single + population run on CPU
- [`examples/03_batch_population.py`](../examples/03_batch_population.py) — population loop with timing
