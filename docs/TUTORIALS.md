# Tutorials

> **Documentation map:** [README](../README.md) · [Getting Started](GETTING_STARTED.md) · **Tutorials** · [Crease Format](CREASE_FORMAT.md) · [Simulation](SIMULATION.md) · [Evaluation](EVALUATION.md) · [API Reference](API_REFERENCE.md) · [Architecture](ARCHITECTURE.md) · [Troubleshooting](TROUBLESHOOTING.md) · [Glossary](GLOSSARY.md) · [Integrations](INTEGRATIONS.md) · [Optimization (ES)](OPTIMIZATION_ES.md) · [RL](RL.md) · [Contributing](CONTRIBUTING.md)

Step-by-step recipes that build on each other. Every snippet assumes the package is installed (see [GETTING_STARTED.md](GETTING_STARTED.md)).

```python
from origami_engine_mjx import (
    OrigamiEngine,
    Crease,
    build_pattern,
    evaluate_individual,
    evaluate_population,
    load_creases_from_json,
    load_default_creases,
    normalize_creases,
    plot_pattern,
)
```

## Tutorial 1 — Inspect a crease pattern without simulating

`build_pattern` turns raw creases into a validated face graph. This is pure geometry (Shapely) and does **not** need MuJoCo.

```python
creases = load_default_creases()
pat = build_pattern(creases)

print(pat.valid)  # True
print(pat.n_faces)  # 16 polygonal faces
print(pat.n_loops)  # 9 loop edges (non-tree edges, closed by equality constraints)
print(pat.root)  # index of the largest-area face (the free root body)
print(max(pat.depth.values()))  # depth of the hinge spanning tree
```

If the pattern is invalid, `pat.valid` is `False` and `pat.reason` says why (e.g. `"sliver face (area ...)"`, `"faces do not tile the sheet"`, `"disconnected faces"`). See [CREASE_FORMAT.md](CREASE_FORMAT.md) for the full list.

## Tutorial 2 — Run your first simulation

```python
from origami_engine_mjx import OrigamiEngine, load_default_creases

engine = OrigamiEngine(backend="jax")  # or backend="warp" on GPU
creases = load_default_creases()

result = engine.run(creases, duration=8.0, fold_duration=3.0, vibrate=True)
print(result.backend)  # "jax" or "warp"
print(result.final_centroid)  # final paper centroid (2D: x, y)
print(result.info["n_faces"], "faces,", result.info["n_hinges"], "hinges")
```

`result` is a `SimulationResult`:

| Field | Meaning |
|---|---|
| `result.model` | Backend bundle dict (`{"mjx_model", "classic_model", "info"}` for JAX; `{"warp_model", "classic_model", "info"}` for Warp) |
| `result.data` | Backend-specific data state (final step) |
| `result.pattern` | The validated `Pattern` |
| `result.info` | `{"hinges": [...], "targets": np.ndarray, "n_faces", "n_hinges", "n_loops"}` |
| `result.initial_centroid` / `result.final_centroid` | 2D paper centroid (mean XY of all non-table geoms) before/after |
| `result.backend` | `"jax"` or `"warp"` |
| `result.extra` | Reserved metadata dict (currently empty) |

`duration=8.0` on the full 16-face reference mesh is very slow on CPU MJX; use the tiny 2-face pattern or a GPU for quick experiments (see [BACKENDS.md](BACKENDS.md)).

Parameters worth knowing: `duration` (8.0 s of simulated time), `fold_duration` (3.0 s smoothstep ramp over which actuator targets go from 0 to full), `vibrate` (True), `amp_x`/`amp_z` (0.0 / 0.0013 m), `freq_hz` (75.0 Hz). See [SIMULATION.md](SIMULATION.md) for the internals.

Runnable version: `python examples/01_evaluate_cpu_jax.py` (CPU) or `python examples/02_evaluate_warp_gpu.py` (CUDA GPU).

## Tutorial 3 — Score a pattern (fitness)

```python
from origami_engine_mjx import OrigamiEngine, load_default_creases

engine = OrigamiEngine(backend="jax")
creases = load_default_creases()

score = engine.evaluate(
    creases,
    duration=8.0,
    trajectory=(1.0, 0.0, 0.0),  # measure displacement along +x
    fold_duration=3.0,
)
print(score)  # forward displacement in meters (float)
```

Ask for the details dict instead of just the float:

```python
from origami_engine_mjx import evaluate_individual

details = evaluate_individual(
    creases,
    duration=8.0,
    trajectory=(1.0, 0.0, 0.0),
    return_details=True,
)
# keys: displacement, initial_centroid, final_centroid,
#       duration, backend, n_faces, n_loops
```

`OrigamiEngine.evaluate(...)` always returns a `float`; use `evaluate_individual(..., return_details=True)` for the dict. `trajectory` accepts a 2- or 3-element direction vector. A 2-vector is used **without normalization** (the score scales with its length); a 3-vector is normalized. `(N, 2)` / `(N, 3)` waypoint arrays are not supported. See [EVALUATION.md](EVALUATION.md#forward-displacement).

## Tutorial 4 — Save and load crease patterns as JSON

```python
import tempfile, pathlib

creases = load_default_creases()
with tempfile.TemporaryDirectory() as tmp:
    path = pathlib.Path(tmp) / "my_pattern.json"
    save_creases_to_json(creases, path, angle_unit="degree")  # human-readable angles
    loaded = load_creases_from_json(path)  # validated against the JSON Schema

    print(len(loaded), loaded[0])  # 24 Crease(p1=(0.2, 0.0), p2=(0.45, 0.25), angle=-0.785...)
```

`normalize_creases` also accepts a file path or a list of dicts directly, so you can pass any of these forms straight into `engine.run` / `engine.evaluate`. The JSON format and schema are documented in [CREASE_FORMAT.md](CREASE_FORMAT.md).

## Tutorial 5 — Define a custom pattern in Python

Creases are 2D segments on the unit sheet (`S = 1.0`). The crease set must tile the square into valid polygons:

```python
import numpy as np

# Convention: negative angles = mountain folds, positive = valley folds
v_fold = np.deg2rad(45)  # valley (positive)
m_fold = np.deg2rad(-45)  # mountain (negative)

creases = [
    Crease((0.0, 0.5), (0.5, 0.5), m_fold),  # mountain at y=0.5
    Crease((0.5, 0.5), (1.0, 0.5), v_fold),  # valley at y=0.5
]
pat = build_pattern(creases)
print(pat.valid, pat.n_faces)  # True 2
```

Always check `pat.valid` before simulating — `engine.run` raises `ValueError("invalid crease pattern: ...")` otherwise. Design rules and common failure reasons: [CREASE_FORMAT.md](CREASE_FORMAT.md).

**Note on mountain/valley convention:** The sign convention is negative=mountain, positive=valley. However, the package constants `M = +π/4` and `V = -π/4` have names swapped vs. their sign (M is a positive angle, V is negative). This is inherited from upstream and **do not change the values** — use `Crease.angle` for the actual fold direction.

## Tutorial 6 — Evaluate a population (e.g. a genetic algorithm)

```python
from origami_engine_mjx import OrigamiEngine, load_default_creases
from origami_engine_mjx.evaluation import evaluate_population

creases = load_default_creases()
population = [creases, creases, creases]

# Both backends evaluate individuals sequentially; there is no process pool.
# Scores are always +x displacement in the current implementation.
scores = evaluate_population(population, duration=4.0, backend="jax", error_policy="raise")
# error_policy="nan" returns float('nan') for failed individuals instead of raising

best = max(zip(scores, population), key=lambda t: t[0])
print("best fitness:", best[0])
```

Results keep the input order. `trajectory` is validated but currently ignored by population scoring (always +x), and `batch_size` has no effect yet — see [EVALUATION.md](EVALUATION.md#current-behaviour-caveats). For a non-+x fitness direction, score individuals with `evaluate_individual(..., trajectory=...)`.

Runnable version: `python examples/03_batch_population.py`.

## Tutorial 7 — Plot a pattern and save the figure

```python
import matplotlib

matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt
from origami_engine_mjx import plot_pattern, load_default_creases, build_pattern

creases = load_default_creases()
pat = build_pattern(creases)
plot_pattern(creases, pat)  # draws; under Agg, show() does not block
plt.gcf().savefig("my_pattern.png", dpi=150, bbox_inches="tight")
```

Style: Mountain = crimson dashed, Valley = royalblue dash-dot, each crease labelled with its fold angle in degrees, face indices in gray, loop edges marked with a black `x`. The repo ships a pre-rendered version at `docs/crease_pattern_plot.png`.

## Tutorial 8 — Get the raw MuJoCo XML

Everything the engine builds is a standard MJCF XML string, so you can inspect or post-process it with plain MuJoCo:

```python
import mujoco
from origami_engine_mjx import build_pattern, generate_xml, load_default_creases

creases = load_default_creases()
pat = build_pattern(creases)
xml, info = generate_xml(pat, creases)

with open("model.xml", "w") as f:
    f.write(xml)

model = mujoco.MjModel.from_xml_string(xml)  # standard MuJoCo API from here on
data = mujoco.MjData(model)
```

The XML structure (bodies, hinges, equality constraints, actuators, table) is documented in [SIMULATION.md](SIMULATION.md).

## See also

- [GETTING_STARTED.md](GETTING_STARTED.md) — installation methods (pip / uv / source)
- [CREASE_FORMAT.md](CREASE_FORMAT.md) — crease & JSON format reference
- [SIMULATION.md](SIMULATION.md) — model generation, stepping loop
- [EVALUATION.md](EVALUATION.md) — fitness functions in depth
- [API_REFERENCE.md](API_REFERENCE.md) — every public function signature
- [OPTIMIZATION_ES.md](OPTIMIZATION_ES.md) — driving the simulation from an external optimizer
- [RL.md](RL.md) — Gymnasium and JAX-functional reinforcement-learning environments
- [examples/README.md](../examples/README.md) — runnable scripts (in `examples/`)