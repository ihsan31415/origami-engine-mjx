# API Reference

> **Documentation map:** [README](../README.md) · [Getting Started](GETTING_STARTED.md) · [Tutorials](TUTORIALS.md) · [Crease Format](CREASE_FORMAT.md) · [Simulation](SIMULATION.md) · [Evaluation](EVALUATION.md) · **API Reference** · [Architecture](ARCHITECTURE.md) · [Troubleshooting](TROUBLESHOOTING.md) · [Glossary](GLOSSARY.md) · [Contributing](CONTRIBUTING.md)

Complete reference for the public API of `origami_engine_mjx` (version `0.1.0`). All names below are importable from the package root:

```python
from origami_engine_mjx import (
    OrigamiEngine,
    Crease,
    build_pattern,
    load_default_creases,
    evaluate_individual,
    evaluate_population,
    compute_forward_displacement,
    generate_xml,
    plot_pattern,
    load_creases_from_json,
    save_creases_to_json,
    normalize_creases,
    DEFAULT_PATTERN_JSON,
    M,
    V,
)
```

## Import paths

| Import style | Example | Notes |
|---|---|---|
| Package root (recommended) | `from origami_engine_mjx import OrigamiEngine, Crease, load_default_creases` | Stable public surface; modules re-exported from `__init__.py` |
| Focused module | `from origami_engine_mjx.geometry import build_pattern` | For code that only needs crease-graph construction |
| Constants / types | `from origami_engine_mjx.constants import S, M, V` or `from origami_engine_mjx.types import Crease, Pattern` | Individual symbols as needed |
| Backend / evaluation | `from origami_engine_mjx.backends import available_backends, get_backend` or `from origami_engine_mjx.evaluation import evaluate_individual` | Optional — backend-aware callers |

## Domain types (`origami_engine_mjx.types`)

### `Crease` — frozen dataclass

```python
Crease(p1: tuple[float, float], p2: tuple[float, float], angle: float)
```

A single crease segment. `p1`/`p2` are 2D points in sheet coordinates (`0..S`, `S=1.0`); `angle` is the signed target fold angle in **radians** — negative = mountain fold, positive = valley fold.

### `Pattern` — dataclass

Returned by `build_pattern`. Fields: `valid: bool`, `reason: str`, `faces: list | None` (Shapely polygons), `edges: list[dict] | None` (each `{"i", "j", "p", "q", "crease"}`), `root: int`, `parent: dict`, `depth: dict`, `tree_edge: dict`, `loop_edges: list[int] | None`, `n_faces: int`, `n_loops: int`.

### `SimulationResult` — dataclass

Returned by `OrigamiEngine.run()`. Fields: `model` (backend-specific model handle), `data` (backend-specific data), `pattern: Pattern`, `info: dict` (`hinges`, `targets`, `n_faces`, `n_hinges`, `n_loops`), `initial_centroid: np.ndarray` (2,), `final_centroid: np.ndarray` (2,), `backend: str` — `"jax"` or `"warp"`.

### `PopulationResult` — dataclass

Returned by `OrigamiEngine.evaluate_population()`. Fields: `scores: np.ndarray`, `backend: str`, `n_individuals: int`, `duration: float`, `extra: dict[str, Any]`.

## Pattern construction & I/O

### `build_pattern(creases) -> Pattern`

(`origami_engine_mjx.geometry`) Reconstruct polygonal faces and the spanning-tree hinge graph from a crease list. `creases` may be a `Crease` list, dict list, or JSON path (normalized internally via `normalize_creases`); see that function. Pure Shapely — no MuJoCo needed. Validation reasons: `"no faces"`, `"sliver face (area …)"`, `"faces do not tile the sheet"`, `"disconnected faces"`.

### `normalize_creases(creases) -> list[Crease]`

(`origami_engine_mjx.serialization`) Coerce a JSON path, list of `Crease`, or list of dicts into `list[Crease]`. Raises `ValueError` (`None`, empty, missing keys, bad item type) / `TypeError` (non-list input).

### `load_creases_from_json(path) -> list[Crease]`

Load and **schema-validate** a crease JSON file (both forms documented in [CREASE_FORMAT.md](CREASE_FORMAT.md)). The file may be either a bare list of crease objects or an object with a ``creases`` key and an optional ``angle_unit`` (``radian`` or ``degree``). Payloads are validated against the packaged JSON Schema; a `ValueError` is raised on schema violation or missing keys (message includes the crease index). Angles are converted from degrees via `np.deg2rad` when `angle_unit` = `"degree"`; default is radians.

### `save_creases_to_json(creases, path, angle_unit="radian") -> Path`

Serialize creases to JSON for reproducible pipeline inputs. `angle_unit` must be `"radian"` or `"degree"` (else `ValueError`). Angles are converted to the requested unit. Writes Form A with 2-space indentation, creating parent directories as needed, and returns the output `Path`.

### `CREASE_SCHEMA`

The packaged JSON Schema (dict) loaded from `origami_engine_mjx/schemas/crease_pattern.schema.json` (shipped as package data). Use with `jsonschema.validate(payload, CREASE_SCHEMA)` in your own tooling.

## Reference patterns (`origami_engine_mjx.patterns`)

| Export | Description |
|---|---|
| `DEFAULT_PATTERN_JSON` | `Path` to the packaged `origami_engine_mjx/crease_patterns/reference_miura_4x4.json` (shipped in wheel + sdist; always present in every install method). |
| `reference_miura_legacy_creases() -> list[Crease]` | Legacy 4×4 Miura-ori crease list (24 creases, 16 faces, 9 loop edges). |
| `reference_miura_render_creases() -> list[Crease]` | Render-tuned 4×4 Miura-ori crease list (same 24-crease / 16-face / 9-loop topology). |
| `load_default_creases() -> list[Crease]` | Load `DEFAULT_PATTERN_JSON` (24 creases, 16 faces, 9 loops). Falls back to `reference_miura_render_creases()` if the file is ever missing. |

## Model generation (`origami_engine_mjx.model`)

### `generate_xml(pat, creases, name="origami_free", coll_thick=None) -> tuple[str, dict]`

Build a MuJoCo XML string for a validated `Pattern` (see [SIMULATION.md](SIMULATION.md) for the full structure). Parameters: `pat` — `Pattern` from `build_pattern`; `creases` — normalized creases (target angles are looked up by edge index); `name` — model name attribute; `coll_thick` — `None` for single-geom mode, or a thickness value enabling separate visual/collision geoms. Returns `(xml, info)` where `info = {"hinges": [{"name", "face", "edge", "target"}...], "targets": np.ndarray, "n_faces", "n_hinges", "n_loops"}`. Pure XML generation — **no MuJoCo runtime is required**; the MJCF string can be consumed by classic `MjModel.from_xml_string`, MJX-JAX `mjx.put_model`, or MJX-Warp `mjw.make_model`.

## Simulation (`origami_engine_mjx.engine`)

### `OrigamiEngine(backend="jax"|"warp"|"auto")`

Facade for running simulations and evaluating fitness with pluggable MJX backends.

**Parameters:**
- `backend`: `{"jax", "warp", "auto"}`
  - `"jax"` — MJX-JAX (CPU/GPU/TPU through JAX; differentiable; runs eager by default; `jit=True` supported for long steady-state runs after warm-up).
  - `"warp"` — MJX-Warp (NVIDIA CUDA GPU; high-throughput batched; requires CUDA-capable GPU).
  - `"auto"` — pick the first available backend (warp preferred if CUDA GPU present, otherwise jax). Raises `RuntimeError` if no MJX backend is available.

**Methods:**

- `.run(creases, duration=8.0, fold_duration=3.0, vibrate=True, amp_x=0.0, amp_z=0.0013, freq_hz=75.0, coll_thick=None)` → `SimulationResult`
  - Run a single origami fold simulation. For the JAX backend, runs the simulation loop with eager `mjx.step` by default (JIT compilation of `mjx.step` on mesh-contact models is prohibitively slow on CPU); set `jit=True` in the `run()` kwargs to enable JIT if desired (Warp backend has no `jit` param; GPU required).
  - **JIT note**: The JAX backend defaults to `jit=False` because JIT-compiling `mjx.step` on mesh-contact models is prohibitively slow on CPU. See [BACKENDS.md](BACKENDS.md) for performance details.

- `.evaluate(creases, duration=8.0, trajectory=(1.0, 0.0, 0.0), fold_duration=3.0, vibrate=True, amp_x=0.0, amp_z=0.0013, freq_hz=75.0) -> float`
  - Evaluate one crease configuration; return forward displacement (projected onto `trajectory`).

- `.evaluate_population(population, duration=8.0, trajectory=(1.0, 0.0, 0.0), fold_duration=3.0, vibrate=True, amp_x=0.0, amp_z=0.0013, freq_hz=75.0, batch_size=None) -> PopulationResult`
  - Evaluate a population of crease patterns in batch. `batch_size` controls per-individual sub-batching in the Warp backend; in the JAX backend each individual is simulated individually but JIT-compiled.

## Evaluation (`origami_engine_mjx.evaluation`)

### `evaluate_individual(creases, duration=8.0, trajectory=(1.0, 0.0, 0.0), fold_duration=3.0, vibrate=True, amp_x=0.0, amp_z=0.0013, freq_hz=75.0, backend="jax", return_details=False) -> float | dict`

Run one simulation and return net forward displacement. With `return_details=True` returns a dict: `displacement`, `initial_centroid`, `final_centroid`, `duration`, `trajectory`, `n_faces`, `n_loops`, `backend`.

**Raises:**
- `ValueError` for `duration <= 0`, invalid crease input, invalid pattern, bad trajectory.
- `RuntimeError` for non-finite score (backend instability).

### `evaluate_population(individuals, duration=8.0, trajectory=(1.0, 0.0, 0.0), fold_duration=3.0, vibrate=True, amp_x=0.0, amp_z=0.0013, freq_hz=75.0, backend="jax", batch_size=None, error_policy="raise") -> list[float]`

Score a list of crease patterns using the given backend. `backend="jax"` or `"warp"`.

**`error_policy`:**
- `"raise"`: a failing individual raises `RuntimeError("individual {i} produced non-finite score: ...")`.
- `"nan"`: the slot becomes `float("nan")` and evaluation continues.

Results preserve input order in both modes.

**`workers` note**: The MJX backends do not use a `workers` parameter; batching is handled natively (per-individual loop in JAX; native Warp GPU batched execution when `batch_size` is set, with fallback per-individual loop on GPU if topology differs).

## Visualization (`origami_engine_mjx.plotting`)

### `plot_pattern(creases: list[Crease], pat: Pattern | None = None) -> None`

Matplotlib figure of the flat pattern: sheet rectangle, Mountain folds (angle < 0): crimson dashed lines, Valley folds (angle > 0): royal blue dash-dot lines, each crease labelled with its fold angle in degrees at its midpoint, face indices drawn in gray at each polygon's representative point, loop edges marked with black "x", legend at lower centre, axes limited to `[-0.05S, 1.05S]` × `[-0.05S, 1.05S]` with equal aspect, title `"{N} creases | {M} faces | {K} loop edges (x)"` when `pat` is valid, otherwise `"{N} creases [ | INVALID: reason]"`.

**Notes:**
- The visual style matches the reference plot: Mountain → crimson dashed, Valley → royalblue dash-dot.
- Crease labels → actual fold angle in degrees at each midpoint.
- Face numbers → gray at each polygon's representative point.
- Loop edges → black "x" at each edge midpoint.
- Under the Agg backend (`matplotlib.use("Agg")`), `plt.show()` does not block; call `plt.gcf()` and save with `plt.savefig(...)`.

## Constants (`origami_engine_mjx.constants`)

| Constant | Value | Meaning |
|---|---|---|
| `S` | `1.0` | Sheet side length |
| `THICK` | `1e-4` | Paper thickness |
| `PAPER_MASS` | `0.2` | Sheet mass |
| `Z0` | `1.1e-3` | Initial paper z-offset |
| `M` | `+π/4` | +45° canonical valley angle (note: `Crease.angle` sign convention is negative=mountain, positive=valley) |
| `V` | `−π/4` | −45° canonical mountain angle |
| `MAX_FOLD` | `0.98π` | Hinge target clip |
| `KP` / `KV` | `2.0` / `0.15` | Hinge position-actuator gains |
| `DAMPING` | `0.003` | Hinge joint damping |
| `FORCE_MAX` | `2.0` | Actuator force limit |
| `MU_PAPER` / `MU_TABLE` | `0.3` / `0.5` | Friction coefficients |
| `VIB_FREQ` / `VIB_AMP_X` / `VIB_AMP_Z` | `75.0` / `0.0` / `0.0013` | Table vibration defaults |
| `EQ_SOLREF` / `EQ_SOLIMP` | `"0.005 1"` / `"0.95 0.99 0.001"` | Equality constraint solver params |
| `STRETCH_K` / `STRETCH_DAMP` / `STRETCH_MAX` | `5e4` / `50` / `1e-4` | Edge stretch spring params |
| `K_PANEL` / `D_PANEL` | `1e5` / `50` | Panel stiffness/damping |
| `MAX_STRAIN` / `COLL_THICK` | `0.005` / `1e-5` | Strain cap / thin collision thickness |

## Complete workflow example

```python
from origami_engine_mjx import (
    OrigamiEngine,
    Crease,
    build_pattern,
    load_default_creases,
    plot_pattern,
)

# 1. Load the reference Miura-ori pattern (24 creases)
creases = load_default_creases()

# 2. Reconstruct and validate the face graph
pat = build_pattern(creases)
assert pat.valid, pat.reason

# 3. Visualize the flat crease pattern
plot_pattern(creases, pat)

# 4. Run the MJX-JAX simulation (eager, CPU default; use GPU for speed)
engine = OrigamiEngine(backend="jax")
result = engine.run(creases, duration=2.0, vibrate=False)

# 5. Score it: forward displacement along +x
score = engine.evaluate(creases, trajectory=(1.0, 0.0, 0.0))
print(f"forward displacement: {score:.6f} m  (backend={result.backend})")

# 6. Plot again with results
plot_pattern(creases, pat)
```

## See also

- [GETTING_STARTED.md](GETTING_STARTED.md) — installation (pip / uv / source) and first run
- [TUTORIALS.md](TUTORIALS.md) — nine step-by-step recipes
- [CREASE_FORMAT.md](CREASE_FORMAT.md) — crease & JSON schema reference
- [SIMULATION.md](SIMULATION.md) — XML model structure, stepping loop, rendering
- [EVALUATION.md](EVALUATION.md) — fitness functions and population evaluation
- [ARCHITECTURE.md](ARCHITECTURE.md) — module layout and data flow
- [BACKENDS.md](BACKENDS.md) — MJX-JAX vs MJX-Warp: semantics, features, limitations
- [COLAB_GPU.md](COLAB_GPU.md) — Colab T4 setup, validation checklist, benchmarking
- [MIGRATION.md](MIGRATION.md) — migrating from `origami-engine` (classic MuJoCo)