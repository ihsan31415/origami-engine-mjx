# API Reference

> **Documentation map:** [README](../README.md) · [Getting Started](GETTING_STARTED.md) · [Tutorials](TUTORIALS.md) · [Crease Format](CREASE_FORMAT.md) · [Simulation](SIMULATION.md) · [Evaluation](EVALUATION.md) · **API Reference** · [Architecture](ARCHITECTURE.md) · [Troubleshooting](TROUBLESHOOTING.md) · [Glossary](GLOSSARY.md) · [Integrations](INTEGRATIONS.md) · [Optimization (ES)](OPTIMIZATION_ES.md) · [Search space](SEARCH_SPACE.md) · [Populations](POPULATIONS.md) · [RL](RL.md) · [Contributing](CONTRIBUTING.md)

Complete reference for the public API of `origami_engine_mjx` (version `1.0.0`). Sections follow the **Core modules** order in the [README](../README.md#core-modules-brief). Every name below is verified against the installed package.

```python
from origami_engine_mjx import (
    OrigamiEngine,
    Crease,
    Pattern,
    SimulationResult,
    PopulationResult,
    BackendName,
    build_pattern,
    generate_xml,
    load_default_creases,
    reference_miura_legacy_creases,
    reference_miura_render_creases,
    DEFAULT_PATTERN_JSON,
    evaluate_individual,
    evaluate_population,
    compute_forward_displacement,
    plot_pattern,
    load_creases_from_json,
    save_creases_to_json,
    normalize_creases,
    CREASE_SCHEMA,
    S,
    M,
    V,
    THICK,
    PAPER_MASS,
    Z0,
    MAX_FOLD,
    VIB_FREQ,
    VIB_AMP_X,
    VIB_AMP_Z,
)
```

## Import paths

| Import style | Example | Notes |
|---|---|---|
| Package root (recommended) | `from origami_engine_mjx import OrigamiEngine, Crease, load_default_creases` | Stable public surface re-exported from `__init__.py` (`__all__` above). |
| Focused shared module | `from origami_engine_mjx.geometry import build_pattern` | Geometry/I/O/model modules need no JAX or Warp runtime. |
| Constants / types | `from origami_engine_mjx.constants import S, M, V, MIN_FACE_AREA` or `from origami_engine_mjx.types import Crease, Pattern` | The package root exports only a subset of constants; import the module for the rest. |
| Backends | `from origami_engine_mjx.backends import available_backends, get_backend` | Registry + `SimulationBackend` interface. |
| Import guards | `from origami_engine_mjx.backend import require_mjx_jax, mujoco` | Optional-dependency access without importing a backend class. |

**Backend applicability:** the shared modules (`constants`, `types`, `geometry`, `model`, `serialization`, `patterns`, `plotting`, `backend`, `engine` facade, `evaluation` wrappers) are backend-agnostic. Only `origami_engine_mjx.backends.jax_backend` / `warp_backend` touch MJX-JAX / MJX-Warp, and only the Warp backend needs a CUDA GPU.

## `origami_engine_mjx.constants` — shared

All values are module-level Python/NumPy scalars; there are no functions.

| Constant | Runtime value | Meaning |
|---|---|---|
| `S` | `1.0` | Sheet side length (the sheet is `[0, S]²`, i.e. 1 m × 1 m). |
| `THICK` | `0.0001` | Paper half-thickness used for the baseline prism geoms (m). |
| `PAPER_MASS` | `0.2` | Total sheet mass (kg); prism density is derived from it. |
| `Z0` | `THICK + 0.001` = `0.0011` | Initial root-body z-offset above the table (m). |
| `MIN_FACE_AREA` | `1e-4 * S * S` = `1e-4` | Smallest accepted face area; smaller faces fail validation as slivers. |
| `MAX_FOLD` | `0.98π` ≈ `3.0788` rad (176.4°) | Clip applied to every hinge target. |
| `KP` / `KV` | `2.0` / `0.15` | Position-actuator stiffness / damping for hinge targets. |
| `DAMPING` | `0.003` | Hinge joint damping. |
| `FORCE_MAX` | `2.0` | Position-actuator force limit (N). |
| `MU_PAPER` / `MU_TABLE` | `0.3` / `0.5` | Sliding friction coefficients for paper / table. |
| `VIB_FREQ` | `75.0` | Table vibration frequency (Hz). |
| `VIB_AMP_X` / `VIB_AMP_Z` | `0.0` / `0.0013` | Table vibration amplitudes (m). |
| `EQ_SOLREF` / `EQ_SOLIMP` | `"0.005 1"` / `"0.95 0.99 0.001"` | Solver parameters for loop-edge `<connect>` equality constraints. |
| `STRETCH_K` / `STRETCH_DAMP` / `STRETCH_MAX` | `5.0e4` / `50.0` / `1.0e-4` | Edge-stretch spring stiffness / damping / slide range. |
| `K_PANEL` / `D_PANEL` | `1e5` / `50.0` | Panel stiffness / damping constants. |
| `MAX_STRAIN` | `0.005` | Strain cap used by the model generator. |
| `COLL_THICK` | `THICK * 0.1` = `1e-5` | Thin collision thickness. |
| `M` | `+π/4` (≈ 0.7854) | Canonical +45° angle; **name is swapped** relative to the runtime sign convention (see below). |
| `V` | `−π/4` (≈ −0.7854) | Canonical −45° angle; **name is swapped** relative to the runtime sign convention. |

**Sign convention:** at runtime on `Crease.angle`, **negative = mountain, positive = valley**. The constants `M = +π/4` and `V = −π/4` are inherited from upstream with their names swapped; do not reinterpret them as the runtime convention. See [CREASE_FORMAT.md](CREASE_FORMAT.md#the-crease-type).

## `origami_engine_mjx.types` — shared

### `BackendName`

```python
BackendName = Literal["jax", "warp"]
```

Type alias for backend identifiers accepted by `evaluate_individual` / `evaluate_population`.

### `Crease` — frozen dataclass

```python
Crease(p1: tuple[float, float], p2: tuple[float, float], angle: float)
```

A single crease segment. `p1`/`p2` are 2D sheet coordinates (`0..S`, `S = 1.0`); `angle` is the signed target fold angle in **radians** — negative = mountain, positive = valley. Frozen, therefore immutable and hashable.

### `Pattern` — dataclass

Returned by `build_pattern`. Defaults are shown in parentheses.

| Field | Type (default) | Meaning |
|---|---|---|
| `valid` | `bool` | Whether reconstruction and validation passed. |
| `reason` | `str` (`""`) | Failure reason when `valid` is `False`. |
| `faces` | `list | None` (`None`) | Shapely polygons, one per panel. |
| `edges` | `list[dict] | None` (`None`) | Face-adjacency edges, each `{"i", "j", "p", "q", "crease"}`. |
| `root` | `int` (`0`) | Index of the largest-area face (root body). |
| `parent` | `dict | None` (`None`) | Face index → parent face index (`None` for the root). |
| `depth` | `dict | None` (`None`) | Face index → spanning-tree depth. |
| `tree_edge` | `dict | None` (`None`) | Face index → edge index connecting it to its parent. |
| `loop_edges` | `list[int] | None` (`None`) | Edge indices not in the spanning tree. |
| `n_faces` | `int` (`0`) | Number of faces. |
| `n_loops` | `int` (`0`) | Number of loop edges. |

### `SimulationResult` — dataclass

Returned by `OrigamiEngine.run()` / backend `run()`.

| Field | Type (default) | Meaning |
|---|---|---|
| `model` | `Any` | Backend bundle: `{"mjx_model", "classic_model", "info"}` for JAX or `{"warp_model", "classic_model", "info"}` for Warp. |
| `data` | `Any` | Final backend state (`mjx.Data` / Warp data). |
| `pattern` | `Pattern` | Validated pattern that produced the model. |
| `info` | `dict[str, Any]` | Metadata from `generate_xml` (`hinges`, `targets`, `n_faces`, `n_hinges`, `n_loops`) plus `table_qposadr` (`table_x`/`table_z` positions in `qpos`). |
| `initial_centroid` | `np.ndarray` (shape `(2,)`) | Mean XY of paper geoms before stepping. |
| `final_centroid` | `np.ndarray` (shape `(2,)`) | Mean XY of paper geoms after stepping. |
| `backend` | `str` (`"jax"`) | `"jax"` or `"warp"`. |
| `extra` | `dict[str, Any]` (`{}`) | With `run(record=True)`: `{"trajectory": {"time", "qpos", "qvel", "centroid", "hinge_qposadr"}}` (see [SIMULATION.md](SIMULATION.md#recording-a-trajectory-recordtrue). |

### `PopulationResult` — dataclass

Returned by `OrigamiEngine.evaluate_population()`.

| Field | Type (default) | Meaning |
|---|---|---|
| `scores` | `np.ndarray` | Forward-displacement score per individual, **always along +x** in the current backends (see the note under `evaluate_population`). |
| `backend` | `str` | `"jax"` or `"warp"`. |
| `n_individuals` | `int` | `len(population)`. |
| `duration` | `float` | Simulated duration per individual (s). |
| `extra` | `dict[str, Any]` (`{}`) | With `run(record=True)`: `{"trajectory": {"time", "qpos", "qvel", "centroid", "hinge_qposadr"}}` (see [SIMULATION.md](SIMULATION.md#recording-a-trajectory-recordtrue). |

## `origami_engine_mjx.geometry` — shared

### `build_pattern(creases) -> Pattern`

```python
from origami_engine_mjx import build_pattern, normalize_creases

pat = build_pattern(normalize_creases(raw_input))
```

Reconstruct polygonal faces and the spanning-tree hinge graph from a sequence of **`Crease` objects**. This function does *not* call `normalize_creases`: each element must expose `.p1`/`.p2` (so dicts and JSON paths are rejected — dicts raise `AttributeError`). Pure Shapely; no MuJoCo/JAX/Warp needed.

**Parameters:** `creases` — any iterable of `Crease` (normally a list). No unit/angle interpretation happens here: geometry uses endpoints only.

**Returns:** a `Pattern`. Rejection reasons (with `valid=False`):

| `reason` | Cause |
|---|---|
| `"no faces"` | The creases plus sheet boundary polygonize to nothing. |
| `"sliver face (area {min:.2e})"` | A face is smaller than `MIN_FACE_AREA` (`1e-4`). |
| `"faces do not tile the sheet"` | Face areas do not sum to `S²` within `1e-6`. |
| `"disconnected faces"` | The face-adjacency graph is not connected. |

**Raises:** `AttributeError` for non-`Crease` elements (e.g. dicts); Shapely exceptions for malformed geometry.

## `origami_engine_mjx.model` — shared

### `generate_xml(pat, creases, name="origami_free", coll_thick=None) -> tuple[str, dict]`

Build a MuJoCo MJCF XML string from a validated `Pattern`. No MuJoCo runtime is required.

**Parameters:**

- `pat` — `Pattern` returned by `build_pattern`; must be valid and its `faces`/`edges` intact.
- `creases` — normalized crease sequence **in the same order** used to build `pat`. Target angles are looked up by `Pattern.edges[...]["crease"]` index.
- `name` — value of the `<mujoco model="...">` attribute.
- `coll_thick` — `None` (default) emits one prism mesh geom per triangle with contact/friction. A float switches to separate visual (non-colliding) and thin collision prisms of that thickness.

**Returns:** `(xml, info)` where `info` is:

| Key | Value |
|---|---|
| `hinges` | `list[{"name", "face", "edge", "target"}]`; `target = clip(sgn * crease.angle, -MAX_FOLD, MAX_FOLD)`. |
| `targets` | `np.ndarray` of hinge targets, same order as `hinges`. |
| `n_faces` | Number of faces. |
| `n_hinges` | Number of tree (hinge) edges. |
| `n_loops` | Number of loop edges closed by equality constraints. |

The resulting XML is accepted by classic `mujoco.MjModel.from_xml_string`, MJX-JAX `mjx.put_model`, or MJX-Warp `mjw.make_model`. See [SIMULATION.md](SIMULATION.md).

## `origami_engine_mjx.serialization` — shared

### `CREASE_SCHEMA`

The packaged JSON Schema (Python `dict`) loaded from `origami_engine_mjx/schemas/crease_pattern.schema.json` (JSON Schema draft 2020-12). Validate your own payloads with `jsonschema.validate(payload, CREASE_SCHEMA)`. See [CREASE_FORMAT.md](CREASE_FORMAT.md#json-file-format).

### `normalize_creases(creases) -> list[Crease]`

Coerce supported inputs into a fresh `list[Crease]`.

| Accepted input | Behavior |
|---|---|
| `str` / `pathlib.Path` | Delegates to `load_creases_from_json` (schema-validated, unit converted). |
| `list` / `tuple` of `Crease` | Returned as a new list (same objects). |
| `list` / `tuple` of dicts | Requires `p1`, `p2`, `angle` keys; `angle` is cast with `float()` (assumed radians). |

**Raises:**

- `ValueError("creases must not be None.")` for `None`.
- `ValueError("creases must contain at least one crease.")` for an empty sequence.
- `TypeError("creases must be a list or tuple of Crease objects.")` for any other type.
- `ValueError(f"crease at index {idx} is missing required keys: {exc}")` for an incomplete dict.
- `ValueError(f"crease at index {idx} is not a valid Crease object or dict.")` for an unsupported element.

### `load_creases_from_json(path) -> list[Crease]`

Read a crease JSON file, validate it against `CREASE_SCHEMA`, and return `list[Crease]`.

**Parameters:** `path` — `str` or `Path`. **Returns:** creases with angles in radians.

The file may be a bare list (radians) or an object with `creases` and optional `angle_unit`. The schema enum requires a lowercase unit; the loader strips and lower-cases the parsed value before checking it. Values `"degree"`, `"degrees"`, `"deg"` convert via `np.deg2rad`, everything else (including `"radian"`, `"radians"`, `"rad"`, or an omitted key) is treated as radians.

**Raises:** `ValueError("invalid crease JSON: ...")` on schema violations; `ValueError(f"crease entry at index {idx} missing key: {exc}")` on missing `p1`/`p2`/`angle`; `json.JSONDecodeError` / `OSError` from file I/O.

### `save_creases_to_json(creases, path, angle_unit="radian") -> Path`

Normalize and serialize creases as Form A: `{"angle_unit": <unit>, "creases": [...]}` with 2-space indentation.

**Parameters:** `creases` — anything accepted by `normalize_creases`; `path` — output path (parent directories are created); `angle_unit` — `"radian"` or `"degree"` after `strip().lower()`; angles are converted with `np.rad2deg` for degrees.

**Returns:** the output `Path`. **Raises:** `ValueError("angle_unit must be 'radian' or 'degree'.")` for other units; `normalize_creases` errors.

## `origami_engine_mjx.patterns` — shared

| Export | Description |
|---|---|
| `DEFAULT_PATTERN_JSON` | `Path` to the packaged `origami_engine_mjx/crease_patterns/reference_miura_4x4.json`; falls back to the repo-root `crease_patterns/reference_miura_4x4.json` only if the packaged copy is missing. |
| `reference_miura_legacy_creases() -> list[Crease]` | Embedded legacy 4×4 Miura-ori crease list (24 creases; 16 faces, 9 loop edges). |
| `reference_miura_render_creases() -> list[Crease]` | Embedded render-tuned 4×4 Miura-ori crease list (same 24-crease / 16-face / 9-loop topology; first crease `(0.2, 0.0) → (0.45, 0.25)`). |
| `load_default_creases() -> list[Crease]` | Load `DEFAULT_PATTERN_JSON` via `load_creases_from_json`; falls back to `reference_miura_render_creases()` if the path disappears. |

All are backend-agnostic. Both embedded builders use the swapped `M`/`V` constants described above.

## `origami_engine_mjx.plotting` — shared

### `plot_pattern(creases: list[Crease], pat: Pattern | None = None) -> None`

Draw a flat crease-pattern figure with Matplotlib and call `plt.show()`.

**Parameters:** `creases` — list of `Crease` (not normalized internally); `pat` — optional validated `Pattern` used to add face indices, loop-edge markers, and counts in the title.

**Returns:** `None`. Under a non-interactive backend (e.g. Agg) `show()` does not block, so you can save `plt.gcf()` afterwards.

**Style:** sheet rectangle `(0,0)-(S,S)`; mountain (`angle < 0`) crimson dashed; valley (`angle > 0`) royalblue dash-dot; each crease labelled `abs(angle)` in degrees at its midpoint; face indices in gray at representative points; loop edges as black `x`; legend lower-centre; equal aspect with axes `[-0.05S, 1.05S]` on both axes; title `"{N} creases"`, extended with `" | {M} faces | {K} loop edges (x)"` when `pat.valid`, or `" | INVALID: {reason}"` when `pat` is provided but invalid.

## `origami_engine_mjx.backend` — import guards

Module-level optional imports plus helpers. Nothing here is backend-specific logic; it exists so geometry/I/O can be imported without JAX/Warp.

| Export | Type | Meaning |
|---|---|---|
| `mujoco` | module or `None` | Classic MuJoCo if import succeeded. |
| `mjx_jax` | module or `None` | `mujoco.mjx` if `jax` + `mujoco-mjx` imported. |
| `mjx_warp` | module or `None` | `mujoco.warp` if installed. |
| `MUJOCO_IMPORT_ERROR` / `MJX_JAX_IMPORT_ERROR` / `MJX_WARP_IMPORT_ERROR` | `BaseException | None` | Captured import failure for each optional dependency. |
| `require_mujoco(message="MuJoCo is required for this operation.")` | `mujoco` | Return the module or raise `RuntimeError(message)` chained from the import error. |
| `require_mjx_jax(message="MJX-JAX backend requires 'mujoco-mjx' and 'jax'.")` | `mujoco.mjx` | Return the module or raise `RuntimeError` with install hint `pip install 'origami-engine-mjx[jax]'`. |
| `require_mjx_warp(message="MJX-Warp backend requires 'mujoco-warp' and an NVIDIA GPU.")` | `mujoco.warp` | Return the module or raise `RuntimeError` with install hint `pip install 'origami-engine-mjx[warp]'`. |

Note: `origami_engine_mjx/backends/__init__.py` imports the two backend modules when the registry is built. Those imports are failure-guarded (errors are stored, not raised), so the import itself never fails when JAX/Warp are absent.

## `origami_engine_mjx.engine`

### `OrigamiEngine(backend="jax")`

Facade that resolves a backend through the registry and forwards simulation calls.

**Parameters:** `backend` — `"jax"`, `"warp"`, or `"auto"`.

- `"jax"` — MJX-JAX (CPU/GPU/TPU through JAX; differentiable; eager stepping by default).
- `"warp"` — MJX-Warp (NVIDIA CUDA GPU; requires `mujoco-warp` + `warp-lang`).
- `"auto"` — uses `available_backends()`; picks `"warp"` if available, else the first available name. Raises `RuntimeError("No MJX backend available. ...")` when none is available.

**Raises:** `ValueError` for an unknown name (from `get_backend`); `RuntimeError` when the selected backend's dependencies are missing.

**Attributes / methods:**

| Member | Signature | Behavior |
|---|---|---|
| `.backend` | property → `SimulationBackend` | The resolved backend instance. |
| `.backend_name` | property → `str` | `"jax"` or `"warp"`. |
| `.device_info()` | `() -> str` | Backend-provided device description (e.g. `"jax backend=cpu, devices=[...]"`). |
| `.run(...)` | see below | Single simulation → `SimulationResult`. |
| `.evaluate(...)` | see below | Single score → `float`. |
| `.evaluate_population(...)` | see below | Population scores → `PopulationResult`. |

#### `OrigamiEngine.run(creases, duration=8.0, fold_duration=3.0, vibrate=True, amp_x=0.0, amp_z=0.0013, freq_hz=75.0, coll_thick=None, record=False, record_every=1) -> SimulationResult`

Runs one fold simulation on the selected backend.

- `creases` — anything `normalize_creases` accepts (list/tuple of `Crease` or dicts, or a JSON path).
- `duration` — simulated seconds; must be `> 0` (`ValueError("duration must be positive.")`).
- `fold_duration` — ramp time for actuator targets; the target scale at time `t` is smoothstep `min(t/fold_duration, 1)²·(3 − 2·min(t/fold_duration, 1))`.
- `vibrate`, `amp_x`, `amp_z`, `freq_hz` — table vibration (`qpos = amp·sin(2πft)`); defaults `VIB_AMP_X`, `VIB_AMP_Z`, `VIB_FREQ`.
- `coll_thick` — forwarded to `generate_xml` (see above).
- `record`, `record_every` — store every `record_every`-th frame (plus the last) in `result.extra["trajectory"]`; physics and score are unchanged.

**Note:** the facade `run` has **no `jit` parameter**. Only `JaxBackend.run(..., jit=False)` exposes JIT control.

**Raises:** `ValueError` (duration, crease input, invalid pattern), `FloatingPointError("unstable simulation: non-finite joint coordinates produced.")`, `RuntimeError` (backend dependencies unavailable).

#### `OrigamiEngine.evaluate(creases, duration=8.0, trajectory=(1.0, 0.0, 0.0), fold_duration=3.0, vibrate=True, amp_x=0.0, amp_z=0.0013, freq_hz=75.0, return_trajectory=False) -> float | tuple[float, dict]`

Calls `run(...)` (without `coll_thick`) and returns `compute_forward_displacement(result, trajectory)`. With `return_trajectory=True` returns `(score, trajectory)` from a single simulation.

#### `OrigamiEngine.evaluate_population(population, duration=8.0, trajectory=(1.0, 0.0, 0.0), fold_duration=3.0, vibrate=True, amp_x=0.0, amp_z=0.0013, freq_hz=75.0, batch_size=None) -> PopulationResult`

Validates `trajectory`: it must be convertable to a float array and have a non-zero norm (`ValueError("trajectory direction must be non-zero.")`; a 2-vector is promoted to 3D). It then calls `backend.run_batch(...)`.

> **Current behavior (verified):** the validated `trajectory` is **not passed to the backends**; both built-in `run_batch` implementations score `final_centroid[0] - initial_centroid[0]` (always +x). `batch_size` is accepted and forwarded but **ignored** by both backends. See [EVALUATION.md](EVALUATION.md#current-behaviour-caveats) and the final report.

Returns a `PopulationResult` with `n_individuals=len(population)` and `extra={}`.

### `compute_forward_displacement(sim_result, trajectory=(1.0, 0.0, 0.0)) -> float`

Projects centroid displacement onto `trajectory`.

```python
delta = sim_result.final_centroid - sim_result.initial_centroid
```

- **2-element `trajectory`:** returns `dot(delta[:2], trajectory)` **without normalizing**, so the score is scaled by the vector's length.
- **3-element `trajectory`:** builds `delta_3d = (delta_x, delta_y, 0)` and returns `dot(delta_3d, trajectory / ‖trajectory‖)`. A zero 3-vector produces `nan` (NumPy divide warning) rather than raising.

`sim_result` may be any object with `initial_centroid` and `final_centroid` attributes (normally a `SimulationResult`). No shape validation beyond NumPy broadcasting; `(N, 2)` waypoint arrays are **not** supported and raise a NumPy shape error.

## `origami_engine_mjx.evaluation`

These wrappers construct a fresh `OrigamiEngine` per call and use only the `run`/`evaluate_population` facade (they do not pass `coll_thick` or `jit`).

### `evaluate_individual(creases, duration=8.0, trajectory=(1.0, 0.0, 0.0), fold_duration=3.0, vibrate=True, amp_x=0.0, amp_z=0.0013, freq_hz=75.0, backend="jax", return_details=False) -> float | dict`

Simulate one pattern and return `compute_forward_displacement(result, trajectory)` as a `float`.

With `return_details=True`, returns exactly these keys:

| Key | Value |
|---|---|
| `displacement` | `float` score. |
| `initial_centroid` | `np.ndarray` shape `(2,)`. |
| `final_centroid` | `np.ndarray` shape `(2,)`. |
| `duration` | `float(duration)` as passed. |
| `backend` | `result.backend` (`"jax"` / `"warp"`). |
| `n_faces` | `int(result.pattern.n_faces)`. |
| `n_loops` | `len(result.pattern.loop_edges or [])`. |

There is **no `trajectory` key** and no finite-score check: a non-finite displacement is returned as-is.

**Raises:** `ValueError` (duration ≤ 0, bad crease input, invalid pattern), `FloatingPointError` (unstable simulation), `RuntimeError` (backend unavailable).

### `evaluate_population(individuals, duration=8.0, trajectory=(1.0, 0.0, 0.0), fold_duration=3.0, vibrate=True, amp_x=0.0, amp_z=0.0013, freq_hz=75.0, backend="jax", batch_size=None, error_policy="raise") -> list[float]`

Calls `OrigamiEngine(backend).evaluate_population(...)` and returns `list(scores)` in input order.

**`error_policy`:**

- `"raise"` (default): after the batch, any non-finite score raises `RuntimeError("individual {i} produced non-finite score: {s}")`.
- Any other value: the raw scores are returned unchanged. Because both backends catch per-individual exceptions and store `nan`, this behaves as a "nan" policy.

**Current behavior (verified):** as with the facade, `trajectory` is validated but ignored by the backends; scores are always +x displacement. `batch_size` is forwarded but ignored. There is no `workers` parameter: both backends use sequential per-individual loops. See [EVALUATION.md](EVALUATION.md#current-behaviour-caveats).

## `origami_engine_mjx.backends`

### `SimulationBackend` (ABC)

Common interface defined in `origami_engine_mjx/backends/base.py`.

| Member | Signature | Notes |
|---|---|---|
| `name` | class attribute `str` | `"abstract"` in the base class. |
| `build_model` | `(xml: str, info: dict) -> Any` | Compile MJCF to a backend-native model/bundle. |
| `make_state` | `(model: Any) -> Any` | Create the initial backend state. |
| `run` | `(creases, duration=8.0, fold_duration=3.0, vibrate=True, amp_x=0.0, amp_z=0.0013, freq_hz=75.0, coll_thick=None) -> SimulationResult` | Single simulation (JAX subclass adds `jit=False`). |
| `run_batch` | `(population: list, duration=8.0, fold_duration=3.0, vibrate=True, amp_x=0.0, amp_z=0.0013, freq_hz=75.0, batch_size=None) -> np.ndarray` | Fitness array, one entry per individual. Implementations are sequential loops, not parallel. |
| `is_available` | `() -> bool` | Dependencies installed **and** usable. |
| `device_info` | `() -> str` | Human-readable device description. |

### Registry functions

| Function | Signature | Behavior |
|---|---|---|
| `register_backend` | `(name: str, cls: type[SimulationBackend]) -> None` | Add/replace a registry entry. |
| `available_backends` | `() -> list[str]` | Instantiate each registered class; include its name if `is_available()` is true. Exceptions during the probe are swallowed. On a CPU-only install this is `["jax"]`. |
| `get_backend` | `(name: str) -> SimulationBackend` | Unknown name → `ValueError(f"unknown backend {name!r}; expected one of {sorted(_REGISTRY)}")`. Unavailable backend → `RuntimeError` with install hint. |

Built-ins are registered at import time as `"jax"` → `JaxBackend` and `"warp"` → `WarpBackend`.

### `JaxBackend` (`origami_engine_mjx.backends.jax_backend`)

`name = "jax"`; available when both `jax` and `mujoco.mjx` import successfully (CPU, GPU, or TPU).

| Method | Behavior |
|---|---|
| `build_model(xml, info)` | `MjModel.from_xml_string(xml)` → `mjx.put_model(...)`; returns `{"mjx_model", "classic_model", "info"}`. |
| `make_state(bundle)` | `mjx.make_data(bundle["mjx_model"])`. |
| `run(..., jit=False)` | Normalizes creases, builds pattern/XML/model, steps `round(duration / 0.001)` times with `mjx.step`; `jit=True` wraps the step function in `jax.jit`. Zeroes hinge `qpos`, optionally drives the table, checks finite `qpos`, and returns a `SimulationResult`. |
| `run_batch(..., batch_size=None)` | Loops over individuals, calls `run(...)`, and scores `delta[0]` (+x). Exceptions become `nan`. **Does not JIT or vmap**, and ignores `batch_size`. |
| `is_available()` | `jax is not None and mjx is not None`. |
| `device_info()` | `"jax not installed"` or `"jax backend=<backend>, devices=[...]"`. |

### `WarpBackend` (`origami_engine_mjx.backends.warp_backend`)

`name = "warp"`; available only when `mujoco.warp` imports **and** `warp.get_device().is_cuda` is true. All Warp imports are failure-guarded; importing the module does not initialise CUDA.

| Method | Behavior |
|---|---|
| `build_model(xml, info)` | `MjModel.from_xml_string(xml)` → `mjw.make_model(...)`; returns `{"warp_model", "classic_model", "info"}`. |
| `make_state(bundle)` | `mjw.make_data(bundle["warp_model"])`. |
| `run(..., coll_thick=None)` | Same normalization/validation as JAX, then a Python loop over `round(duration / 0.001)` `mjw.step` calls with NumPy control writes. No `jit` parameter. |
| `run_batch(..., batch_size=None)` | Sequential per-individual loop calling `run(...)`, scoring `delta[0]`; exceptions become `nan`. **Ignores `batch_size`**; native batched Warp execution is not implemented. |
| `is_available()` | `False` when `mujoco`/`mujoco.warp` are missing or no CUDA device is present (`warp.get_device().is_cuda`). |
| `device_info()` | `"mujoco-warp not installed"`, `"warp device=<dev>, cuda=<bool>"`, or a diagnostic error string. |

See [BACKENDS.md](BACKENDS.md) for the semantics comparison and [COLAB_GPU.md](COLAB_GPU.md) for GPU validation steps.

## `origami_engine_mjx.schemas` — packaged data

This directory has no `__init__.py`; it ships one data file, `crease_pattern.schema.json` (JSON Schema draft 2020-12, `$id` `https://example.org/origami-simulator/crease_pattern.schema.json`). It is loaded into `CREASE_SCHEMA` and included in wheels/sdists via package data.

## `origami_engine_mjx.crease_patterns` — packaged data

This directory has no `__init__.py`; it ships `reference_miura_4x4.json`, the 4×4 Miura-ori reference pattern (24 creases, 16 faces, 9 loop edges) with `"angle_unit": "degree"` and ±45° angles. Access it through `DEFAULT_PATTERN_JSON` / `load_default_creases()`.

## Complete workflow example

```python
from origami_engine_mjx import (
    OrigamiEngine,
    build_pattern,
    generate_xml,
    load_default_creases,
    normalize_creases,
    plot_pattern,
)

# 1. Load the reference Miura-ori pattern (24 creases)
creases = load_default_creases()

# 2. Reconstruct and validate the face graph (shared Shapely geometry)
pat = build_pattern(creases)
assert pat.valid, pat.reason

# 3. Visualize the flat crease pattern
plot_pattern(creases, pat)

# 4. Inspect the generated MJCF (no MuJoCo runtime needed)
xml, info = generate_xml(pat, creases)
print(info["n_faces"], info["n_hinges"], info["n_loops"])

# 5. Run the MJX-JAX simulation (eager, CPU default; use a GPU for speed)
engine = OrigamiEngine(backend="jax")
result = engine.run(normalize_creases(creases), duration=2.0, vibrate=False)

# 6. Score it: forward displacement along +x
score = engine.evaluate(creases, trajectory=(1.0, 0.0, 0.0))
print(f"forward displacement: {score:.6f} m  (backend={result.backend})")
```

## `origami_engine_mjx.integrations` — design space & populations

The integration substrate is documented in depth in
[INTEGRATIONS.md](INTEGRATIONS.md); the names added by the configurable search
space and the population helpers are:

- `integrations.design_space`: `DesignSpace`, `DesignSpaceBuilder`, `Variable`,
  `ANGLE`, `COORD`, `PRESENCE` — see [SEARCH_SPACE.md](SEARCH_SPACE.md).
- `integrations.population`: `Population`, `SeedSet`, `load_seeds`,
  `interpolate`, `piecewise_path`, `random_barycentric`, `jitter`,
  `random_population`, `lhs_population`, `sobol_population`,
  `gaussian_population`, `deduplicate`, `diversity_filter`, `mixed_population`,
  `save_population`, `load_population`, `export_creases_dir`,
  `population_stats` — see [POPULATIONS.md](POPULATIONS.md).
- `integrations.evaluator.topology_signature` and `EvalRecord.path` /
  `BatchResult.paths` — the per-individual scoring path.

All of these are re-exported from `origami_engine_mjx.integrations`; existing
imports are unchanged.

### `GenerationLogger` / `GenerationSummary`, `load_generation_history`

`generation_log` records the caller's own loop: per-generation
best/mean/std/worst/median, global best, wall time and eval counts, plus a
best-per-generation checkpoint (`keep_every_best=True`). See
[EVALUATION.md](EVALUATION.md#logging-generations-generationlogger).

```python
GenerationLogger(path, space=None, *, maximising=True, best_path="auto",
                 summary_path="auto", keep_every_best=False,
                 descriptor_names=None, flush=True)
.end_generation(generation, batch_or_records, genomes=None) -> GenerationSummary
.log_batch(batch, genomes=None, *, generation=None)   # optional eval rows
.log(record, genome=None, *, generation=None)
.curve() -> (generations, best, mean, global_best)
```

`RunLogger(path, format=None, descriptor_names=None, *, include_generation=False)`
gains the `include_generation` flag and `generation=` tags; the default schema
is unchanged.

## `origami_engine_mjx.rendering` — video, cameras, plots

Rendering replays a recorded trajectory (`run(record=True)`) through MuJoCo's
offscreen renderer. Needs `MUJOCO_GL=egl` (or `osmesa`); MP4 needs `ffmpeg`
(GIF works without it). See [SIMULATION.md](SIMULATION.md#rendering).

| Function | Purpose |
|---|---|
| `render_video(result, path, *, camera=None, cameras=None, views=None, prefix=None, plot=None, overlay="freq/amp", resolution=(480, 360), fps=25, max_frames=None, lookat=None, zoom=1.0, excitation=None, overlay_extra=None, save_camera_json=True) -> Path` | Render and encode one video (MP4, or GIF by suffix). |
| `render_frames(result, *, cameras=None, views=None, plot=None, max_frames=None, resolution=(480, 360), ...) -> list[np.ndarray]` | Same compositing without encoding. |
| `layout_frames(tiles, *, plot=None, plot_images=None, overlay=..., overlay_values=None, resolution=None) -> np.ndarray` | Concatenate camera tiles (+ graph column) and draw the overlay. |
| `CameraSpec` | `name`, `lookat`, `distance`, `azimuth`, `elevation` (`as_dict()`). |
| `resolve_camera(camera=None, *, prefix=None, zoom=1.0, lookat=None, bounds=None) -> CameraSpec` | Preset name, `(az, el[, dist[, zoom]])`, mapping or `CameraSpec` → `CameraSpec`. |
| `resolve_cameras(cameras) -> list[CameraSpec]` | Any of the above, or a list of them. |
| `available_cameras(prefix=None)`, `CAMERA_PRESETS`, `cinematic_views(prefix=None)` | Preset names (26) and the 4-view set used by `views="all_sides"`. |
| `save_camera(spec, path)`, `load_camera(path)` | Camera JSON sidecar round-trip. |
| `Trajectory` | `.from_result(result)` / `.from_dict(dict)`; `time`, `qpos`, `qvel`, `centroid`, `displacement`, `center_of_mass(relative=True)`, `velocity(...)`, `hinge_angles()`, `subsample(n)`, `subset(idx)`. |
| `plot_trajectory(obj, axis="all", *, path=None, ...)`, `plot_axes`, `movement_figure`, `axes_figure` | Movement graphs (per axis / hinge actuation). |
| `trajectory_to_image(obj, axis="x", size=..., progress=None, style=...)` | One panel as an RGB array (used inside videos). |
| `plot_history(history, *, path=None, maximising=None, ...)`, `history_to_image(history, ...)` | Best/mean/global-best per generation from `GenerationLogger`. |
| `default_overlay(freq_hz, amp_x=..., amp_z=..., vibrate=True, extra=None)` | Default bottom-left freq/amp text factory. |
| `draw_text`, `caption`, `overlay_text`, `hconcat`, `vconcat`, `with_letterbox` | Compositing helpers. |
| `VideoWriter`, `ffmpeg_available`, `write_gif` | Encoding helpers. |

## See also

- [GETTING_STARTED.md](GETTING_STARTED.md) — installation (pip / uv / source) and first run
- [TUTORIALS.md](TUTORIALS.md) — eight step-by-step recipes
- [CREASE_FORMAT.md](CREASE_FORMAT.md) — crease & JSON schema reference
- [SIMULATION.md](SIMULATION.md) — XML model structure and stepping loop
- [EVALUATION.md](EVALUATION.md) — fitness functions, population evaluation, current caveats
- [ARCHITECTURE.md](ARCHITECTURE.md) — module layout and data flow
- [BACKENDS.md](BACKENDS.md) — MJX-JAX vs MJX-Warp: semantics, features, limitations
- [COLAB_GPU.md](COLAB_GPU.md) — Colab T4 setup, validation checklist, benchmarking
- [MIGRATION.md](MIGRATION.md) — migrating from `origami-engine` (classic MuJoCo)
- [INTEGRATIONS.md](INTEGRATIONS.md) — ES/RL/AI interfaces (`origami_engine_mjx.integrations`)
- [OPTIMIZATION_ES.md](OPTIMIZATION_ES.md) — search space, evaluator, optimizer adapters
- [SEARCH_SPACE.md](SEARCH_SPACE.md) — configurable `DesignSpace` (`integrations.design_space`)
- [POPULATIONS.md](POPULATIONS.md) — population seeding (`integrations.population`)
- [RL.md](RL.md) — Gymnasium and JAX-functional environments
