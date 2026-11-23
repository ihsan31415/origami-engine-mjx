# Crease Pattern & JSON Input Format

> **Documentation map:** [README](../README.md) · [Getting Started](GETTING_STARTED.md) · [Tutorials](TUTORIALS.md) · **Crease Format** · [Simulation](SIMULATION.md) · [Evaluation](EVALUATION.md) · [API Reference](API_REFERENCE.md) · [Architecture](ARCHITECTURE.md) · [Troubleshooting](TROUBLESHOOTING.md) · [Glossary](GLOSSARY.md) · [Contributing](CONTRIBUTING.md)

This document is the complete reference for the crease input format: the `Crease` type, the JSON Schema, accepted input shapes, validation rules, and how a crease list becomes a validated `Pattern`.

## The `Crease` type

```python
from origami_engine_mjx import Crease

Crease(p1=(0.2, 0.0), p2=(0.45, 0.25), angle=-0.7853981633974483)
```

| Field | Type | Meaning |
|---|---|---|
| `p1`, `p2` | `tuple[float, float]` | Endpoints of the crease segment, in sheet coordinates. The sheet is the unit square: `0 <= x, y <= S` with `S = 1.0`. |
| `angle` | `float` | **Signed target fold angle in radians** (in the Python API). **Negative = mountain fold, positive = valley fold.** |

`Crease` is a frozen dataclass (`origami_engine_mjx/types.py`), so instances are immutable and hashable.

The package constants `M = +π/4` and `V = −π/4` (`origami_engine_mjx/constants.py`) are the canonical ±45° angles used by the reference pattern builders (`reference_miura_legacy_creases()`, `reference_miura_render_creases()`). The sign convention that matters at runtime is the one on `Crease.angle` above (also used by `plot_pattern`: positive → Valley, blue dash-dot; negative → Mountain, crimson dashed). When the XML model is generated, the target is additionally mirrored (`sgn * angle`) depending on which side of the hinge the child face lies on, then clipped to `±MAX_FOLD` (`MAX_FOLD = 0.98π`) — see [SIMULATION.md](SIMULATION.md).

**Note on constants naming:** The constants `M = +π/4` and `V = −π/4` have names that are swapped relative to the runtime sign convention (negative=mountain, positive=valley). `M` (Mountain) is actually a positive angle (valley), and `V` (Valley) is actually a negative angle (mountain). This is inherited verbatim from the upstream `origami-engine` codebase. **Do not change the constant values** — they are part of the public API and changing them would alter the reference pattern geometry. Use `Crease.angle` directly for the actual fold direction.

## Accepted input shapes

The simulation entry points — `OrigamiEngine.run`, `OrigamiEngine.evaluate`, `OrigamiEngine.evaluate_population`, `evaluate_individual`, `evaluate_population` — accept any of:

1. A **list or tuple of `Crease` objects**
2. A **list or tuple of dicts** `{"p1": [x, y], "p2": [x, y], "angle": <radians>}`
3. A **path to a JSON file** (string or `pathlib.Path`) — loaded and schema-validated by `load_creases_from_json`

They normalize internally via `normalize_creases`. `build_pattern`, `generate_xml`, and `plot_pattern` do **not** normalize: pass them a sequence of `Crease` objects (for example `normalize_creases(raw)`), because `build_pattern` reads `crease.p1`/`crease.p2` directly and raises `AttributeError` for dicts. A JSON path is also rejected by `build_pattern`.

```python
from origami_engine_mjx import DEFAULT_PATTERN_JSON, Crease, normalize_creases

normalize_creases([Crease((0, 0), (1, 0), 0.5)])  # Crease list
normalize_creases([{"p1": [0, 0], "p2": [1, 0], "angle": 0.5}])  # dict list
normalize_creases("crease_patterns/reference_miura_4x4.json")  # JSON path (repo checkout)
normalize_creases(DEFAULT_PATTERN_JSON)  # packaged copy (any install)
```

Errors raised by `normalize_creases`: `ValueError` for `None`, empty sequences, dicts with missing keys, or unsupported element types; `TypeError` if the input is not a list/tuple/path.

## JSON file format

Validated at load time against the packaged schema [`crease_pattern.schema.json`](../origami_engine_mjx/schemas/crease_pattern.schema.json) (JSON Schema draft 2020-12) using `jsonschema`. Invalid files raise `ValueError("invalid crease JSON: ...")`.

Two forms are allowed (`oneOf` in the schema):

**Form A — object with an explicit angle unit** (recommended for humans):

```json
{
  "angle_unit": "degree",
  "creases": [
    { "p1": [0.0, 0.5], "p2": [0.5, 0.5], "angle": -90.0 },
    { "p1": [0.5, 0.5], "p2": [1.0, 0.5], "angle": 90.0 }
  ]
}
```

- `angle_unit`: one of `"radian"`, `"radians"`, `"rad"`, `"degree"`, `"degrees"`, `"deg"` (the schema enum is lowercase; the loader lower-cases the parsed string for its degree check). **Default/omitted = radians.** Degrees are converted with `np.deg2rad`.
- `creases`: array, `minItems: 1`.

**Form B — bare list of creases** (angles are **radians**):

```json
[
  { "p1": [0.0, 0.5], "p2": [0.5, 0.5], "angle": -1.5707963 },
  { "p1": [0.5, 0.5], "p2": [1.0, 0.5], "angle": 1.5707963 }
]
```

Each crease object requires exactly the three keys:

| Key | Schema | Meaning |
|---|---|---|
| `p1` | `array` of 2 numbers | Start point (sheet coordinates) |
| `p2` | `array` of 2 numbers | End point (sheet coordinates) |
| `angle` | `number` | Signed target fold angle (unit per `angle_unit`; radians in bare-list form) |

The schema object is also exported as `origami_engine_mjx.CREASE_SCHEMA` if you want to validate files in your own tooling before calling the library:

```python
import json, jsonschema
from origami_engine_mjx import CREASE_SCHEMA

payload = json.load(open("my_pattern.json"))
jsonschema.validate(payload, CREASE_SCHEMA)  # raises jsonschema.ValidationError if invalid
```

### Saving patterns

```python
from origami_engine_mjx import save_creases_to_json

save_creases_to_json(creases, "out.json", angle_unit="degree")  # or "radian" (default)
```

Writes Form A with 2-space indentation, creating parent directories as needed, and returns the output `Path`. `angle_unit` must be `"radian"` or `"degree"` (else `ValueError`). Round-tripping (`save` → `load`) is covered by `tests_mjx/test_shared.py::test_json_round_trip`.

## The reference pattern

The reference pattern is a 4×4 Miura-ori: **24 creases, 16 faces, 9 loop edges**, stored with `"angle_unit": "degree"` and ±45° angles. It exists in two byte-identical copies (guarded by `tests/test_packaging.py`): the repo-root `crease_patterns/reference_miura_4x4.json` (linked from the README) and the packaged `origami_engine_mjx/crease_patterns/reference_miura_4x4.json`, which **ships in the wheel** so it is available in every install method.

```python
from origami_engine_mjx import DEFAULT_PATTERN_JSON, load_default_creases

DEFAULT_PATTERN_JSON  # -> <package>/origami_engine_mjx/crease_patterns/reference_miura_4x4.json
load_default_creases()  # reads DEFAULT_PATTERN_JSON (always present in wheel + checkout)
```

## From creases to a validated `Pattern`

`build_pattern(creases)` (in `origami_engine_mjx/geometry.py`, Shapely-based) performs:

1. **Noding & polygonization** — the sheet boundary `box(0,0,S,S)` and all crease segments are unioned (`unary_union`) and polygonized into faces.
2. **Validation** — the pattern is rejected (with a reason string) when:

| `pat.reason` | Meaning | Typical fix |
|---|---|---|
| `"no faces"` | Creases do not partition the sheet into any polygon | Add boundary-to-boundary or closed-loop creases |
| `"sliver face (area …)"` | Some face is smaller than `MIN_FACE_AREA = 1e-4` | Remove near-duplicate crease endpoints |
| `"faces do not tile the sheet"` | Face areas don't sum to `S²` within 1e-6 | Creases must end exactly on other creases or the boundary (snap coordinates) |
| `"disconnected faces"` | The face adjacency graph is not connected | Connect all panels through shared edges |

3. **Edge extraction** — each pair of touching faces contributes an edge `{"i", "j", "p", "q", "crease"}` (`p`/`q` = endpoints, `crease` = index into the crease list, −1 for boundary segments).
4. **Spanning tree** — BFS from the largest-area face (`pat.root`) assigns `parent`, `depth`, `tree_edge` per face. Tree edges become **hinge joints** in the MuJoCo model; all remaining edges are `loop_edges` and are closed with equality constraints.

The returned `Pattern` dataclass:

| Field | Type | Meaning |
|---|---|---|
| `valid` | `bool` | Whether all checks passed |
| `reason` | `str` | Failure reason (`""` when valid) |
| `faces` | `list` | Shapely polygons |
| `edges` | `list[dict]` | Face-adjacency edges (see above) |
| `root` | `int` | Root face index (largest area) |
| `parent` / `depth` / `tree_edge` | `dict` | Spanning-tree bookkeeping per face |
| `loop_edges` | `list[int]` | Edge indices **not** in the spanning tree |
| `n_faces` / `n_loops` | `int` | Convenience counts |

## Designing your own patterns

- Work in the unit square `[0, 1]²`; all crease endpoints must lie on the boundary or exactly on another crease (the tiling check is exact to 1e-6).
- Every region enclosed by creases/boundary becomes a face; avoid faces smaller than `1e-4` in area.
- The face graph must be connected — an island of polygons floating disconnected from the rest is rejected.
- The engine enforces topology (tiling, area, connectivity), **not** rigid-foldability theorems (Kawasaki/Maekawa). For physically foldable patterns, keep the usual origami design rules in mind (e.g. alternating mountain/valley at interior vertices), but note the simulator's hinge targets are what actually drive the fold.
- Angles are targets for position actuators, clipped to `±MAX_FOLD = 0.98π` ≈ 176.4°.

## See also

- [TUTORIALS.md](TUTORIALS.md) — Tutorial 4 (JSON round-trip) and Tutorial 5 (custom pattern)
- [SIMULATION.md](SIMULATION.md) — how `Pattern` becomes MuJoCo XML
- [API_REFERENCE.md](API_REFERENCE.md) — `Crease`, `Pattern`, `normalize_creases`, `load_creases_from_json`, `save_creases_to_json`, `CREASE_SCHEMA`, `build_pattern`
- [GLOSSARY.md](GLOSSARY.md) — mountain/valley, loop edge, spanning tree, …