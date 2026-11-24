# Configurable search space (`DesignSpace`)

> **Documentation map:** [README](../README.md) · [Docs index](README.md) · [Getting Started](GETTING_STARTED.md) · [Tutorials](TUTORIALS.md) · [Crease Format](CREASE_FORMAT.md) · [Simulation](SIMULATION.md) · [Evaluation](EVALUATION.md) · [Fitness](FITNESS.md) · [API Reference](API_REFERENCE.md) · [Architecture](ARCHITECTURE.md) · [Backends](BACKENDS.md) · [Integrations](INTEGRATIONS.md) · [Optimization (ES)](OPTIMIZATION_ES.md) · **Search space** · [Populations](POPULATIONS.md) · [RL](RL.md) · [Troubleshooting](TROUBLESHOOTING.md) · [Glossary](GLOSSARY.md) · [Contributing](CONTRIBUTING.md)

`DesignSpace` is the **configurable** companion to
[`SearchSpace`](OPTIMIZATION_ES.md#1-define-the-search-space). It can vary fold
angles, loop-edge angles, crease endpoints (geometry) and crease presence
(variable crease count). `SearchSpace` remains the default and the angles-only
preset of `DesignSpace` reproduces it **exactly** — same vectors, bounds,
parameter names and decoded creases — so existing pipelines are unaffected.

```python
from origami_engine_mjx.integrations import DesignSpace, SearchSpace

default = DesignSpace.from_default_pattern()  # 15 actuated angles, like SearchSpace
assert default.dim == SearchSpace.from_default_pattern().dim  # 15
```

The class lives in `origami_engine_mjx.integrations.design_space` and is
re-exported from `origami_engine_mjx.integrations` together with
`DesignSpaceBuilder`, `Variable`, and the kind tags `ANGLE`, `COORD`,
`PRESENCE`. See [POPULATIONS.md](POPULATIONS.md) for seeding an initial
population in a design space.

## 1. What can vary

Every variable is a `Variable` ([GLOSSARY.md](GLOSSARY.md)) targeting one or more crease
slots:

| Kind | Tag | Varies | Notes |
|---|---|---|---|
| Fold angle | `ANGLE` | one crease's signed angle | same semantics as `SearchSpace` |
| Coordinate | `COORD` | a crease endpoint `p1`/`p2`, component `x`/`y` | `mode="vertex"` shares one variable per unique vertex; `mode="segment"` gives each endpoint its own |
| Presence | `PRESENCE` | whether a crease is present | value `< presence_threshold` drops the crease |

### Presets

```python
from origami_engine_mjx.integrations import DesignSpace

DesignSpace.angles_only()  # dim 15  (default; == SearchSpace)
DesignSpace.angles_only(active_only=False)  # dim 24  (all creases, incl. inert loop edges)
DesignSpace.angles_and_geometry()  # dim 57  (15 angles + 42 vertex coords)
DesignSpace.geometry_only()  # dim 42  (vertex mode)
DesignSpace.geometry_only(geometry_mode="segment")  # dim 96 (per-crease endpoints)
DesignSpace.everything()  # dim 57  (angles + geometry)
DesignSpace.everything(presence=[0, 1, 2])  # dim 60 (adds 3 presence variables)
```

- `angles_only(active_only=True)` (the default) varies only **spanning-tree
  hinges**, exactly like `SearchSpace`. `active_only=False` includes the inert
  loop-edge creases; changing those has **no effect** on the simulation (they
  become equality constraints), which is useful only for a full-length genome.
- `geometry_mode="vertex"` deduplicates endpoints by rounded coordinate, so a
  shared vertex is one variable. `geometry_mode="segment"` gives every crease
  endpoint its own variable (larger, but independent).
- `axes` is any non-empty subset of `"xy"` (e.g. `axes="x"` halves the
  coordinate dimension).

For the packaged 4×4 Miura-ori reference: 24 creases → 16 faces → 9 loop edges
→ **15 actuated hinges** → 15 variables in the default preset.

## 2. Build a space from your own pattern

```python
from origami_engine_mjx.integrations import DesignSpace

space = DesignSpace.from_creases(creases, angle_limits=(-1.5, 1.5), name="my_pattern")
space = DesignSpace.from_default_pattern(active_only=True)
```

A space can also be converted from an existing `SearchSpace` and back:

```python
from origami_engine_mjx.integrations import DesignSpace, SearchSpace

ss = SearchSpace.from_default_pattern()
ds = DesignSpace.from_search_space(ss)
assert ds.to_search_space() == ss  # angle-only, unconstrained
```

`to_search_space()` raises a clear `ValueError` when the design space is not a
plain angle-only, unconstrained genome — geometry, presence or constraints
cannot be expressed as a fixed-topology `SearchSpace`. That is exactly the check
the RL environments use (see [RL.md](RL.md)).

## 3. Selecting variables

`select()` returns the genome indices matching an **AND** of the given filters;
`group_slice(group)` returns the indices of a named group.

```python
space = DesignSpace.angles_and_geometry()

space.group_slice("angles")  # [0, ..., 14]
space.select(group="geometry")  # all coordinate variables
space.select(name=space.parameter_names[0])  # a single variable by name
space.select(indices=slice(0, 3))  # first three
space.select(indices=[1, 4, 9])  # explicit list
space.select(indices=mask)  # boolean mask of length space.dim
space.select(predicate=lambda v: v.kind == "coord")  # by variable property
```

Out-of-range or wrong-length selections raise `ValueError` naming the offending
index/length, and an empty selection yields a space with no variables, which
`build()` rejects with a clear error rather than returning a degenerate space.
Groups are set by the builder (`group=` argument) and default to `"angles"`,
`"loop_angles"`, `"geometry"` and `"presence"`.

## 4. Fluent builder and dict/JSON form

`DesignSpaceBuilder` composes a space from variable sources:

```python
from origami_engine_mjx.integrations import DesignSpaceBuilder
from origami_engine_mjx.patterns import load_default_creases

space = (
    DesignSpaceBuilder(load_default_creases())
    .angles(active_only=True)  # spanning-tree hinges
    .loop_angles()  # the 9 inert loop edges
    .geometry(mode="vertex", axes="xy")  # shared-vertex endpoints
    .presence(indices=[0, 1, 2])  # optional crease count
    .constrain(symmetries=[(0, 1)])  # reduce the genome (see §6)
    .build(name="combo", count_mode="free")
)
```

`add(variable)` appends a hand-built `Variable`; `select(**filters)` previews
indices before building. A built space round-trips through JSON and `pickle`:

```python
import json

payload = json.dumps(space.to_dict())
restored = DesignSpace.from_dict(json.loads(payload))
assert restored == space
```

## 5. Bounds, sampling and the normalized view

`DesignSpace` exposes the same array surface as `SearchSpace`:

```python
genome = space.sample(0)  # (dim,) uniform within bounds
batch = space.sample(0, size=32)  # (32, dim)
z = space.to_normalized(genome)  # view in [-1, 1]
back = space.from_normalized(z)  # clipped back to bounds
safe = space.clip(candidate)  # clip to the per-variable bounds
safe = space.repair(candidate)  # clip + replace non-finite values with the base vector
```

Per-variable bounds come from the builder/preset (`bounds=`), so angle and
coordinate variables can have different limits. `space.lower` / `space.upper`
return the flattened per-genome bounds, and
[`space_spec(space)`](INTEGRATIONS.md) exposes them (plus names, groups and the
presence indices) to an external optimizer.

## 6. Constraints reduce the dimension

`with_constraints(equalities, symmetries)` links variables and **removes** the
dependent one from the genome; `decode` re-expands it. `(i, j)` in `equalities`
means *variable `j` equals variable `i`*; in `symmetries` it means *variable `j`
equals `-variable[i]`*.

```python
base = DesignSpace.from_default_pattern()  # dim 15
linked = base.with_constraints(symmetries=[(0, 1)], equalities=[(4, 5)])
assert linked.dim == base.dim - 2  # dim 13
assert linked.n_fixed == 2
```

Cycles, self-links and out-of-range indices raise `ValueError`. A constrained
space reports `n_free = dim` and `n_fixed = n_vars - dim`, and cannot be
converted to `SearchSpace` (the RL environments do not support the reduced
genome).

## 7. Variable crease count

With presence variables the crease **count** can change. Two views are
available:

- **Padded view** — one entry per base slot; `presence_mask(genome)` is the
  `(n_slots,)` boolean mask and `decode_padded(genome)` mirrors `decode`.
- **List view** — `decode(genome)` returns only the *present* creases, so its
  length changes with the genome.

```python
space = DesignSpace.everything(presence=[0, 1, 2], count_mode="free")
base = space.base_vector()
assert space.presence_mask(base).all()

dropped = base.copy()
dropped[space.select(group="presence")] = 0.0  # below the 0.5 threshold
assert len(space.decode(dropped)) == space.n_slots - 3
```

`count_mode` is one of `"fixed"`, `"range"` (with `count_bounds=(min, max)`) or
`"free"`. `count_bounds` is validated against the template size. `presence_count(genome)`
returns the decoded crease count and `count_ok(genome)` checks it against
`count_bounds` (always `True` unless `count_mode="range"`); `rejection_sample`
honours both `is_valid` and `count_ok`. The padded view suits fixed-length
pipelines; the list view suits anything that consumes a `list[Crease]`.

> **Trade-off:** dropping creases can make the remaining pattern no longer tile
> the sheet, so `build_pattern` may report it invalid. Use
> `is_valid` / `rejection_sample` and let the evaluator's `penalty` / `status`
> fallback absorb the rest (never a crash).

## 8. Validity, repair and rejection sampling

```python
ok = space.is_valid(genome)  # decodes to a build_pattern-valid pattern?
reason = space.validity_reason(genome)  # "" when valid, else the failure reason
repaired = space.repair(genome)  # clip + replace non-finite values
sample = space.rejection_sample(0, size=8)  # only valid genomes, bounded attempts
```

`is_valid` runs the cheap shared `geometry.build_pattern` check; `repair` is a
bounds projection that never raises for a correctly sized input, and
`rejection_sample` resamples until it finds genomes that are both geometrically
valid (`is_valid`) and, when `count_mode="range"`, within the crease-count bounds
(`count_ok`). It raises only if it exhausts its attempt budget.

## 9. Introspection

```python
info = space.describe()
# {"name", "dim", "n_vars", "n_free", "n_fixed", "n_slots", "kinds",
#  "groups", "has_presence", "count_mode", "count_bounds", "parameter_names"}
```

`n_free == dim` (variables stored in the genome), `n_fixed` counts constrained
(dependent) variables, `n_vars` is their sum, and `n_slots` is the template
crease count. `parameter_names`, `groups` and `variable_kinds` are aligned with
the genome.

## 10. Evaluating a design space

`OrigamiEvaluator` accepts a `SearchSpace` **or** a `DesignSpace` with no API
change. When the topology can change between individuals (geometry or presence
variables), the evaluator buckets genomes by
[`topology_signature`](API_REFERENCE.md) and reports **which path** scored each
one:

| `EvalRecord.path` / `BatchResult.paths` | Meaning |
|---|---|
| `"batched"` | one backend call for a same-topology batch |
| `"bucketed"` | a topology bucket holding more than one individual |
| `"single"` | one individual / one backend call |
| `"invalid"` | decode failed (scored with the penalty) |
| `"cached"` | served from the evaluator cache |

```python
from origami_engine_mjx.integrations import DesignSpace, OrigamiEvaluator

space = DesignSpace.from_default_pattern()  # one fixed topology
evaluator = OrigamiEvaluator(space, backend="jax")
result = evaluator.evaluate_batch(space.sample(0, size=8), return_details=False)
print(result.paths)  # ['batched', ...] — the whole batch is one topology

geo = DesignSpace.angles_and_geometry()  # geometry can change the topology
geo_result = OrigamiEvaluator(geo, backend="jax").evaluate_batch(geo.sample(0, size=8), return_details=False)
print(geo_result.paths)  # mostly ['single', ...] — each random geometry differs
```

For the built-in backends, `run_batch` already loops per individual (and
rebuilds the model each iteration), so a mixed-topology batch is scored
**exactly like one-by-one evaluation**, only grouped. The bucketing is therefore
bookkeeping only — same-topology rows share one backend *call*, but that call
still loops per individual — and currently gives **no throughput benefit**; see
[EVALUATION.md](EVALUATION.md) for the cost caveats.

## 11. Determinism

The design space is a plain dataclass over NumPy arrays: picklable, comparable,
and with no hidden RNG. `sample(rng)`, `rejection_sample(rng)` and the
population helpers all take an explicit seed or `numpy.random.Generator`.

## See also

- [OPTIMIZATION_ES.md](OPTIMIZATION_ES.md) — the default `SearchSpace` workflow
- [POPULATIONS.md](POPULATIONS.md) — seeding an initial population
- [EVALUATION.md](EVALUATION.md) — the fitness signal and cost caveats
- [RL.md](RL.md) — why RL stays angle-only
- [API_REFERENCE.md](API_REFERENCE.md) — integration signatures
- [GLOSSARY.md](GLOSSARY.md) — variable, presence, genome
