# Optimization (ES / black-box)

> **Documentation map:** [README](../README.md) · [Docs index](README.md) · [Getting Started](GETTING_STARTED.md) · [Tutorials](TUTORIALS.md) · [Crease Format](CREASE_FORMAT.md) · [Simulation](SIMULATION.md) · [Evaluation](EVALUATION.md) · [API Reference](API_REFERENCE.md) · [Architecture](ARCHITECTURE.md) · [Backends](BACKENDS.md) · [Integrations](INTEGRATIONS.md) · **Optimization (ES)** · [RL](RL.md) · [Troubleshooting](TROUBLESHOOTING.md) · [Glossary](GLOSSARY.md) · [Contributing](CONTRIBUTING.md)

This guide shows how to drive the folding simulation from an external
black-box optimizer (evolution strategy, CMA-ES, Bayesian optimization, genetic
algorithm, MAP-Elites, …). The library provides the **search space** and the
**objective**; the algorithm comes from a third-party package.

## 1. Define the search space

`SearchSpace` is a plain dataclass over NumPy arrays, so it is picklable and can
be shipped to worker processes.

```python
from origami_engine_mjx.integrations import SearchSpace

# Packaged 4x4 Miura-ori reference pattern (from origami_engine_mjx.load_default_creases()).
space = SearchSpace.from_default_pattern()
print(space.dim)  # 15  (actuated hinges only)
print(space.n_creases)  # 24  (total creases in the fixed topology)
print(space.lower[:3], space.upper[:3])  # (-0.98π, ...,) (+0.98π, ...)
```

### Fixed topology, variable angles

- The crease **endpoints** (`p1`, `p2`) are fixed; only the signed fold
  **angles** are variables. Moving endpoints can change the number of faces and
  hinges; [`DesignSpace`](SEARCH_SPACE.md) supports that, but the built-in
  backends then score each topology separately (they already loop per
  individual).
- Only creases that become **spanning-tree hinges** are actuated. The
  remaining **loop-edge** creases become equality `connect` constraints and
  their angle is ignored by the dynamics. With `active_only=True` (the
  default) the genome therefore contains one angle per *actuated* hinge.
- `active_only=False` makes every crease angle a variable, including the inert
  loop creases — useful if you want a full-length genome, but those extra
  variables have **no effect** on the simulation.

For the default pattern: 24 creases → 16 faces → 9 loop edges → **15 actuated
hinges** → 15 variables.

The genome is **configurable**: `SearchSpace` stays the default, while
[`DesignSpace`](SEARCH_SPACE.md) can also vary the loop-edge angles, crease
**endpoints** (geometry) and crease **presence** (a variable crease count), with
presets, selection, equality/symmetry constraints and a JSON form. Its
`angles_only` preset reproduces `SearchSpace` exactly, so nothing changes unless
you opt in. When the topology can vary between individuals, the evaluator buckets
them by `topology_signature` and reports the per-individual scoring path (see
[EVALUATION.md](EVALUATION.md)). Seed an initial population with
[`population`](POPULATIONS.md) before handing it to an optimizer.

### Build a space from your own pattern

```python
from origami_engine_mjx import Crease
from origami_engine_mjx.integrations import SearchSpace

creases = [
    Crease((0.2, 0.0), (0.5, 0.5), -0.7853981633974483),
    Crease((0.5, 0.5), (0.8, 0.0), -0.7853981633974483),
]
space = SearchSpace.from_creases(creases, angle_limits=(-1.5, 1.5), name="my_pattern")
```

### Bounds, sampling, repair

```python
genome = space.sample(0)  # uniform in angle space, (dim,); first arg is a seed or RNG
batch = space.sample(0, size=32)  # (32, dim)

z = space.to_normalized(genome)  # view in [-1, 1]
back = space.from_normalized(z)  # back to angles (clipped)

safe = space.repair(candidate)  # clip + replace NaN with base angle
ok = space.is_valid(candidate)  # decodes to a valid pattern?
```

`decode(genome)` returns the full `list[Crease]` (inactive creases keep their
base angles); `encode(creases)` maps a matching crease list back to a genome and
raises if the topology differs.

## 2. Define the objective

`OrigamiEvaluator` wraps `OrigamiEngine.run` + `compute_forward_displacement`
into a stateless, picklable callable. It contains no search.

```python
from origami_engine_mjx.integrations import OrigamiEvaluator

evaluator = OrigamiEvaluator(
    space,
    backend="jax",
    duration=1.0,  # simulated seconds per evaluation
    fold_duration=0.5,
    vibrate=False,  # deterministic, fold-only score
    maximize=True,  # fitness = raw displacement
)
```

### Fitness convention

Fitness is **always higher-is-better**:

| `maximize` | raw displacement | `fitness` |
|---|---|---|
| `True` (default) | `d` | `d` |
| `False` | `d` | `-d` |

A minimizer (most libraries) should optimize `evaluator.to_minimizer()`, which
returns `-fitness`. The `maximize` flag still decides which physical direction
is "better".

### Single vs batch

```python
record = evaluator.evaluate(genome)  # EvalRecord: fitness, raw, status, descriptors
result = evaluator.evaluate_batch(X)  # BatchResult over (N, dim)

# Fitness-only fast path (skips descriptor bookkeeping):
result = evaluator.evaluate_batch(X, return_details=False)

# Convenience callables:
f = evaluator(genome)  # float
F = evaluator(X)  # (N,) array
```

On the fast path `BatchResult.raw` is the backend's +x displacement (the
`trajectory` is validated but not applied); the structured path projects onto
`trajectory`.

### Failures never crash a batch

With the default `error_policy="penalty"`, a bad genome returns the documented
`penalty` (default `-1e6`) and a `status` string instead of raising. Use
`error_policy="raise"` to re-raise decode/simulation exceptions. `EvalRecord.status`
is one of `"ok"`, `"invalid"` (decode failed), `"nonfinite"` (non-finite score —
never raised) or `"error"` (simulation raised). A wrong-length genome is a
programming error and always raises `ValueError`, regardless of the policy.

### Behavior descriptors (for quality-diversity)

`EvalRecord.descriptors` and `BatchResult.descriptors` expose
`DESCRIPTOR_NAMES`, computed from the final simulation state:

```python
from origami_engine_mjx.integrations import DESCRIPTOR_NAMES

print(DESCRIPTOR_NAMES)
# ('displacement_x', 'displacement_y', 'displacement_norm', 'direction_x',
#  'direction_y', 'final_x', 'final_y', 'fold_amount')
```

Descriptors require the **structured** path (one `engine.run` per genome). The
fast batch path returns empty descriptors. For a MAP-Elites-style archive, call
`evaluate_batch(..., return_details=True)` (or set `return_descriptors=True`)
and bin on e.g. `direction_x`/`displacement_norm`.

### Caching

Evolutionary loops re-evaluate elites; enable the cache to skip exact repeats:

```python
evaluator = OrigamiEvaluator(space, use_cache=True)
evaluator.evaluate(genome)  # runs the simulation
evaluator.evaluate(genome)  # cached (record.cached == True)
print(evaluator.n_evaluations, evaluator.n_cache_hits)
```

## 3. Hand it to an optimizer

The adapters reshape the evaluator into a library's API. They never implement a
search.

| Adapter | Returns | For |
|---|---|---|
| `to_nevergrad_function(evaluator)` | `f(x) -> float`, minimized | `nevergrad`, `cma` |
| `to_evosax_fitness(evaluator)` | `f(X) -> (N,)`, higher-is-better | `evosax` (JAX; negate before `tell`) |
| `to_pymoo_problem(evaluator)` | `pymoo.core.problem.Problem` | `pymoo` |

```python
import numpy as np
import nevergrad as ng
from origami_engine_mjx.integrations.adapters import to_nevergrad_function

objective = to_nevergrad_function(evaluator)
optimizer = ng.optimizers.OnePlusOne(
    parametrization=ng.p.Array(init=np.zeros(space.dim)).set_bounds(space.lower, space.upper),
    budget=64,
)
recommendation = optimizer.minimize(objective, verbosity=0)
best = space.repair(np.asarray(recommendation.value))
```

```python
from origami_engine_mjx.integrations.adapters import to_pymoo_problem

problem = to_pymoo_problem(evaluator)  # n_var=15, n_obj=1, xl/xu from the space
```

A complete, runnable version is
[`examples/04_optimize_es.py`](../examples/04_optimize_es.py).

> **Direction:** `nevergrad`, `cma`, `pymoo` **and** `evosax` minimize. The
> first three adapters return `-fitness`. `to_evosax_fitness` returns the
> evaluator's higher-is-better fitness, so negate it when calling
> `es.tell(..., -fitness, ...)` (`state.best_fitness` is the lowest value seen).
> Because `fitness` already folds in `maximize`, you never flip signs twice.

## 4. Plumbing

```python
from origami_engine_mjx.integrations import (
    make_rng,
    set_global_seed,
    save_genome,
    load_genome,
    save_best,
    load_best,
    RunLogger,
    benchmark,
)

set_global_seed(0)  # seeds Python random + NumPy (JAX has no global RNG)
rng = make_rng(0)  # reproducible NumPy Generator

save_genome(genome, space, "genome.json")  # standard crease JSON
genome = load_genome("genome.json", space)

save_best(genome, space, "best.json", fitness=1.2, extra={"cycle": 3})
genome, meta = load_best("best.json", space)

with RunLogger("run.jsonl") as log:  # or "run.csv"
    log.log_batch(evaluator.evaluate_batch(space.sample(0, size=8)))

metrics = benchmark(evaluator, n=16)  # throughput on random genomes
print(metrics["evals_per_sec"])
```

`save_best` writes a normal crease file plus `fitness` and `meta` keys that the
crease reader ignores, so checkpoints stay loadable by
`load_creases_from_json`.

## 5. A minimal evolution-strategy loop

This is the shape every optimizer ends up using; the loop below is illustrative
(the real algorithm would be CMA-ES/ES from a library):

```python
budget = 64
rng = make_rng(0)
genome = space.sample(rng)
best = genome
best_fit = evaluator(genome)

for _ in range(budget):
    candidates = space.sample(rng, size=8)  # explore
    fitness = evaluator(candidates)  # (8,) signed fitness
    i = int(np.argmax(fitness))
    if fitness[i] > best_fit:
        best, best_fit = candidates[i], fitness[i]

print("best raw displacement:", evaluator.raw(best))
```

## 6. Cost notes

- Every evaluation is one full `OrigamiEngine.run`; the integration layer adds
  no physics cost.
- On **CPU**, MJX-JAX mesh-contact models are slow (README: ~70 s/step for the
  16-face reference model, ~1.4 s/step for a tiny 2-face model). Use short
  `duration` and tiny patterns for tests; use a GPU for real budgets.
- Descriptors and the cache force the **structured** per-genome path; the
  fitness-only fast path delegates to `evaluate_population`, which still loops
  per individual today (`batch_size` ignored — see [EVALUATION.md](EVALUATION.md)).
- GPU-only performance claims are not verified in this repository.

## See also

- [INTEGRATIONS.md](INTEGRATIONS.md) — package overview and boundaries
- [SEARCH_SPACE.md](SEARCH_SPACE.md) — the configurable `DesignSpace`
- [POPULATIONS.md](POPULATIONS.md) — seeding an initial population
- [EVALUATION.md](EVALUATION.md) — the raw fitness signal and its caveats
- [RL.md](RL.md) — the reinforcement-learning environments
- [`examples/04_optimize_es.py`](../examples/04_optimize_es.py) — runnable ES demo
