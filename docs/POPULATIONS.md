# Population seeding & initialization

> **Documentation map:** [README](../README.md) · [Docs index](README.md) · [Getting Started](GETTING_STARTED.md) · [Tutorials](TUTORIALS.md) · [Crease Format](CREASE_FORMAT.md) · [Simulation](SIMULATION.md) · [Evaluation](EVALUATION.md) · [Fitness](FITNESS.md) · [API Reference](API_REFERENCE.md) · [Architecture](ARCHITECTURE.md) · [Backends](BACKENDS.md) · [Integrations](INTEGRATIONS.md) · [Optimization (ES)](OPTIMIZATION_ES.md) · [Search space](SEARCH_SPACE.md) · **Populations** · [RL](RL.md) · [Troubleshooting](TROUBLESHOOTING.md) · [Glossary](GLOSSARY.md) · [Contributing](CONTRIBUTING.md)

The `origami_engine_mjx.integrations.population` module produces the **initial
population** an external optimizer starts from. It ships **no** mutation,
crossover, selection or search — only seeding and initialization helpers:

- **seed loading** from crease-pattern JSON files,
- **interpolation** between seeds,
- **random initialization** (uniform, Gaussian, Latin hypercube, Sobol'),
- **mixed** populations (e.g. 20% seeds / 50% interpolation / 30% random),
- **validity-aware** generation, **de-duplication** and **diversity** selection,
- **population I/O** and **statistics**.

The helpers are duck-typed on the space: they use `dim`, `lower`, `upper`,
`sample`, `encode`, `is_valid`, `to_normalized` and `from_normalized`, all of
which both
[`SearchSpace`](OPTIMIZATION_ES.md#1-define-the-search-space) and
[`DesignSpace`](SEARCH_SPACE.md) provide. The Gaussian initializer also needs a
centre genome — `DesignSpace.base_vector()` or, for `SearchSpace`, the
`base_angles`/`active_indices` slice (falling back to the bounds midpoint).

## 1. Random initialization

```python
from origami_engine_mjx.integrations import (
    DesignSpace,
    gaussian_population,
    lhs_population,
    random_population,
    sobol_population,
)

space = DesignSpace.from_default_pattern()

random_population(space, 32, rng=0)  # uniform within bounds
random_population(space, 32, rng=0, method="gaussian")
random_population(space, 32, rng=0, method="lhs")
random_population(space, 32, rng=0, method="sobol")
```

- `random_population(space, n, rng=None, method="uniform")` dispatches to the
  named sampler. `rng` is a seed or `numpy.random.Generator`.
- `gaussian_population(space, n, sigma=0.25, rng=None)` perturbs the space's
  base genome; `sigma` is a fraction of each variable's span, clipped to bounds.
- `lhs_population` draws a Latin-hypercube sample in the normalized `[-1, 1]`
  view and maps it back into the space.
- `sobol_population` uses SciPy's `qmc.Sobol` when SciPy is installed and
  **falls back to Latin-hypercube** otherwise (the fallback is not flagged, so
  check for SciPy if you need true Sobol' points).

All samplers are reproducible from their `rng` and respect the per-variable
bounds.

## 2. Loading seed files

```python
from origami_engine_mjx.integrations import load_seeds

seeds = load_seeds("seed_patterns/", space)  # a directory of *.json files
seeds = load_seeds(["a.json", "b.json"], space)  # or an explicit list

print(len(seeds), len(seeds.rejected))
for path, reason in seeds.rejected:
    print("rejected", path, reason)
```

`load_seeds(sources, space)` reads each crease-pattern JSON file (via
[`load_creases_from_json`](CREASE_FORMAT.md)) and encodes it into the space:

- A seed whose **topology matches** the space is encoded normally.
- A seed that cannot be encoded (different crease count/topology, bad JSON,
  read error) is **rejected with a reason** in `SeedSet.rejected` — it never
  aborts the load and is never silently mangled.
- `SeedSet.genomes` is the `(n, dim)` matrix of accepted seeds and
  `SeedSet.sources` their paths.

## 3. Interpolation

```python
import numpy as np
from origami_engine_mjx.integrations import (
    interpolate,
    piecewise_path,
    random_barycentric,
)

a, b = space.sample(0), space.sample(1)
interpolate(a, b, 0.5)  # midpoint
interpolate(a, b, np.linspace(0, 1, 8))  # (8, dim) sweep

path = piecewise_path(np.vstack([a, b, space.sample(2)]), 16, rng=0)
path = piecewise_path(seeds, 16, rng=0, closed=True)  # loop back to the first
mix = random_barycentric(seeds, 16, rng=0)  # convex combinations
```

- `interpolate(a, b, t)` is `(1 - t) * a + t * b`; `t` may be a scalar or a
  vector. Shape mismatch raises `ValueError`.
- `piecewise_path(seeds, n, rng=None, closed=False)` samples `n` points along
  the polyline through `seeds` (at least two required).
- `random_barycentric(seeds, n, rng=None)` draws Dirichlet weights, so each row
  is a convex combination of the seeds.

Interpolation happens in **genome space**, so a coordinate that differs between
two seeds is interpolated too. Compatibility is explicit: seeds that live in the
same space are interpolated directly; seeds whose topology cannot be encoded are
rejected by [`load_seeds`](#2-loading-seed-files) before they reach an
interpolator, rather than producing nonsense.

Interpolation does **not** check geometric validity: an intermediate genome that
blends two valid seeds is not guaranteed to tile the sheet. Filter the result
with `DesignSpace.is_valid` / `rejection_sample`, or generate with
`mixed_population(..., valid_only=True)`.

## 4. Jitter

```python
from origami_engine_mjx.integrations import jitter

jitter(genomes, sigma=0.05, space=space, rng=0)
jitter(genomes, sigma=0.1, space=space, rng=0, normalized=True)
```

Adds Gaussian noise; with `space` given the result is clipped back into bounds.
`normalized=True` applies the noise in the `[-1, 1]` view, so `sigma` means the
same thing for angle and geometry variables.

## 5. Mixed populations with provenance

```python
from origami_engine_mjx.integrations import mixed_population, random_population

seeds = random_population(space, 5, rng=1)
pop = mixed_population(
    space,
    30,
    seeds=seeds,
    fractions={"seed": 0.2, "interp": 0.5, "random": 0.3},
    rng=0,
)

print(pop.genomes.shape)  # (30, dim)
print(pop.provenance[0])  # {'method': 'seed', 'seed_index': 3} etc.
```

`mixed_population(space, n, seeds=None, fractions=None, rng=None,
method="uniform", valid_only=False, dedup=True)`:

- `fractions` maps `"seed"`, `"interp"` and `"random"` to relative weights
  (normalized to `n`; the default is `0.2 / 0.5 / 0.3`). Unknown keys are
  ignored.
- The interpolated part uses `piecewise_path` (or `random_barycentric` for a
  single seed).
- If `seeds` are missing but the fractions ask for them, the whole population
  **falls back to random** rather than raising.
- `valid_only=True` drops genomes the space reports invalid;
  `dedup=True` removes duplicates.
- `Population.provenance` records `{"method", ...}` per genome, and the rows are
  shuffled so the parts are not contiguous.

## 6. Validity, de-duplication and diversity

```python
from origami_engine_mjx.integrations import deduplicate, diversity_filter

unique, keep = deduplicate(genomes)  # (unique_rows, kept_indices)
selected = diversity_filter(genomes, 16, space=space, rng=0)
```

- `deduplicate(genomes, space=None, ndigits=9)` keeps the first occurrence of
  each rounded genome and returns the unique rows plus their indices. With
  `space`, duplicates are detected in the normalized view (the returned rows are
  still genome-space rows).
- `diversity_filter(genomes, k, space=None, rng=None)` is **farthest-point
  sampling**: it seeds with one random genome and greedily adds the point
  farthest from those already chosen. With `space` the distance is measured in
  the normalized view, so angle and geometry variables are comparable. `k >= n`
  returns everything; `k <= 0` returns an empty selection.

## 7. Population I/O

```python
from origami_engine_mjx.integrations import (
    export_creases_dir,
    load_population,
    save_population,
)

save_population("pop.json", pop, space)  # or "pop.npz"
restored = load_population("pop.json")  # space spec round-trips

paths = export_creases_dir("creases/", pop, space)  # one crease JSON per genome
```

- `save_population(path, population, space=None, meta=None)` writes JSON or
  `.npz` (the suffix selects the format). The payload stores the genomes, the
  per-genome provenance and, when a space is given, the space spec
  (`DesignSpace.to_dict()` / `SearchSpace.to_dict()`) plus optional `meta`.
- `load_population(path, space=None)` rebuilds a `Population`. When `space` is
  omitted it restores a stored `DesignSpace` spec if present; a `SearchSpace`
  spec has no `from_dict`, so it is kept as metadata and `space` stays `None`.
- `export_creases_dir(directory, population, space)` writes one standard crease
  JSON per genome, handy for tools that read the crease format.

## 8. Statistics

```python
from origami_engine_mjx.integrations import population_stats

stats = population_stats(pop.genomes, space)
# {"n", "dim", "mean", "std", "min", "max",
#  "valid_fraction", "mean_pairwise_distance"}
```

`valid_fraction` is the share of genomes the space reports valid (or `None`
without a space); `mean_pairwise_distance` is measured in the normalized view.

## 9. Determinism and cost

- Every helper takes an explicit `rng` (seed or `Generator`) and is
  reproducible; there is no hidden global RNG.
- All of the above is **pure NumPy** except the optional SciPy Sobol' path. No
  simulation runs during initialization, so building a population is cheap
  regardless of the physics cost.

## See also

- [SEARCH_SPACE.md](SEARCH_SPACE.md) — the configurable `DesignSpace`
- [OPTIMIZATION_ES.md](OPTIMIZATION_ES.md) — feeding a population to an optimizer
- [CREASE_FORMAT.md](CREASE_FORMAT.md) — the seed / export JSON format
- [API_REFERENCE.md](API_REFERENCE.md) — integration signatures
- [GLOSSARY.md](GLOSSARY.md) — population, individual, genome
