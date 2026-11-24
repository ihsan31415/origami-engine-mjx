---
name: origami-optimizer-integration
description: Integrate an external black-box optimizer, reinforcement-learning trainer, or AI/LLM proposer with origami-engine-mjx without adding algorithm code to the library. Use when writing examples, notebooks, adapters, or user code that consumes SearchSpace, DesignSpace, OrigamiEvaluator, OrigamiEnv, or the population/logging helpers.
---

# Integrate an optimizer with `origami-engine-mjx`

This is a working recipe for **consumer-side** integrations. The library is an
algorithm-free substrate: it owns the design space, the simulation, the
objective, the environment and the checkpoint format. The optimizer, trainer or
agent lives outside the package.

> **Non-negotiable boundary.** Do not add CMA-ES, ES, GA, PSO, RL or LLM code to
> `origami_engine_mjx/`. Add examples, notebooks and docs instead. If an
> integration needs a new library primitive, make it an interface/adapter and
> keep the algorithm out.

## When to use this recipe

Use it when you are:

* wiring `nevergrad`, `cma`, `pymoo`, `evosax`, `scipy`, a custom ES/GA, or a
  Bayesian-optimization package to the folding simulation;
* handing the simulation to a Gymnasium-compatible RL trainer
  (Stable-Baselines3, CleanRL, …);
* putting an LLM or other proposer in the loop;
* adding an example notebook or a docs page for any of the above.

## Ground rules

1. **Default behavior is sacred.** The packaged 4×4 pattern, 15 active angles
   and fixed topology must keep working unchanged. Pin it with
   `DesignSpace.from_default_pattern().to_search_space() == SearchSpace.from_default_pattern()`
   and never edit the engine, model or default constants to make an
   integration easier.
2. **Additive changes only.** New modules, examples, notebooks and docs are
   fine. Do not delete or restructure existing APIs, tests or documentation.
3. **No algorithm in the library.** Import third-party optimizers lazily and
   only from examples/notebooks. Never make an optimizer a hard dependency.
4. **MJX-JAX is the integration backend.** The Gymnasium/vector environments
   wrap the MJX-JAX backend only; the Warp backend is stateful and is not
   wrapped. Do not claim Warp RL support.
5. **Be honest about cost.** Every physics evaluation is a full simulation.
   `evaluate_batch` currently loops per individual; it is a bookkeeping path,
   not a throughput win. Do not promise CPU speed-ups from batching.
6. **Verify by running.** An integration is not done until the relevant tests,
   example or notebook actually execute.

## Step 1 — choose the search space

| Need | Use | Notes |
|---|---|---|
| Fixed crease pattern, vary fold angles | `SearchSpace.from_default_pattern()` / `SearchSpace.from_creases(...)` | Same semantics as the engine; default is 15-D |
| Fixed topology with normalized/design helpers | `DesignSpace.angles_only(pattern)` | Gives `base_vector()`, `to_normalized` / `from_normalized`, presets |
| Vary geometry, loop-edge angles, crease presence/count | `DesignSpace` builder / presets | Use the **evaluator**; not valid for RL |
| RL | `SearchSpace` or an **angle-only, unconstrained** `DesignSpace` | Geometry / variable-count spaces are rejected by `OrigamiEnv` |
| QD / behavior descriptors | `OrigamiEvaluator(...).evaluate()` | `DESCRIPTOR_NAMES` defines the descriptor keys |

Rules:

* Keep **one fixed topology** for RL and for evaluator caching coherence.
* `design_space.to_search_space()` is the bridge when the space is angle-only
  and unconstrained; otherwise it raises a clear `ValueError`.
* Use `space.repair(candidate)` on every vector from an external optimizer
  before evaluating or decoding it. Optimizers routinely emit `nan`, `inf` or
  out-of-bounds values.
* For scale-free step sizes, optimize in the normalized `[-1, 1]` view and map
  with `space.from_normalized(z)`.

```python
import numpy as np
from origami_engine_mjx.integrations import DesignSpace, OrigamiEvaluator

space = DesignSpace.angles_only(pattern, name="miura-2x2")  # 4-D demo
evaluator = OrigamiEvaluator(
    space,
    backend="jax",
    duration=0.01,
    fold_duration=0.0075,
    vibrate=False,
    maximize=True,
    error_policy="penalty",
)


def genome_from_z(z):
    return space.repair(space.from_normalized(np.asarray(z, dtype=float)))
```

## Step 2 — pin the objective contract

`OrigamiEvaluator` is the single source of truth:

* `evaluate(genome) -> EvalRecord` — one simulation plus descriptors.
* `evaluate_batch(X, return_details=False) -> BatchResult` — fitness-only fast
  path; still loops per individual.
* `fitness` is **higher-is-better**; `raw` is the projected displacement.
* `maximize=False` flips the sign in the record.
* `status` is `"ok"`, `"invalid"`, `"nonfinite"` or `"error"`. `error_policy="penalty"`
  keeps a failed genome from crashing a batch; inspect `status` / `errors` when
  something looks wrong.
* `use_cache=True` memoizes by genome/topology; disable it when you need fresh
  timing or stochastic physics.

Direction cheat sheet:

| Consumer | Direction | What to pass |
|---|---|---|
| `OrigamiEvaluator` | higher-is-better | `record.fitness` |
| `nevergrad`, `cma`, `pymoo` | minimize | `evaluator.to_minimizer()` / `to_pymoo_problem(evaluator)` |
| `evosax` | **minimize** | `-evaluator.evaluate_batch(X).fitness` |
| Gymnasium `OrigamiEnv` | reward (maximize return) | `env.step(action)` |
| LLM proposer | any ranking | feed back `fitness`; keep a best-so-far archive |

`to_evosax_fitness(evaluator)` returns the **higher-is-better** array; negate it
before `tell` because `evosax` minimizes `state.best_fitness`. This is the most
common integration bug.

## Step 3 — use or write a thin adapter

Prefer the existing adapters before writing glue:

```python
from origami_engine_mjx.integrations.adapters import (
    space_spec,  # JSON-friendly dim/bounds/names for prompts & logging
    to_nevergrad_function,  # minimization function
    to_pymoo_problem,  # pymoo Problem, xl/xu set, returns -fitness
    to_evosax_fitness,  # higher-is-better batch function (negate for tell)
)
```

An adapter must:

* expose the consumer's native signature and direction;
* call the evaluator, never reimplement physics;
* keep bounds and repair in the loop;
* fail clearly if the optional package is missing (`RuntimeError` with an
  install hint), never at import time of the library.

If you need a new adapter, put it in `origami_engine_mjx/integrations/adapters.py`
with a lazy import and a docstring that states the direction explicitly.

## Step 4 — seed the population

Population helpers are pure NumPy: they do not run the simulation, so they are
free to call before a budget starts.

```python
from origami_engine_mjx.integrations import (
    lhs_population,
    mixed_population,
    population_stats,
)

seeds = lhs_population(space, 8, rng=0)
population = mixed_population(
    space,
    16,
    seeds=seeds,
    fractions={"seed": 0.5, "interp": 0.25, "random": 0.25},
    rng=0,
)
print(population.genomes.shape, population.provenance[:3])
print(population_stats(population.genomes, space))
```

Guidance:

* Seed with LHS/Sobol for coverage, then mutate locally around the best.
* Keep provenance; it makes ablations and debugging possible.
* Run `population_stats` / `diversity_filter` before paying for evaluations.
* Seed the *sampler* (`rng=`, `np.random.default_rng`, `jax.random.key`); the
  physics itself is deterministic.

## Step 5 — log and checkpoint everything

```python
from origami_engine_mjx.integrations import (
    RunLogger,
    load_best,
    save_best,
    save_genome,
)

with RunLogger("runs/my_run.jsonl") as log:
    for genome, record in zip(genomes, records, strict=True):
        log.log(record, genome)  # one row per evaluation

path = save_best(best_genome, space, "runs/best.json", fitness=best_record.fitness, extra={"optimizer": "my_es"})
genome, meta = load_best(path, space)
save_genome(best_genome, space, "runs/best_pattern.json")
```

* `save_best` writes a normal crease-pattern JSON with extra `fitness` / `meta`
  keys, so downstream crease tools still load it.
* Keep `RunLogger` writes in the loop, not only at the end; interrupted runs
  should still be inspectable.
* Log the space name/`space_spec`, seed, backend, duration, optimizer version
  and objective mode in metadata or the run log.

## Step 6 — RL specifically

Use `OrigamiEnv` for Gymnasium trainers:

```python
from origami_engine_mjx.integrations.env import OrigamiEnv

env = OrigamiEnv(space=space, max_steps=100, frame_skip=1, vibrate=False)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

* Action modes: normalized `[-1, 1]` (default) or raw angles; `absolute` or
  `delta` (`max_delta`). Targets are always clipped to the space bounds.
* Reward default is the step change in +x centroid displacement.
* `terminated` only on non-finite physics; `truncated` at `max_steps`.
* Run `check_env(..., skip_render_check=True)`; do not add rendering to CI.
* For JAX-native training use `integrations.jax_env`: `build_jax_model` →
  `make_params` → `reset` → jitted `step` → `lax.scan`. JIT the step **once**;
  contacts are non-smooth, so do not rely on gradients through `step`.
* `OrigamiVectorEnv` is a GPU-oriented `vmap` path. Treat it as an API, not a
  CPU speed-up claim.

## Step 7 — LLM / agent proposer specifically

* Give the model `space_spec(space)` (dim, bounds, names), not a prose guess.
* Ask for JSON only; accept a list or `{"candidates": [...]}`.
* **Validate**: parse JSON, require shape `(n, dim)`, reject non-finite values,
  clip to `[-1, 1]`, `space.repair` before evaluation.
* Keep a deterministic mock proposer so the example runs offline without a key.
* Fall back to the mock on a rejected response; never let one bad completion
  kill a run.
* Feed back a compact state: best fitness/genome, last few scored candidates.
  Do not dump the entire history into the prompt.
* Keep a best-so-far archive based on real scores; a model can regress.
* Record model name/version/temperature/seed; LLM runs are not reproducible by
  default.

## Verification checklist

Before calling an integration done:

- [ ] The default 15-variable pattern is untouched and byte-identical:
      `DesignSpace.from_default_pattern().to_search_space() == SearchSpace.from_default_pattern()`.
- [ ] No optimizer/RL/LLM import appears in `origami_engine_mjx/` (only lazy,
      optional adapter imports).
- [ ] The example or notebook actually runs end to end on a clean process.
- [ ] Direction is explicit and tested (maximize vs minimize; `evosax` negation).
- [ ] Every external candidate passes through `space.repair` and respects bounds.
- [ ] Failures produce `status != "ok"` and do not crash the batch.
- [ ] `RunLogger` output and `save_best` / `load_best` round-trip were checked.
- [ ] Cost claims match reality (no batching speed-up claim; GPU-only paths labelled).
- [ ] `python scripts/check_docs_links.py` passes on new Markdown.
- [ ] `ruff check .` and `mypy` (per project config) pass if library code changed.
- [ ] Relevant tests run (`pytest -q tests_mjx -m "not slow"`, plus slow tests
      when physics changed).
- [ ] `CHANGELOG.md` has an entry; README edits, if any, are limited to the
      sections explicitly allowed by the project.

## Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Optimizer drives fitness to `-inf` | Maximizing a minimizer or vice versa | Check the direction table; `evosax` minimizes |
| All candidates identical after round 1 | Step size collapsed or clipping at a bound | Work in normalized space; keep exploration seeds |
| `ValueError: pattern has no actuated creases to vary` | A loop with no spanning-tree hinge | Use a pattern with real actuated creases (e.g. a Miura patch) |
| Geometry / variable-count space passed to `OrigamiEnv` | RL requires one fixed topology | Use `SearchSpace` or angle-only `DesignSpace` |
| `KeyError`/`AttributeError` in an `evosax` strategy | `init` signature differs | Dispatch on `population` / `means`; provide `sampling_fn` for `RandomSearch` |
| Notebook dies on a CPU runtime | Real physics budget or a GPU-only path | Use the surrogate switch / tiny pattern / GPU runtime |
| `Failed to import warp` | `warp-lang` is absent | Expected; integrations use MJX-JAX |
| `save_best` file will not open in another tool | It is JSON, not XML/MJCF | It is a crease-pattern JSON; use the crease loader |

## Where the pieces live

* Spaces: `docs/SEARCH_SPACE.md`, `origami_engine_mjx/integrations/search_space.py`,
  `design_space.py`
* Objective: `docs/OPTIMIZATION_ES.md`, `docs/EVALUATION.md`,
  `integrations/evaluator.py`
* Adapters: `integrations/adapters.py`
* Populations / logging: `docs/POPULATIONS.md`, `integrations/population.py`,
  `integrations/plumbing.py`
* RL: `docs/RL.md`, `integrations/env.py`, `integrations/jax_env.py`
* Notebooks: [`notebook_examples/`](../notebook_examples/README.md)
* Runnable examples: [`examples/04_optimize_es.py`](../examples/04_optimize_es.py),
  [`examples/05_rl_env.py`](../examples/05_rl_env.py),
  [`examples/06_design_space.py`](../examples/06_design_space.py),
  [`examples/07_populations.py`](../examples/07_populations.py)
* Colab/runtime guidance: `docs/COLAB_GPU.md`
