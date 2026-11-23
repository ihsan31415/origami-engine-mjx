# Integrations: ES / RL / AI interfaces

> **Documentation map:** [README](../README.md) · [Docs index](README.md) · [Getting Started](GETTING_STARTED.md) · [Tutorials](TUTORIALS.md) · [Crease Format](CREASE_FORMAT.md) · [Simulation](SIMULATION.md) · [Evaluation](EVALUATION.md) · [API Reference](API_REFERENCE.md) · [Architecture](ARCHITECTURE.md) · [Backends](BACKENDS.md) · **Integrations** · [Optimization (ES)](OPTIMIZATION_ES.md) · [RL](RL.md) · [Troubleshooting](TROUBLESHOOTING.md) · [Glossary](GLOSSARY.md) · [Contributing](CONTRIBUTING.md)

This document is the overview of the `origami_engine_mjx.integrations`
subpackage. It explains what the package is for, the design boundary it keeps,
and how the pieces fit together. Deeper guides:

- [OPTIMIZATION_ES.md](OPTIMIZATION_ES.md) — search space, evaluator, optimizer
  adapters, quality-diversity descriptors.
- [RL.md](RL.md) — Gymnasium environments and the pure-JAX `reset`/`step`.

## What this package is (and is not)

`origami-engine-mjx` is the **environment / evaluator substrate** for external
search and learning libraries. The `integrations` subpackage defines the small,
stable interfaces those libraries consume:

- a description of the **design space** (`SearchSpace`),
- a **black-box objective** (`OrigamiEvaluator`) that maps a genome to a
  simulation score,
- thin **adapters** that reshape that objective into a library's calling
  convention,
- **environments** for reinforcement learning (Gymnasium and JAX-functional),
- **plumbing** (seeding, genome/checkpoint I/O, run logging, benchmarking).

It deliberately ships **no** optimization or learning algorithm. Gradient-free
optimizers (CMA-ES, evolution strategies), gradient-based RL trainers
(PPO/SAC), and quality-diversity methods (MAP-Elites) all live in their own
packages; this library only lets them drive the folding simulation.

> **Boundary rule:** every third-party dependency of the integration layer is
> optional. The core package (`constants`, `geometry`, `model`,
> `serialization`, `patterns`, `engine`, `evaluation`) never imports
> `gymnasium`, `pymoo`, `nevergrad`, `cma` or `evosax`.

## Installation

The interfaces are always importable; the *optional* pieces need extras:

```bash
pip install 'origami-engine-mjx[jax]'   # MJX-JAX backend (needed by env + evaluator on CPU)
pip install 'origami-engine-mjx[opt]'   # pymoo, nevergrad, cma, evosax adapters
pip install 'origami-engine-mjx[rl]'    # gymnasium environments
pip install 'origami-engine-mjx[all]'   # everything above + dev + warp
```

Third-party imports are lazy, so `import origami_engine_mjx.integrations`
succeeds even when none of the optional libraries are installed. A missing
library raises a `RuntimeError` (or `ImportError`) only when you call the
function that needs it, with an install hint.

## Module map

| Module | Public names | Purpose |
|---|---|---|
| `integrations/search_space.py` | `SearchSpace` | Fixed-topology fold-angle genome: encode/decode, bounds, sampling, repair, JSON description |
| `integrations/evaluator.py` | `OrigamiEvaluator`, `EvalRecord`, `BatchResult`, `DESCRIPTOR_NAMES` | Stateless, picklable objective + behavior descriptors |
| `integrations/adapters.py` | `to_pymoo_problem`, `to_nevergrad_function`, `to_evosax_fitness` | Reshape the evaluator to a library's API |
| `integrations/env.py` | `OrigamiEnv`, `OrigamiVectorEnv` | Gymnasium single + vectorized environments |
| `integrations/jax_env.py` | `reset`, `step`, `observe`, `build_jax_model`, `make_params`, `EnvState`, `EnvParams` | Pure JAX functions (`jit`/`vmap`/`scan`) |
| `integrations/plumbing.py` | `make_rng`, `set_global_seed`, `save_genome`, `load_genome`, `save_best`, `load_best`, `RunLogger`, `benchmark` | Reproducibility, I/O, logging, throughput |

All of these are re-exported from `origami_engine_mjx.integrations`; the
environment symbols are exposed lazily so a NumPy-only user never imports JAX or
Gymnasium.

## The design space in one paragraph

For the current geometry backend the crease **topology is fixed**: the segment
endpoints (`p1`, `p2`) of every crease are held constant and only the signed
fold **angles** vary. This is exactly what `build_pattern` and `generate_xml`
support — the face graph is reconstructed from the segment geometry, so moving
endpoints can change the number of faces and hinges, which the simulators cannot
batch over. Only creases that become **spanning-tree hinges** are actuated;
loop-edge creases become equality `connect` constraints and their angle is
ignored by the dynamics (verified: changing a loop-crease angle leaves the
generated XML and the hinge targets byte-identical). With the default
`active_only=True`, the packaged 4×4 Miura-ori reference pattern therefore has a
**15-dimensional** genome (24 creases − 9 loop edges). See
[OPTIMIZATION_ES.md](OPTIMIZATION_ES.md) for the full description.

## Minimal end-to-end example

```python
from origami_engine_mjx.integrations import SearchSpace, OrigamiEvaluator

space = SearchSpace.from_default_pattern()  # 15-d genome
evaluator = OrigamiEvaluator(space, backend="jax")  # no algorithm

genome = space.sample(0)  # seeded sampling
record = evaluator.evaluate(genome)  # EvalRecord
print(record.fitness, record.status, record.descriptors)

batch = evaluator.evaluate_batch(space.sample(0, size=8))
print(batch.fitness, batch.best_index())
```

From there, hand `evaluator.to_minimizer()` to `nevergrad`/`cma`, or
`to_pymoo_problem(evaluator)` to `pymoo`, or wrap the environment in
`gym.make`-style training. See the two deeper guides and
[`examples/04_optimize_es.py`](../examples/04_optimize_es.py) /
[`examples/05_rl_env.py`](../examples/05_rl_env.py).

## Determinism and reproducibility

- The **physics is deterministic**: identical creases, `duration`, vibration
  settings and backend give byte-identical scores (verified on the JAX backend
  by evaluating the same genome twice). There is no hidden RNG in the
  simulation.
- The only stochastic parts of a pipeline are **genome sampling** and optional
  **environment randomization**. Both take a seed: `SearchSpace.sample(rng)` /
  `make_rng(seed)` / `set_global_seed(seed)`, and `OrigamiEnv.reset(seed=...)`.
- `set_global_seed` seeds Python `random` and NumPy. JAX has no global RNG, so
  JAX code takes an explicit `jax.random.key(seed)`. It does not change the
  deterministic physics.

## Performance expectations

The integration layer adds **no** physics cost of its own; every evaluation is
exactly one `OrigamiEngine` run.

- **MJX-JAX on CPU is slow for mesh-contact models** — the README's measured
  numbers are ~70 s/step for the 16-face reference model and ~1.4 s/step for a
  tiny 2-face model. Use CPU for correctness tests with short `duration`.
- The evaluator's **fast batch path** (`evaluate_batch(..., return_details=False)`)
  delegates to `OrigamiEngine.evaluate_population`, which currently loops
  per individual (no vmap, `batch_size` ignored — see
  [EVALUATION.md](EVALUATION.md)). It is not faster than the structured path on
  the built-in backends; it only skips descriptor bookkeeping.
- The Gymnasium **vector environment** (`OrigamiVectorEnv`) *does* use
  `jax.vmap` over `jax_env.step`, so `N` environments share one compiled model
  and step together. Compiling that step is expensive; it is intended for GPU
  workloads.
- GPU-only claims (Warp throughput, T4 timings) are **not verified in this
  repository** on CPU-only machines. Treat them as expectations, not measured
  facts, until reproduced.

## Relationship to the rest of the package

- The evaluator is a thin layer over `OrigamiEngine.run` and
  `compute_forward_displacement`; it does not change their behaviour.
- The environments are built directly on the backend internals
  (`build_model` / `make_state` / `mjx.step`) because there is no per-step
  public backend API. The core modules are unchanged.
- Nothing in `integrations/` is imported by the core package, so the public API
  and its signatures are unaffected.

## See also

- [OPTIMIZATION_ES.md](OPTIMIZATION_ES.md) — the optimization guide
- [RL.md](RL.md) — the reinforcement-learning guide
- [EVALUATION.md](EVALUATION.md) — the underlying fitness signal
- [ARCHITECTURE.md](ARCHITECTURE.md) — where `integrations/` sits in the package
- [GLOSSARY.md](GLOSSARY.md) — search space, genome, descriptor, episode
