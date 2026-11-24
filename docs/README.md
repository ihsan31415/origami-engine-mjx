# Documentation index

> **Documentation map:** **Docs index** · [README](../README.md) · [Getting Started](GETTING_STARTED.md) · [Tutorials](TUTORIALS.md) · [Crease Format](CREASE_FORMAT.md) · [Simulation](SIMULATION.md) · [Evaluation](EVALUATION.md) · [API Reference](API_REFERENCE.md) · [Architecture](ARCHITECTURE.md) · [Backends](BACKENDS.md) · [Integrations](INTEGRATIONS.md) · [Optimization (ES)](OPTIMIZATION_ES.md) · [Search space](SEARCH_SPACE.md) · [Populations](POPULATIONS.md) · [RL](RL.md) · [Troubleshooting](TROUBLESHOOTING.md) · [Glossary](GLOSSARY.md) · [Contributing](CONTRIBUTING.md)

This directory holds the long-form documentation for `origami-engine-mjx`. The
[top-level README](../README.md) is the entry point; everything below expands a
part of it.

## Start here

| Document | What it covers |
|---|---|
| [GETTING_STARTED.md](GETTING_STARTED.md) | Install (pip / uv / source), first simulation, first evaluation |
| [TUTORIALS.md](TUTORIALS.md) | Step-by-step recipes from a single run to population evaluation |

## Core concepts

| Document | What it covers |
|---|---|
| [CREASE_FORMAT.md](CREASE_FORMAT.md) | `p1` / `p2` / `angle` data and the JSON schema |
| [SIMULATION.md](SIMULATION.md) | Model structure, stepping loop, rendering |
| [EVALUATION.md](EVALUATION.md) | Forward displacement, fitness, population evaluation |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Package layout, data flow, backend responsibilities |
| [BACKENDS.md](BACKENDS.md) | MJX-JAX vs MJX-Warp: semantics, features, limitations |

## Optimization & learning interfaces

| Document | What it covers |
|---|---|
| [INTEGRATIONS.md](INTEGRATIONS.md) | The `origami_engine_mjx.integrations` package at a glance |
| [OPTIMIZATION_ES.md](OPTIMIZATION_ES.md) | Search space, evaluator, optimizer adapters, QD descriptors |
| [SEARCH_SPACE.md](SEARCH_SPACE.md) | Configurable `DesignSpace`: angles, geometry, crease count, constraints |
| [POPULATIONS.md](POPULATIONS.md) | Population seeding: seeds, interpolation, random, mixed, I/O |
| [RL.md](RL.md) | Gymnasium `OrigamiEnv`, the vector env, and pure-JAX `reset`/`step` |

## Reference & operations

| Document | What it covers |
|---|---|
| [API_REFERENCE.md](API_REFERENCE.md) | Complete API signatures |
| [COLAB_GPU.md](COLAB_GPU.md) | Colab T4 setup, validation checklist, benchmarking |
| [MIGRATION.md](MIGRATION.md) | Migrating from `origami-engine` (classic MuJoCo) |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Runtime problems and guidance |
| [GLOSSARY.md](GLOSSARY.md) | Terminology used across the codebase |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute, tests, CI |

## Runnable examples

The [`examples/`](../examples/README.md) directory contains scripts that mirror
these docs:

| Script | Doc |
|---|---|
| [`01_evaluate_cpu_jax.py`](../examples/01_evaluate_cpu_jax.py) | [EVALUATION.md](EVALUATION.md) |
| [`02_evaluate_warp_gpu.py`](../examples/02_evaluate_warp_gpu.py) | [BACKENDS.md](BACKENDS.md) |
| [`03_batch_population.py`](../examples/03_batch_population.py) | [EVALUATION.md](EVALUATION.md) |
| [`04_optimize_es.py`](../examples/04_optimize_es.py) | [OPTIMIZATION_ES.md](OPTIMIZATION_ES.md) |
| [`05_rl_env.py`](../examples/05_rl_env.py) | [RL.md](RL.md) |
| [`06_design_space.py`](../examples/06_design_space.py) | [SEARCH_SPACE.md](SEARCH_SPACE.md) |
| [`07_populations.py`](../examples/07_populations.py) | [POPULATIONS.md](POPULATIONS.md) |
| [`colab_gpu_validation.py`](../examples/colab_gpu_validation.py) | [COLAB_GPU.md](COLAB_GPU.md) |

## See also

- [README.md](../README.md) — project overview and quick start
- [CHANGELOG.md](../CHANGELOG.md) — release history
