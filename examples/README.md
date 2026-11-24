# Examples

> **Documentation map:** [README](../README.md) · [GETTING_STARTED.md](../docs/GETTING_STARTED.md) · [TUTORIALS.md](../docs/TUTORIALS.md) · [API_REFERENCE.md](../docs/API_REFERENCE.md) · [EVALUATION.md](../docs/EVALUATION.md) · [CREASE_FORMAT.md](../docs/CREASE_FORMAT.md) · [SIMULATION.md](../docs/SIMULATION.md) · [TROUBLESHOOTING.md](../docs/TROUBLESHOOTING.md) · [GLOSSARY.md](../docs/GLOSSARY.md) · [INTEGRATIONS.md](../docs/INTEGRATIONS.md) · [OPTIMIZATION_ES.md](../docs/OPTIMIZATION_ES.md) · [SEARCH_SPACE.md](../docs/SEARCH_SPACE.md) · [POPULATIONS.md](../docs/POPULATIONS.md) · [RL.md](../docs/RL.md) · [CONTRIBUTING.md](../docs/CONTRIBUTING.md)

Runnable scripts showing how to use the `origami_engine_mjx` library. Each script inserts the repository root onto `sys.path`, so they run directly from a source checkout (with the package installed editable, `pip install -e ".[dev]"` — see [GETTING_STARTED.md](../docs/GETTING_STARTED.md), Method 3):

```bash
. .venv/bin/activate
python examples/01_evaluate_cpu_jax.py
# or all of them:
make examples
```

**Note:** the `sys.path` bootstrap means these scripts are meant to run **from a source checkout**. If you installed the package from PyPI without dev extras, use the tutorials instead — `import origami_engine_mjx` works anywhere, and the [TUTORIALS.md](../docs/TUTORIALS.md) show the equivalent snippets.

## The scripts

| Script | What it demonstrates | Key APIs | Output |
|---|---|---|---|
| [`01_evaluate_cpu_jax.py`](01_evaluate_cpu_jax.py) | Single simulation + population evaluation on CPU using MJX-JAX | `OrigamiEngine` (`evaluate`, `evaluate_population`) | Prints forward displacement (m) and optionally scores |
| [`02_evaluate_warp_gpu.py`](02_evaluate_warp_gpu.py) | MJX-Warp single simulation on NVIDIA GPU | `OrigamiEngine, load_default_creases` | Prints forward displacement (warp/GPU) |
| [`03_batch_population.py`](03_batch_population.py) | Batched population evaluation with configurable batch size | `OrigamiEngine, evaluate_population` | One score per individual |
| [`04_optimize_es.py`](04_optimize_es.py) | Black-box optimization over the fold-angle search space with `nevergrad` (falls back to random search) | `SearchSpace, OrigamiEvaluator, to_nevergrad_function` | Best genome, objective, raw displacement |
| [`05_rl_env.py`](05_rl_env.py) | Gymnasium `OrigamiEnv` rollout + pure-JAX `reset`/`step` | `OrigamiEnv, jax_env.reset/step` | Per-step rewards and episode return |
| [`06_design_space.py`](06_design_space.py) | Configurable `DesignSpace`: presets, selection, constraints, presence, JSON, per-individual eval path | `DesignSpace, DesignSpaceBuilder, topology_signature, OrigamiEvaluator` | Dimensions, selections, paths |
| [`07_populations.py`](07_populations.py) | Population seeding: seeds, interpolation, random, mixed, dedup, diversity, I/O, stats | `load_seeds, mixed_population, diversity_filter, save_population, population_stats` | Population shapes and provenance |
| [`colab_gpu_validation.py`](colab_gpu_validation.py) | End-to-end MJX-Warp validation + benchmark on a Colab T4 | `OrigamiEngine, evaluate_population` | Timing and throughput table |

## How each maps to the tutorials

| Example | Tutorial in [TUTORIALS.md](../docs/TUTORIALS.md) | Deeper doc |
|---|---|---|
| 01 | Tutorial 2–3 — run & score a pattern | [EVALUATION.md](../docs/EVALUATION.md) |
| 02 | Tutorial 2 — run on GPU (Warp) | [BACKENDS.md](../docs/BACKENDS.md) |
| 03 | Tutorial 6 — population evaluation | [EVALUATION.md](../docs/EVALUATION.md) |
| 04 | — (optimization) | [OPTIMIZATION_ES.md](../docs/OPTIMIZATION_ES.md) |
| 05 | — (reinforcement learning) | [RL.md](../docs/RL.md) |
| 06 | — (search-space configuration) | [SEARCH_SPACE.md](../docs/SEARCH_SPACE.md) |
| 07 | — (population seeding) | [POPULATIONS.md](../docs/POPULATIONS.md) |
| colab | Tutorial 2 — GPU quick start | [COLAB_GPU.md](../docs/COLAB_GPU.md) |

## Related entry points

- `make example` → runs the quick-start script `examples/01_evaluate_cpu_jax.py` (CPU).
- `make examples` → runs every `examples/0*.py` script sequentially (the GPU/Warp one requires a CUDA device).
- `make lint` → `ruff check .` + `ruff format --check .`.
- `make format` → `ruff format .` + `ruff check --fix .`.
- `make typecheck` → `mypy origami_engine_mjx --config-file pyproject.toml`.
- `make docs-check` → `python scripts/check_docs_links.py` (verifies Markdown links and heading anchors).

## See also

- [GETTING_STARTED.md](../docs/GETTING_STARTED.md) — installation methods (pip / uv / source)
- [TUTORIALS.md](../docs/TUTORIALS.md) — eight step-by-step recipes
- [API_REFERENCE.md](../docs/API_REFERENCE.md) — full API signatures
- [EVALUATION.md](../docs/EVALUATION.md) — fitness functions and population evaluation
- [OPTIMIZATION_ES.md](../docs/OPTIMIZATION_ES.md) — black-box optimization over the search space
- [SEARCH_SPACE.md](../docs/SEARCH_SPACE.md) — the configurable `DesignSpace`
- [POPULATIONS.md](../docs/POPULATIONS.md) — population seeding and initialization
- [RL.md](../docs/RL.md) — Gymnasium and JAX-functional environments
- [CREASE_FORMAT.md](../docs/CREASE_FORMAT.md) — crease & JSON format reference