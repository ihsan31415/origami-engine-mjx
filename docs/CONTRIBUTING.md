# Contributing

> **Documentation map:** [README](../README.md) · [Getting Started](GETTING_STARTED.md) · [Tutorials](TUTORIALS.md) · [Crease Format](CREASE_FORMAT.md) · [Simulation](SIMULATION.md) · [Evaluation](EVALUATION.md) · [API Reference](API_REFERENCE.md) · [Architecture](ARCHITECTURE.md) · [Troubleshooting](TROUBLESHOOTING.md) · [Glossary](GLOSSARY.md) · [Integrations](INTEGRATIONS.md) · [Optimization (ES)](OPTIMIZATION_ES.md) · [RL](RL.md) · **Contributing**

Thanks for your interest in contributing! You can help with code, tests, documentation, example crease patterns, or bug reports.

## Development setup

Clone the repository and install in **editable** mode with the dev extras:

```bash
git clone https://github.com/ihsan31415/origami-engine-mjx.git
cd origami-engine-mjx

# with pip
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,jax,warp]"

# or with uv (equivalent, faster)
uv venv .venv && . .venv/bin/activate
uv pip install -e ".[dev,jax,warp]"
```

`[dev]` adds `pytest`, `pytest-cov`, `ruff`, `mypy`, and `pre-commit`; `[jax]` adds the MJX-JAX backend; `[warp]` adds the MJX-Warp backend (requires CUDA GPU); `[opt]` adds the optimizer adapters (`pymoo`, `nevergrad`, `cma`, `evosax`) and `[rl]` adds `gymnasium`. The `[opt]`/`[rl]` extras are only needed for the integration tests and examples — the core package never imports them.

## Everyday commands (Makefile)

| Command | What it runs |
|---|---|
| `make help` | List all targets |
| `make venv` / `make install` | Create `.venv` / install package + dev deps editable |
| `make test` | `pytest -q` (uses `testpaths = ["tests_mjx"]`; coverage is not enabled by default) |
| `make examples` | Run every `examples/0*.py` script |
| `make lint` | `ruff check .` + `ruff format --check .` |
| `make format` | `ruff format .` + `ruff check --fix .` |
| `make typecheck` | `mypy origami_engine_mjx --config-file pyproject.toml` |
| `make docs-check` | `python scripts/check_docs_links.py` (Markdown links + heading anchors) |
| `make clean` | Remove generated artifacts and caches |

Without make, the equivalents are plain commands, e.g. `python -m pytest -q`, `python -m ruff check .`.

## Running the examples

Examples use `sys.path` bootstrap so they run from a source checkout:

```bash
. .venv/bin/activate
pip install -e ".[dev,jax]"  # warp optional if no GPU
python examples/01_evaluate_cpu_jax.py
# or all of them:
make examples
```

**Note:** on a PyPI wheel install without editable mode, `examples/` is not included. Use `load_default_creases()` and the tutorials instead.

## Pre-commit hooks

Install once per clone; hooks then run on every commit:

```bash
pre-commit install
```

Configured hooks (`.pre-commit-config.yaml`): `ruff --fix`, `ruff-format`, and `mypy` (with `files: ^origami_engine_mjx/`).

## Code style & type checking

- **ruff** with `line-length = 120`, `target-version = "py310"` (`pyproject.toml [tool.ruff]`).
- **mypy** with `files = ["origami_engine_mjx"]`, `python_version = "3.10"`.
- Prefer the package-root imports in new code (`from origami_engine_mjx import ...`).
- Public functions should raise `ValueError`/`TypeError` with messages that name the offending index/key (see `serialization.py` for the pattern).

## Tests

- Suite: `tests_mjx/` with fixtures in `tests_mjx/conftest.py` (`tiny_creases`).
- Config: `testpaths = ["tests_mjx"]`, `python_files = ["test_*.py"]`, and a `slow` marker. `pytest-cov` is installed via `[dev]`, so run `pytest --cov=origami_engine_mjx` if you want a coverage report.
- Add or update tests for any new behavior: input validation, JSON round-trips, pattern validity reasons, evaluation semantics (order preservation, `error_policy`), and a smoke run with short `duration`.
- Integration interfaces are covered by `tests_mjx/test_integrations.py` (search space, evaluator, adapters, plumbing) and `tests_mjx/test_rl_env.py` (Gymnasium env, vector env). The simulation-backed cases there are marked `slow`; the rest are fast and run with `pytest -q`.
- Run `make lint && make typecheck && make test` before opening a PR.

**CPU tests note:** MJX-JAX on CPU is slow for mesh-contact models (~1.4 s/step for 2-face mini models; ~70 s/step for the 16-face reference model). The test suite uses tiny 2-face patterns; tests marked `@pytest.mark.slow` are skipped by default (`pytest -q` excludes them) and require `pytest -q -m slow` on a faster machine or GPU.

## Continuous integration

`.github/workflows/ci.yml` runs on push/PR to `main`:

1. **lint-and-test** — matrix over Python 3.10 and 3.11: `ruff check .`, `ruff format --check .`, `mypy origami_engine_mjx --config-file pyproject.toml`, `pytest -q` (after `pip install -e ".[dev]"`; JAX/MJX tests use `pytest.importorskip`, so they are skipped when the optional backend is not installed).
2. **publish-to-pypi** — builds the sdist/wheel with `python -m build` and publishes via `pypa/gh-action-pypi-publish` (PyPI Trusted Publishing, OIDC).

Note: The publish job only runs on pushes to `main` and requires Docker image pre-pull.

## Submitting changes

1. Fork the repository and create a focused branch: `feat/...`, `fix/...`, `docs/...`, `chore/...`.
2. Make your changes in small, reviewable commits.
3. Add/update tests and documentation for any behavior change.
4. Run the quality gates locally (`make lint`, `make typecheck`, `make test`).
5. Open a pull request against `main` with a clear description and, for behavior changes, a minimal repro or example.

## Documentation contributions

- The documentation set lives in `docs/` (`README.md` index, `GETTING_STARTED.md`, `TUTORIALS.md`, `CREASE_FORMAT.md`, `SIMULATION.md`, `EVALUATION.md`, `API_REFERENCE.md`, `ARCHITECTURE.md`, `TROUBLESHOOTING.md`, `GLOSSARY.md`, `CONTRIBUTING.md`, `BACKENDS.md`, `COLAB_GPU.md`, `MIGRATION.md`, `INTEGRATIONS.md`, `OPTIMIZATION_ES.md`, `RL.md`) plus `examples/README.md` and `README.md`/`CHANGELOG.md` at the repo root.
- Link to docs with the bare filename as the link text (e.g. `[TUTORIALS.md](TUTORIALS.md)`) — never a path prefix inside the button.
- Keep the **documentation map** nav line at the top of each doc in `docs/` intact and add new pages to it, so every `.md` stays traceable from `README.md`.
- Keep docs accurate to the code: signatures in [API_REFERENCE.md](API_REFERENCE.md) must match the source; run the examples when you change behavior.

## Project layout

```
origami-engine-mjx/
├── docs/                    # Documentation (see above)
├── examples/                # Runnable demo scripts
│   ├── 01_evaluate_cpu_jax.py
│   ├── 02_evaluate_warp_gpu.py
│   ├── 03_batch_population.py
│   ├── 04_optimize_es.py        # black-box optimization demo
│   ├── 05_rl_env.py             # Gymnasium / JAX-functional env demo
│   └── colab_gpu_validation.py  # GPU benchmark (Colab T4)
├── origami_engine_mjx/      # MJX dual-backend package
│   ├── backends/            # MJX-JAX and MJX-Warp implementations
│   ├── integrations/        # ES/RL/AI interfaces (no algorithms)
│   ├── crease_patterns/     # Packaged reference Miura JSON
│   ├── schemas/             # JSON Schema for crease input
│   ├── __init__.py          # Public exports
│   ├── constants.py         # Physical / solver constants
│   ├── types.py             # Dataclasses (Crease, Pattern, SimulationResult, PopulationResult)
│   ├── geometry.py          # build_pattern (shared)
│   ├── model.py             # generate_xml (shared)
│   ├── serialization.py     # JSON I/O (shared)
│   ├── patterns.py          # Reference pattern loaders (shared)
│   ├── plotting.py          # plot_pattern (shared)
│   ├── backend.py           # Import guards
│   ├── engine.py            # OrigamiEngine facade
│   └── evaluation.py        # evaluate_individual, evaluate_population
├── tests_mjx/               # Test suite
├── crease_patterns/         # Repo-root copy of reference Miura (byte-identical to packaged)
├── pyproject.toml           # Project config
├── Makefile                 # Common targets
└── README.md                # Project overview (documentation links at bottom)
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the module responsibilities and data flow.

## Code of conduct

Be respectful and constructive in issues and reviews. Assume good intent, give actionable feedback, and keep discussions focused on the change.