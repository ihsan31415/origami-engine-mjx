# Examples

> **Documentation map:** [README](../README.md) · [GETTING_STARTED.md](../docs/GETTING_STARTED.md) · [TUTORIALS.md](../docs/TUTORIALS.md) · [API_REFERENCE.md](../docs/API_REFERENCE.md) · [EVALUATION.md](../docs/EVALUATION.md) · [CREASE_FORMAT.md](../docs/CREASE_FORMAT.md) · [SIMULATION.md](../docs/SIMULATION.md) · [TROUBLESHOOTING.md](../docs/TROUBLESHOOTING.md) · [GLOSSARY.md](../docs/GLOSSARY.md) · [CONTRIBUTING.md](../docs/CONTRIBUTING.md)

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
| [`01_evaluate_cpu_jax.py`](01_evaluate_cpu_jax.py) | Single simulation + population evaluation on CPU using MJX-JAX | `OrigamiEngine, evaluate_individual, evaluate_population` | Prints forward displacement (m) and optionally scores |
| [`02_evaluate_warp_gpu.py`](02_evaluate_warp_gpu.py) | MJX-Warp single simulation on NVIDIA GPU | `OrigamiEngine, load_default_creases` | Prints forward displacement (warp/GPU) |
| [`03_batch_population.py`](03_batch_population.py) | Batched population evaluation with configurable batch size | `OrigamiEngine, evaluate_population` | One score per individual |
| [`04_evaluate_population.py`](04_evaluate_population.py) | Population evaluation with `error_policy` (raise / nan) | `evaluate_population` | Best fitness score, optionally ranked |
| [`05_plot_pattern.py`](05_plot_pattern.py) | Plot the default pattern headlessly (Agg backend) and save a PNG | `plot_pattern, build_pattern, load_default_creases` | Writes `examples/05_plot_pattern.png` (gitignored) |
| [`02_json_input.py`](02_json_input.py) | Save creases to JSON and load them back (schema-validated round-trip), then evaluate the loaded pattern | `save_creases_to_json, load_creases_from_json, evaluate_individual` | Prints file size, loaded crease count/first angle, score |
| [`03_inspect_pattern.py`](03_inspect_pattern.py) | Reconstruct and inspect the face graph without simulating (no MuJoCo needed for `build_pattern`) | `build_pattern, load_default_creases / DEFAULT_PATTERN_JSON` | Prints validity, face/hinge/loop-edge counts, root face, tree depth |

## How each maps to the tutorials

| Example | Tutorial in [TUTORIALS.md](../docs/TUTORIALS.md) | Deeper doc |
|---|---|---|
| 01 | Tutorial 3 — score a pattern | [EVALUATION.md](../docs/EVALUATION.md) |
| 02 | Tutorial 4 — JSON round-trip | [CREASE_FORMAT.md](../docs/CREASE_FORMAT.md) |
| 03 | Tutorial 1 — inspect a pattern | [CREASE_FORMAT.md](../docs/CREASE_FORMAT.md) |
| 04 | Tutorial 6 — population evaluation | [EVALUATION.md](../docs/EVALUATION.md) |
| 05 | Tutorial 7 — plot & save PNG | [SIMULATION.md](../docs/SIMULATION.md) (rendering) |
| 02 JSON | Tutorial 4 — JSON round-trip | [CREASE_FORMAT.md](../docs/CREASE_FORMAT.md) |
| 03 inspect | Tutorial 1 — inspect a pattern | [CREASE_FORMAT.md](../docs/CREASE_FORMAT.md) |

## Related entry points

- `make example` → runs every `examples/0*.py` script sequentially.
- `make lint` → `ruff check .` + `ruff format --check .`.
- `make format` → `ruff format .` + `ruff check --fix .`.
- `make typecheck` → `mypy origami_engine_mjx --config-file pyproject.toml`.

## See also

- [GETTING_STARTED.md](../docs/GETTING_STARTED.md) — installation methods (pip / uv / source)
- [TUTORIALS.md](../docs/TUTORIALS.md) — nine step-by-step recipes
- [API_REFERENCE.md](../docs/API_REFERENCE.md) — full API signatures
- [EVALUATION.md](../docs/EVALUATION.md) — fitness functions and population evaluation
- [CREASE_FORMAT.md](../docs/CREASE_FORMAT.md) — crease & JSON format reference