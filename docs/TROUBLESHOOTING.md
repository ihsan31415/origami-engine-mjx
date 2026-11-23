# Troubleshooting

> **Documentation map:** [README](../README.md) · [Getting Started](GETTING_STARTED.md) · [Tutorials](TUTORIALS.md) · [Crease Format](CREASE_FORMAT.md) · [Simulation](SIMULATION.md) · [Evaluation](EVALUATION.md) · [API Reference](API_REFERENCE.md) · [Architecture](ARCHITECTURE.md) · **Troubleshooting** · [Glossary](GLOSSARY.md) · [Integrations](INTEGRATIONS.md) · [Optimization (ES)](OPTIMIZATION_ES.md) · [Search space](SEARCH_SPACE.md) · [Populations](POPULATIONS.md) · [RL](RL.md) · [Contributing](CONTRIBUTING.md)

Common problems and their fixes, grouped by stage.

## Installation

**`ModuleNotFoundError: No module named 'origami_engine_mjx'`** — the package is not installed in the interpreter you are using. Check `which python` / `python -c "import sys; print(sys.executable)"` and reinstall into that environment (`pip install origami-engine-mjx` or `pip install -e ".[dev]"` from a checkout).

**`ERROR: Package 'origami-engine-mjx' requires a different Python**`** — the package needs Python ≥ 3.10 (CI tests 3.10 and 3.11). Create the environment with a supported interpreter: `python3.11 -m venv .venv` or `uv venv --python 3.11`.

**`uv pip install` says "No virtual environment found"** — `uv pip` targets the *active* environment. Either activate one (`. .venv/bin/activate`) or use project mode (`uv add origami-engine-mjx` + `uv run ...`).

## Import errors

**`ModuleNotFoundError: No module named 'mujoco'`** — MuJoCo ships manylinux/macOS/windows wheels; on other platforms install fails at pip time. On headless Linux servers the import works but **rendering** needs a GL backend — see below.

**`RuntimeError: MuJoCo is required for this operation.`** — expected behavior of the import guard (`backend.py`) when `mujoco` is not installed. Pure-geometry functions (`build_pattern`, `normalize_creases`, `load_creases_from_json`, `plot_pattern`, constants) still work without it.

**`RuntimeError: MJX-JAX backend requires 'mujoco-mjx' and 'jax'.**`**** — install with `pip install 'origami-engine-mjx[jax]'`. Same for Warp: `pip install 'origami-engine-mjx[warp]'`.

## Rendering and video

`origami-engine-mjx` ships **no renderer and no CLI**. MJX-JAX/MJX-Warp models cannot be drawn directly by `mujoco.Renderer`; copy state back to classic MuJoCo (or keep using the legacy `origami-engine` renderer), which needs OpenGL. On a server without a display, set one of:

```bash
export MUJOCO_GL=osmesa    # software rendering (pip install osmesa is not needed; mujoco bundles it)
# or
export MUJOCO_GL=egl       # GPU servers with EGL
export MUJOCO_GL=disable   # skip rendering entirely (simulation still runs)
```

`plot_pattern` uses Matplotlib; on headless machines use the Agg backend (`matplotlib.use("Agg")`) and save with `plt.gcf().savefig(...)` instead of relying on `plt.show()` (see [TUTORIALS.md](TUTORIALS.md), Tutorial 7).

## Invalid crease patterns

`build_pattern` returns `pat.valid == False` (and `engine.run` raises `ValueError("invalid crease pattern: {reason}")`) for:

| `reason` | Cause | Fix |
|---|---|---|
| `no faces` | Creases don't partition the sheet | Add creases that connect boundary to boundary or form closed loops |
| `sliver face (area …)` | A face smaller than `MIN_FACE_AREA = 1e-4` | Snap nearly-coincident endpoints together; avoid tiny triangles |
| `faces do not tile the sheet` | Face areas don't sum to `S²` within 1e-6 | Crease endpoints must lie **exactly** on other creases or the boundary — floating-point offsets break tiling |
| `disconnected faces` | Face adjacency graph is disconnected | Connect every panel to the rest through shared edges |

See [CREASE_FORMAT.md](CREASE_FORMAT.md) for the full format rules and the JSON Schema failure modes (`ValueError("invalid crease JSON: ...")` from `load_creases_from_json`).

## Simulation stability

**`FloatingPointError: unstable simulation: non-finite joint coordinates produced.`** — the fold exploded (usually extreme target angles, too-long `duration`, or aggressive vibration). Try: smaller fold angles (`|angle| ≤ MAX_FOLD = 0.98π` is enforced, but smaller is more stable), shorter `duration`, `vibrate=False` or smaller `amp_z`/`freq_hz`, or `generate_xml(..., coll_thick=...)` thin-collision mode.

**Simulation is slow** — cost is `duration / 0.001` steps (8 s ⇒ 8000 step calls) plus mesh contact. Use short `duration` while iterating. Population evaluation is **sequential** in both backends (`batch_size` is accepted but currently ignored), so splitting work across independent processes or using a GPU is the practical way to increase throughput.

## PyPI install vs source checkout

| Behavior | PyPI wheel | Source checkout (`pip install -e .[dev,jax,warp]`) |
|---|---|---|
| `import origami_engine_mjx`, `OrigamiEngine` | identical | identical |
| `load_default_creases()` | reads the packaged `origami_engine_mjx/crease_patterns/reference_miura_4x4.json` | same file (byte-identical to repo-root `crease_patterns/reference_miura_4x4.json`) |
| `tests_mjx/`, `examples/`, `Makefile` targets | not shipped | available (`make test`, `make examples`, …) |
| `plot_pattern` output | caller saves `plt.gcf()` wherever they choose | same behavior |
| Video files | not produced by this package | not produced by this package |
| Ruff / mypy / pytest | identical | identical |
| `generate_xml` output | identical | identical |

## Upgrading / uninstalling

```bash
pip install -U origami-engine-mjx     # upgrade        | uv pip install -U origami-engine-mjx
pip uninstall origami-engine-mjx      # remove         | uv pip uninstall origami-engine-mjx
```

## Optimization / RL environments

**`RuntimeError: OrigamiEnv requires 'gymnasium'.`** — install the RL extra: `pip install 'origami-engine-mjx[rl]'`.

**`RuntimeError: ... requires 'mujoco-mjx' and 'jax'.`** from `OrigamiEnv` or `integrations.jax_env` — install the JAX extra: `pip install 'origami-engine-mjx[jax]'`.

**`RuntimeError: pymoo is required for to_pymoo_problem().`** — install the optimizer extra: `pip install 'origami-engine-mjx[opt]'`. `nevergrad`/`cma`/`evosax` are imported by your own script, not by the library, so install whichever optimizer you use.

**`ValueError: OrigamiEnv currently supports backend='jax' only`** — the Warp backend is stateful and is not wrapped as an environment; use the MJX-JAX backend (see [RL.md](RL.md)).

**`RuntimeError: call reset() before step()`** — call `env.reset()` (or `OrigamiVectorEnv.reset()`) before the first `step()`.

**`ValueError: genome has length N, expected M`** — the vector length does not match `SearchSpace.dim` / `DesignSpace.dim`. Remember the default pattern has 15 *actuated* variables, not 24 creases; build the space with the same `active_only` setting you use elsewhere. See [OPTIMIZATION_ES.md](OPTIMIZATION_ES.md).

**`ValueError: ... selection matched no variables`** / **`conflicting selections`** — a `DesignSpace.select(...)` (or builder `select`) matched nothing, or two constraints act on the same variable. Fix the group/index/predicate, or drop one of the conflicting constraints. See [SEARCH_SPACE.md](SEARCH_SPACE.md#3-selecting-variables).

**`DesignSpace` geometry gives `status != ok` or `path == "invalid"`** — a sampled genome can move endpoints so the crease set no longer tiles the sheet. The evaluator never crashes: the individual is reported `invalid` (or scored with the `error_policy` penalty) and `BatchResult.paths` says so. Repair genomes with `space.repair(...)`, keep them in bounds with `space.clip(...)`, or filter with `space.is_valid(...)`. See [SEARCH_SPACE.md](SEARCH_SPACE.md#8-validity-repair-and-rejection-sampling).

**`ValueError: the RL environments need a fixed topology and an angle-only genome`** — `OrigamiEnv` / `OrigamiVectorEnv` share one fixed topology, so a `DesignSpace` that varies geometry, crease count or constraints is rejected. Use an angle-only `SearchSpace` (or `DesignSpace.angles_only()`) for RL, and the evaluator for geometry/count search. See [RL.md](RL.md).

**`load_seeds` reports files as rejected** — malformed or non-tiling crease files are collected in `SeedSet.rejected` (`[(path, reason), ...]`) instead of raising. Check the reason (e.g. `"read error"`, `"invalid crease pattern"`) and the file's JSON. See [POPULATIONS.md](POPULATIONS.md#2-loading-seed-files).

**The integration tests are skipped** — `test_integrations.py` / `test_rl_env.py` skip themselves when their optional dependencies are absent (`pytest.importorskip`, or the `requires_jax` guard in `test_integrations.py`), and simulation cases are marked `slow`. Install `[opt]`/`[rl]`/`[jax]` and run `pytest -q -m slow`.

## Getting help

- Search/open an issue: <https://github.com/ihsan31415/origami-engine-mjx/issues>
- When reporting, include: Python version, install method (pip/uv/wheel/editable), the exact command, the full traceback, and a minimal crease list that reproduces the problem.
- For usage questions, check [TUTORIALS.md](TUTORIALS.md) and [GLOSSARY.md](GLOSSARY.md) first.