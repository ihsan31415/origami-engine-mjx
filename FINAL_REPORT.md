# Final Report — v1.0.0 ES / RL / AI Integration Interfaces

## Overview

| Item | Value |
|---|---|
| Branch | `feat/optimization-integrations-v1` (from `chore/remove-legacy-classic-engine` @ `b891a5c`) |
| Commits on branch | see `git log --oneline chore/remove-legacy-classic-engine..HEAD` (local only — **not** pushed, tagged, or published) |
| Review cycles run | 8 (the agreed maximum); cycles 1–7 fixed interface/code issues, cycle 8 fixed pre-existing doc/CI issues |
| Head | branch tip on `feat/optimization-integrations-v1` (see `git log -1`; this report is the last commit) |
| Version | `1.0.0` (`pyproject.toml` + `origami_engine_mjx.__version__`) |

The branch adds the **environment / evaluator substrate** for external search and
learning tools. It ships **no** optimization or learning algorithm: every
algorithm lives in a third-party package that consumes the small, stable
interfaces added here. All additions are under
`origami_engine_mjx/integrations/` and use only the existing public
`OrigamiEngine` facade, so they work with both backends and change no existing
public API or signature.

---

## 1. Modules added (interfaces only)

| Module | Public surface |
|---|---|
| `integrations/search_space.py` | `SearchSpace` — fixed-topology fold-angle genome: `from_creases` / `from_default_pattern`, `encode` / `decode`, `sample` / `sample_normalized`, `clip` / `repair` / `is_valid`, `to_normalized` / `from_normalized`, `bounds`, `to_dict`. Default 4×4 Miura = 15 actuated variables (9 loop creases are inert; `active_only=False` exposes all 24). |
| `integrations/evaluator.py` | `OrigamiEvaluator` — stateless, picklable objective (single + batch); `EvalRecord`, `BatchResult`, `DESCRIPTOR_NAMES` (quality-diversity behavior descriptors); `to_minimizer` / `to_maximizer`; `error_policy="penalty"\|"raise"`; optional memoization. |
| `integrations/adapters.py` | `to_pymoo_problem`, `to_nevergrad_function`, `to_evosax_fitness` — thin shape adapters; third-party libs imported lazily. |
| `integrations/env.py` | `OrigamiEnv` (Gymnasium) + `OrigamiVectorEnv`; `EnvParams` / `EnvState`; action modes; per-hinge fold-sign handling. |
| `integrations/jax_env.py` | Pure-JAX `build_jax_model` / `make_params` / `reset` / `step` / `observe` for `jit` / `vmap` / `scan`. |
| `integrations/plumbing.py` | `make_rng`, `set_global_seed`, `save_genome` / `load_genome`, `save_best` / `load_best`, `RunLogger` (JSONL/CSV), `benchmark`. |
| `integrations/__init__.py` | Eager exports of the NumPy-only surface + lazy `__getattr__` for the JAX/Gymnasium symbols. |

### Deliberate deferrals (not implemented, by design)

- **No algorithms** — no CMA-ES, no GA, no policy-gradient, no QD/MAP-Elites implementation. Only the interfaces above.
- **No cross-topology batching / `vmap`** — individuals with different crease patterns have different MJX models, so `evaluate_batch` loops; `jax_env` is vmappable only when the model is shared.
- **Structured descriptors require the slow path** — `evaluate_population`'s fast path returns scores only; per-individual `engine.run` is needed for trajectory-derived descriptors (documented in `OPTIMIZATION_ES.md`).
- **No Warp-specific env implementation** — the env is backend-agnostic; the Warp path is documented but not runtime-verified here (no GPU).

---

## 2. Documentation added / enriched

**New files**

- `docs/README.md` — documentation index.
- `docs/INTEGRATIONS.md` — overview, install extras, the no-algorithms boundary, module map.
- `docs/OPTIMIZATION_ES.md` — search space, evaluator, optimizer adapters, QD descriptors, error contract, throughput benchmark.
- `docs/RL.md` — Gymnasium + pure-JAX environments, observation/action spaces, `ctrl_perm`/`ctrl_sign` reindexing, reward, termination/info keys.
- `examples/04_optimize_es.py`, `examples/05_rl_env.py` — runnable, optional-dependency-guarded examples.

**Enriched (existing docs)**

- A "Documentation map" nav line linking the three new docs added to every existing doc.
- `README.md` — **+9/−0 only**: one "What this project does" bullet, three entries in the "Documentation" list, one "Core modules" bullet, and one short "Optimization & RL interfaces" section. The Examples table is untouched (see §6).
- `docs/ARCHITECTURE.md` — `integrations/` row added to the module-responsibility table.
- `docs/CONTRIBUTING.md` — `[opt]`/`[rl]` extras, the `requires_jax` skip guard, `make docs-check`.
- `docs/TROUBLESHOOTING.md` — integration error strings and skip-guard behaviour.
- `docs/API_REFERENCE.md`, `docs/GETTING_STARTED.md`, `docs/GLOSSARY.md`, `docs/EVALUATION.md`, `docs/TUTORIALS.md`, `docs/CREASE_FORMAT.md`, `docs/SIMULATION.md`, `docs/BACKENDS.md`, `docs/MIGRATION.md`, `docs/COLAB_GPU.md`, `examples/README.md` — map links plus targeted corrections.
- `CHANGELOG.md` — repo-scoped `[1.0.0]` entry.
- `scripts/check_docs_links.py` (new) + `make docs-check` target; now also run in CI.

No existing doc or code file was deleted or restructured; changes are additive or corrections of factually wrong statements (listed in §3/§4).

---

## 3. Fixes made during the review cycles

Cycles 1–7 (interface / code correctness):

| Commit | Fix |
|---|---|
| `6fb9c81` | Initial integration interfaces |
| `602b983` | Docs + examples |
| `384beb3` | Correct `space.sample(0)` doc call; CSV descriptor inference |
| `8e71e9f` | Release prep: version `1.0.0` |
| `4eca9a2` | Accurate docs; `render()` error message; env tidy-up |
| `24f6d0d` | Document `benchmark` `batch_size` return key |
| `6b85505` | Align RL action order with the search-space order |
| `a571fc7` | `make_params` requires the hinge→crease map |
| `441d7e5` | `close()` renderer; guard `benchmark(n<1)` |
| `44a733d` | Document info-key differences on terminal step vs reset |
| `190433a` | Apply per-hinge fold sign in RL envs |
| `73bd585` | Consistent error handling; `BatchResult.errors`/`cached`; warm-up not counted as a cache hit |

Cycle 8 (final — pre-existing docs/CI defects surfaced by review):

| Commit | Fix |
|---|---|
| `adaf8ab` | Guard JAX import in `test_rl_env.py` (optional-deps CI) |
| `a9ccab5` | List `integrations/` in `ARCHITECTURE.md` module table |
| `08e0c0b` | `API_REFERENCE.md`: TUTORIALS has **eight** recipes, not nine |
| `2dab4ff` | `EVALUATION.md`: GA-loop snippet had `scores = ([],)` (a tuple) → `scores = []` |
| `373dfd2` | `requires_jax` skip guard for the 7 JAX-backed tests; run docs link check in CI |
| `224537d` | Correct the `[all]` description (now includes `[opt]`/`[rl]`); fix example "Key APIs" label |
| `f7bd296` | `TROUBLESHOOTING.md`: integration tests also use the `requires_jax` guard |
| `43b43c2` | Clarify `RunLogger` docstring (append within session / truncate on first open) |

The most consequential cycle-8 fix was **CI**: the `lint-and-test` job installs
only `.[dev]` (no `jax`/`mujoco-mjx`), yet seven tests in
`test_integrations.py` built a JAX-backed evaluator and errored instead of
skipping. They now skip via a `find_spec`-based `requires_jax` guard, matching
the existing `importorskip` convention; reproduced locally by blocking `jax`
(`33 passed, 13 skipped`) and confirmed green with `jax` present
(`55 passed, 10 skipped`).

---

## 4. Issues not fixed (require a user decision)

| Issue | Location | Why left as-is |
|---|---|---|
| Classifier says `Development Status :: 3 - Alpha` while the version is `1.0.0` | `pyproject.toml:17` | Changing the release status is a policy decision for the maintainer. |
| README says "nine step-by-step recipes"; `TUTORIALS.md` has **eight** | `README.md:163` | `README.md` is protected under the agreed constraints (only the listed edits are allowed). The same sentence in `docs/API_REFERENCE.md` **was** fixed (`08e0c0b`). |
| `SearchSpace.__eq__` ignores `parameter_names` | `integrations/search_space.py:214` | Deliberate: two spaces with identical topology/bounds but different labels compare equal. Changing it is a behavioural decision. |
| CSV `RunLogger` uses `extrasaction="ignore"` | `integrations/plumbing.py:175` | Descriptors absent from the first record are silently dropped; erroring instead would be stricter but could break valid runs. |
| Cache key uses `row.tobytes()` vs `np.ascontiguousarray(X[i]).tobytes()` | `integrations/evaluator.py` | Differs only for non-contiguous input; unlikely to matter but worth a decision. |
| `status="invalid"` is effectively unreachable through the public API | `integrations/evaluator.py` | Kept for forward compatibility. |
| `[opt]` lists `nevergrad`/`cma`/`evosax`, which the library never imports | `pyproject.toml:45` | Convenience installs for the *user's* optimizer; `pymoo` is the only one imported (lazily). |
| Module docstrings sit after `from __future__ import annotations`, so `__doc__` is `None` | repo-wide | Existing convention; fixing it touches many files and is out of scope. |
| `integrations.__all__` omits the 7 lazily-exported env symbols | `integrations/__init__.py:55` | Deliberate: listing them would make `from … import *` eagerly import `env`/`jax_env` (and their optional deps), defeating the lazy re-export. |
| `README.md` Examples table lists only 4 of the 6 scripts (04/05 missing) | `README.md` Examples table | The README Examples table is explicitly protected ("do not touch"). The two new scripts are documented in `examples/README.md` and the new docs instead. |
| `README.md` Examples table | `README.md` | Untouched by design (protected). |

---

## 5. Unverifiable items (GPU / Warp only)

- **MJX-Warp backend execution** — `warp` / `mujoco-warp` are not installed and no CUDA GPU is present (`available_backends()` returns `["jax"]` here). The Warp path is documented, not runtime-verified.
- **`OrigamiEnv` / `jax_env` on Warp** — the env is written to work with both backends via `OrigamiEngine`, but only the JAX path has been executed.
- **GPU throughput numbers** (evals/s, steps/s, memory) — require a Colab T4 or similar; `examples/colab_gpu_validation.py` is the intended harness.
- **`jax.jit` / `jax.vmap` / `jax.grad` on the Warp backend** — documented as not applicable (Warp state is not a JAX pytree).

All GPU-only claims in the docs are marked as unverified.

---

## 6. README diff (protected file)

`git diff chore/remove-legacy-classic-engine -- README.md` → **+9 / −0** (added lines shown; surrounding context elided):

```diff
@@ -29,6 +29,7 @@
+- Exposes ES / RL / AI **integration interfaces** (a fixed-topology search space, a stateless evaluator, Gymnasium and JAX-functional environments, optimizer adapters) without shipping any optimization or learning algorithm

@@ -171,6 +172,9 @@
+- **[INTEGRATIONS.md](docs/INTEGRATIONS.md)** — ES/RL/AI integration interfaces overview
+- **[OPTIMIZATION_ES.md](docs/OPTIMIZATION_ES.md)** — search space, evaluator, optimizer adapters, QD descriptors
+- **[RL.md](docs/RL.md)** — Gymnasium and JAX-functional reinforcement-learning environments

@@ -196,6 +200,7 @@
+- `origami_engine_mjx/integrations/` — ES/RL/AI interfaces (no algorithms): `search_space.py` (`SearchSpace`), `evaluator.py` (`OrigamiEvaluator`, descriptors), `adapters.py` (pymoo / nevergrad / cma / evosax), `env.py` (`OrigamiEnv`, `OrigamiVectorEnv`), `jax_env.py` (pure-JAX `reset` / `step`), `plumbing.py` (seeding, genome/checkpoint I/O, run logging, benchmark)

@@ -217,6 +222,10 @@
+## Optimization & RL interfaces
+
+The library is the **environment / evaluator substrate** for external search and learning tools — it ships **no** optimization or learning algorithm. `origami_engine_mjx.integrations` provides a fixed-topology `SearchSpace` (15 actuated fold angles for the default Miura pattern), a stateless picklable `OrigamiEvaluator` with behavior descriptors for quality-diversity, a Gymnasium `OrigamiEnv` plus pure-JAX `reset`/`step` (jittable / vmappable), thin adapters for `pymoo` / `nevergrad` / `cma` / `evosax`, and plumbing (seeding, genome/checkpoint I/O, run logging, throughput benchmark). See [INTEGRATIONS.md](docs/INTEGRATIONS.md), [OPTIMIZATION_ES.md](docs/OPTIMIZATION_ES.md) and [RL.md](docs/RL.md).
```

The Examples table and all other README content are unchanged.

---

## 7. Verification (final state)

| Gate | Result |
|---|---|
| `ruff check .` | All checks passed |
| `ruff format --check .` | 56 files already formatted |
| `mypy origami_engine_mjx --config-file pyproject.toml` | Success: no issues in 22 source files |
| `python scripts/check_docs_links.py` | OK: 21 Markdown files, no broken links |
| `pytest -q` (fast suite) | 55 passed, 10 skipped |
| `pytest -q` with `jax` blocked (CI simulation) | 33 passed, 13 skipped, 0 failed |
| `pytest -q -m slow` (MJX simulation) | 10 passed, 55 deselected (~107 s) |
| Working tree | clean |

The slow MJX suite was run once during the cycles; the cycle-8 fixes touched
only test skip guards, docs, and CI, none of which affect the simulation path.

A final independent read-only review of the complete branch (interfaces, docs,
examples, tests, CI, packaging) found **no high-severity defects**; the only
medium finding — the protected README Examples table still lists 4 of the 6
scripts — is recorded in §4 as a decision for the maintainer.

---

## 8. Review / push / tag / release commands

**Review the branch (no side effects):**

```bash
git log --oneline --reverse chore/remove-legacy-classic-engine..feat/optimization-integrations-v1
git diff chore/remove-legacy-classic-engine -- README.md          # expect +9/-0
git diff chore/remove-legacy-classic-engine --stat
ruff check . && ruff format --check .
mypy origami_engine_mjx --config-file pyproject.toml
python scripts/check_docs_links.py
pytest -q
pytest -q -m slow        # optional; ~2 min on CPU
```

**Push the branch:**

```bash
git push -u origin feat/optimization-integrations-v1
```

**Tag the release** (no `v1.0.0` tag exists yet; `v0.1.0` and `v0.3.0` do):

```bash
git tag -a v1.0.0 -m "origami-engine-mjx 1.0.0"
git push origin v1.0.0
```

**Build and release:**

```bash
python -m build
twine check dist/*
# PyPI publish is automated by .github/workflows/ci.yml on a push to main
# (PyPI Trusted Publishing). Merging this branch to main triggers it.
gh release create v1.0.0 --notes-from-tag dist/*
```

---

## 9. Commit list (local; the commit containing this report is last)

```
6fb9c81 feat(integrations): add ES/RL/AI integration interfaces
602b983 docs(integrations): document ES/RL interfaces and add examples
384beb3 fix(integrations): correct doc sample() call and CSV descriptor inference
8e71e9f release: prepare v1.0.0
4eca9a2 fix(integrations): accurate docs and render() error, tidy env
24f6d0d docs(integrations): document benchmark batch_size return key
6b85505 fix(integrations): align RL action order with the search space
a571fc7 fix(integrations): require the hinge->crease map in make_params
441d7e5 fix(integrations): close renderer, guard benchmark(n<1)
44a733d docs(rl): info keys differ on terminal step and reset
190433a fix(integrations): apply per-hinge fold sign in RL envs
73bd585 fix(integrations): consistent error handling and benchmark warm-up
adaf8ab test(integrations): guard JAX import in RL env tests for optional-deps CI
a9ccab5 docs(architecture): list integrations/ in the module responsibilities table
08e0c0b docs(api-reference): TUTORIALS.md has eight recipes, not nine
2dab4ff docs(evaluation): fix GA-loop snippet (scores = [] not tuple)
373dfd2 test(integrations): skip JAX-backed tests without the [jax] extra
224537d docs: correct [all] extra description and example key-API label
f7bd296 docs(troubleshooting): integration tests also use the requires_jax guard
43b43c2 docs(integrations): clarify RunLogger append/truncate wording
```
