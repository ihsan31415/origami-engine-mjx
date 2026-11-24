# Final Report — configurable search space & population seeding

## Overview

| Item | Value |
|---|---|
| Branch | `feat/optimization-integrations-v1` |
| Base | `db93f01` (previous branch tip; this report's work is **local only** — not pushed, tagged or published) |
| Commits added | 8 (`5047469` A–D implementation + `cycle 1`…`cycle 6` review cycles + this report) |
| Review cycles run | 6 (two consecutive clean full passes required; see §7) |
| Version | `1.0.0` (`pyproject.toml` + `origami_engine_mjx.__version__`, unchanged) |

This work **extends** the existing ES/RL/AI integration substrate (added on the
same branch by `db93f01` and earlier). It makes the search space
**dynamic/configurable** and adds a **population seeding/initialization** module.
It ships **no** optimizer: no mutation, crossover, selection or search algorithm.
Everything is additive — the existing public API and the default genome are
unchanged.

The single most important invariant: **the default behaviour is byte-for-byte
identical.** `DesignSpace.from_default_pattern()` is the 15-actuated-angle,
fixed-topology genome that `SearchSpace.from_default_pattern()` already exposed
(same vectors, bounds, parameter names and decoded creases), and
`to_search_space()` reproduces the `SearchSpace` exactly. `SearchSpace` remains
the default everywhere.

---

## 1. Modules added (interfaces only)

| Module | Public surface |
|---|---|
| `integrations/design_space.py` (new, ~1200 lines) | `DesignSpace`, `DesignSpaceBuilder`, `Variable`, `ANGLE` / `COORD` / `PRESENCE`. Configurable genome over fold angles, loop-edge angles, crease endpoints (geometry) and crease presence (variable count); presets `angles_only`, `angles_and_geometry`, `geometry_only`, `everything`; selection by group/index/slice/list/bool-mask/name/predicate; per-variable bounds; normalized `[-1, 1]` view; equality/symmetry constraints that **reduce** the dimension and are re-expanded by `decode`; `is_valid` / `validity_reason` / `repair` / `rejection_sample`; `presence_mask` / `presence_count` / `count_ok`; `describe()`; dict/JSON + pickle round-trip; `to_search_space()`. |
| `integrations/population.py` (new, ~660 lines) | `Population`, `SeedSet`, `load_seeds`, `interpolate`, `piecewise_path`, `random_barycentric`, `jitter`, `random_population`, `lhs_population`, `sobol_population`, `gaussian_population`, `deduplicate`, `diversity_filter`, `mixed_population`, `save_population`, `load_population`, `export_creases_dir`, `population_stats`. Seeding/initialization only. |
| `integrations/evaluator.py` (extended) | Accepts `SearchSpace` **or** `DesignSpace`; adds `topology_signature`, per-individual `EvalRecord.path` / `BatchResult.paths`, and topology bucketing in the fast batch path. |
| `integrations/env.py` (extended) | `OrigamiEnv` / `OrigamiVectorEnv` accept `SearchSpace` or an angle-only, unconstrained `DesignSpace` via `_as_rl_space`; anything else raises a clear `ValueError`. |
| `integrations/adapters.py` (extended) | `space_spec(space_or_evaluator)` exposes dim / lower / upper / names / groups / presence indices / n_creases. |
| `integrations/__init__.py` (extended) | Re-exports the new names; existing exports unchanged. |
| `tests_mjx/test_design_space.py`, `tests_mjx/test_population.py` (new) | Fast unit tests plus one `slow`/`requires_jax` real-engine equality test. |
| `examples/06_design_space.py`, `examples/07_populations.py` (new) | Runnable, CPU-safe examples. |

`DesignSpace` selection/semantics: variable kinds `ANGLE` / `COORD` /
`PRESENCE`; coordinate variables are one per unique vertex (`geometry_mode="vertex"`)
or one per crease endpoint (`"segment"`); a presence value below
`presence_threshold=0.5` drops the crease (list view from `decode`, mask from
`presence_mask`); constraints remove dependent variables from the genome.

### Deliberate deferrals (not implemented)

- **No optimizer / operators** — no mutation, crossover or selection. See §5.
- **No cross-topology batched `vmap`** — see §3.
- **Optional Task-B items not built** (listed for a decision in §5):
  parameterized pattern generators, structure-aware distance/alignment for
  variable-length genomes. (A warm-start-from-best loader already exists as
  `plumbing.save_best` / `load_best`.)

---

## 2. Default behaviour is byte-for-byte unchanged

- `DesignSpace.from_default_pattern()` → dim `15`, `n_creases == 24`,
  `n_slots == 24`.
- `DesignSpace.from_default_pattern().to_search_space() == SearchSpace.from_default_pattern()`
  (compared via `SearchSpace.__eq__`), and `lower`, `upper`, `parameter_names`
  and the base vector are element-wise identical.
- Pinned by `test_angles_only_reproduces_search_space`,
  `test_to_search_space_round_trip_is_identical` and
  `test_from_search_space_round_trip`.
- `OrigamiEnv` / `OrigamiVectorEnv` still default to the fixed-topology angle
  space; RL remains angle-only by design (a geometry/presence/constrained
  `DesignSpace` is rejected with a clear error).

---

## 3. Measured batching (honest)

Both built-in backends loop **per individual** inside `run_batch` (each
iteration rebuilds the MJX model; `batch_size` is ignored), and
`OrigamiEngine.evaluate_population` ignores `trajectory` (it always scores +x).
So grouping genomes by `topology_signature` is **bookkeeping only** and gives
**no throughput benefit** today. Measured on this CPU host (JAX backend, tiny
2-crease pattern, `duration=0.003`, `N=3`; model build dominates):

| Mode | Batch wall | One-by-one wall | Paths |
|---|---|---|---|
| angles-only (one topology) | 30.46 s (first call, incl. JIT compile) | 14.02 s | `batched` |
| geometry (mixed topology) | 14.40 s | 14.84 s | `single` |

The mixed-topology batch is the meaningful comparison (compile is warm):
**4.80 s/ind batched vs 4.95 s/ind one-by-one** — statistically the same, i.e.
no speed-up. The evaluator still reports *which* path scored each individual
(`batched` / `bucketed` / `single` / `invalid` / `cached`) so a caller can see
this.

**Equality is verified with the real engine**, not just a stub:
`test_real_engine_mixed_topology_scores_match_one_by_one` builds a geometry
space over a tiny pattern, perturbs one vertex (a second topology), and asserts
the batch fitness equals the one-by-one fitness for all three rows, with paths
`['bucketed', 'bucketed', 'single']` and all statuses `ok`.

---

## 4. Fixes made during the review cycles

| Commit | Fix |
|---|---|
| `148e405` cycle 1 | `rejection_sample(size=0)` returns an empty `(0, dim)`; `everything()` docstring corrected (`presence=None` adds **no** presence variables); `DesignSpace` rejects a `dependent` source that is itself dependent; `diversity_filter` returns genome-space rows while measuring distances in the normalized view; `deduplicate` gained a `space` argument (normalized duplicate detection, genome rows returned); `population_stats` reports `n=0` for empty input. |
| `a304565` cycle 2 | Documented that interpolation does **not** check geometric validity; use `is_valid` / `rejection_sample` / `mixed_population(valid_only=True)`. |
| `0510336` cycle 3 | Stated honestly that topology bucketing gives **no** throughput benefit today (`EVALUATION.md`, `SEARCH_SPACE.md`). |
| `c74d14d` cycle 4 | Fixed a real bug: `piecewise_path(closed=True)` sampled the path parameter in `[0, m]` instead of `[0, m-1]`, extrapolating one segment past the final vertex (e.g. seeds `[0, 1, 2]` gave range `[-2, 2]` instead of `[0, 2]`). Added regression test. Also made `jitter(normalized=True)` without a `space` raise `ValueError` (it previously added raw-unit noise silently). |
| `17f03ba` cycle 5 | Made `count_mode="range"` meaningful: added `presence_count()` / `count_ok()` and made `rejection_sample` honour both geometry validity and the crease-count bounds. |
| `0ec401d` cycle 6 | Added the real-engine mixed-topology equality test (the fast suite only had a stub-engine test). |

An independent read-only probe (`/tmp/opencode/review_probe.py`,
`review_probe2.py`) — 15 checks covering default byte-identity, selection
errors, geometry-invalid never crashing, closed-path hull containment,
`jitter` guard and count bounds — passes.

---

## 5. Issues that need your decision (not built)

| Item | Why left as-is |
|---|---|
| **Mutation / crossover / selection operators** | The agreed boundary is *interfaces only*; the library ships no algorithm. Building operators would cross that line. |
| Parameterized pattern generators | Optional Task-B item; not clearly in scope and no clean, verified implementation was attempted. |
| Structure-aware distance / alignment for variable-length genomes | Optional; the shipped `diversity_filter` measures distance in the normalized padded view, which is the documented behaviour. |
| `pyproject.toml` classifier `Development Status :: 3 - Alpha` while version is `1.0.0` | Pre-existing; changing release status is a maintainer policy decision. |
| `README.md` says "nine step-by-step recipes"; `TUTORIALS.md` has eight | Pre-existing; `README.md` is protected under the agreed constraints. |
| `SearchSpace.__eq__` ignores `parameter_names` | Pre-existing deliberate behaviour. |
| `[opt]` extra lists `nevergrad`/`cma`/`evosax`, which the package never imports | Pre-existing convenience installs. |
| Module docstrings sit after `from __future__ import annotations`, so `__doc__` is `None` | Pre-existing repo-wide convention. |

---

## 6. Unverifiable items (GPU / Warp only)

- **MJX-Warp backend execution and throughput** — `warp` / `mujoco-warp` are not
  installed and there is no CUDA GPU here (`available_backends()` returns
  `["jax"]`). All Warp claims remain documented, not runtime-verified.
- **`OrigamiEnv` / `jax_env` on Warp**, **GPU evals/s**, and
  **`jit` / `vmap` / `grad` on the Warp backend** — same reason.
- The batching numbers in §3 are **CPU/JAX** only; the mixed-topology
  no-benefit conclusion follows from the backend loop structure and should be
  re-measured on a GPU before any batching claim there.

---

## 7. Verification (final state)

| Gate | Result |
|---|---|
| `ruff check .` | All checks passed |
| `ruff format --check .` | all files formatted |
| `mypy origami_engine_mjx` | Success: no issues in 24 source files |
| `python scripts/check_docs_links.py` (`make docs-check`) | OK: 23 Markdown files, no broken links |
| `pytest tests_mjx -q` (fast) | 114 passed, 10 skipped |
| `pytest tests_mjx -m slow -q` (MJX sims) | 11 passed, 114 deselected (~150 s) |
| `python examples/06_design_space.py` / `07_populations.py` | exit 0 |
| Independent probes | all checks pass |

Two consecutive clean full passes (lint + format + mypy + docs links + fast suite
+ slow suite + examples + probes) were run on the frozen final tree.

---

## 8. README diff (protected file)

`git diff db93f01..HEAD -- README.md` → **+5 / −2**:

```diff
@@ -30,6 +30,7 @@
+- Offers a **configurable design space** (`DesignSpace`: fold angles, crease endpoints/geometry, crease count) and **population seeding** helpers (seed files, interpolation, random and mixed initialization) on top of the default fixed-topology search space

@@ -174,6 +175,8 @@
+- **[SEARCH_SPACE.md](docs/SEARCH_SPACE.md)** — the configurable `DesignSpace` (angles, geometry, crease count)
+- **[POPULATIONS.md](docs/POPULATIONS.md)** — population seeding / initialization

@@ -200,7 +203,7 @@   (Core modules (brief): integrations line)
-... `search_space.py` (`SearchSpace`), `evaluator.py` ...
+... `search_space.py` (`SearchSpace`), `design_space.py` (`DesignSpace`, the configurable angles/geometry/crease-count space), `evaluator.py` ..., `population.py` (seed loading, interpolation, random / mixed initialization, population I/O), ...

@@ -224,7 +227,7 @@   (Optimization & RL interfaces paragraph)
+ ... The search space is configurable — `DesignSpace` can also vary loop-edge angles, crease endpoints (geometry) and crease count, with presets, selection and equality/symmetry constraints — and initial populations can be seeded from crease JSON files or generated by interpolation, random and mixed sampling. See ... [SEARCH_SPACE.md], [POPULATIONS.md] and [RL.md].
```

All other README content (including the Examples table) is unchanged.

---

## 9. Review / push commands

**Review the branch (read-only):**

```bash
git log --oneline --reverse db93f01..feat/optimization-integrations-v1
git diff --stat db93f01..feat/optimization-integrations-v1
git diff db93f01..feat/optimization-integrations-v1 -- README.md   # expect +5/-2
ruff check . && ruff format --check .
mypy origami_engine_mjx
python scripts/check_docs_links.py
pytest tests_mjx -q
pytest tests_mjx -m slow -q        # optional; ~2.5 min on CPU
python examples/06_design_space.py && python examples/07_populations.py
```

**Push (only when you decide to):**

```bash
git push -u origin feat/optimization-integrations-v1
```

No tag or release was created; publishing remains your call.
