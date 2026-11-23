# Final Report: feat/mjx-dual-backend Documentation Migration

## Overview
This report documents all changes made to the `origami_engine_mjx` package on branch `feat/mjx-dual-backend` (commits: `1702eba` code fixes, `ee4399a` docs migration). The work adds/enriches documentation for the dual-backend MJX package and fixes genuinely wrong code found along the way. All changes are committed locally; nothing has been pushed.

---

## 1. Documentation Added / Enriched

### `docs/API_REFERENCE.md` — Complete rewrite
- Signatures, behavior, params/units/ranges, returns, exceptions, backend applicability, examples, see-also for every public symbol
- Grouped by module in README "Core modules" order (lines 178–192)
- Covers: `OrigamiEngine`, `Crease`, `build_pattern`, `load_default_creases`, `evaluate_individual`, `evaluate_population`, `compute_forward_displacement`, `generate_xml`, `plot_pattern`, `load_creases_from_json`, `save_creases_to_json`, `normalize_creases`, `DEFAULT_PATTERN_JSON`, `M`, `V`, `S`, `THICK`, `PAPER_MASS`, `Z0`, `MAX_FOLD`, `VIB_*`, `CREASE_SCHEMA`, all `backends/`, `schemas/`, `crease_patterns/`, all `__init__.py` exports
- Includes complete workflow example and see-also cross-references to all other docs

### `docs/GETTING_STARTED.md` — Migrated from legacy
- Installation commands (pip / uv, `[jax]`, `[warp]`, `[all]`)
- Backend selection (`jax`, `warp`, `auto`)
- First simulation (CPU MJX-JAX and GPU MJX-Warp)
- All legacy references removed (no `workers=`, no `run_simulation`, no version 0.1.1 claims)

### `docs/TUTORIALS.md` — Migrated from legacy
- 9 step-by-step recipes (inspect pattern, first simulation, score pattern, JSON round-trip, custom pattern, population evaluation, plot & save PNG, raw XML, see-also)
- Replaces legacy `render_individual`/`origami-render` references with MJX-appropriate workflows
- Documents 2-vec vs 3-vec trajectory semantics, mountain/valley sign convention

### `docs/CREASE_FORMAT.md` — Migrated and enriched
- `Crease` type specification, sign convention (negative=mountain, positive=valley)
- JSON Schema reference (Form A with `angle_unit`, Form B bare list)
- `normalize_creases` error types and messages
- `load_creases_from_json` schema validation details
- `save_creases_to_json` round-trip behavior
- Reference pattern specs (24 creases, 16 faces, 9 loop edges)
- `build_pattern` validation reasons and design guidelines
- Swapped constants note (M/V naming) documented

### `docs/SIMULATION.md` — Migrated and enriched
- Full pipeline diagram (crease → normalize → build_pattern → generate_xml → backend.run → SimulationResult)
- Model generation details (nq=54, nv=53, nu=15, neq=18, ngeom=33, nbody=18 for reference pattern)
- Actuator targets, smoothstep ramp, table vibration
- Backend-specific state handling (JAX immutable pytree vs Warp `.numpy()` views)
- Performance notes (~70 s/step CPU, ~1.4 s/step 2-face)
- Rendering note: neither MJX backend renders directly

### `docs/EVALUATION.md` — Migrated and enriched
- `compute_forward_displacement` semantics (2-vec not normalized, 3-vec IS normalized; returns nan for degenerate cases)
- `evaluate_individual` full signature and raise conditions
- `evaluate_population` `error_policy` (`"raise"` vs `"nan"`), workers note (no `workers` param in MJX)
- GA-style fitness loop example

### `docs/TROUBLESHOOTING.md` — Migrated
- Installation problems, import errors, headless rendering
- Invalid crease pattern reasons (no faces, sliver, tiling, disconnected)
- Simulation stability (FloatingPointError)
- PyPI vs source checkout differences

### `docs/GLOSSARY.md` — Enriched
- 28 terms defined (crease, mountain/valley, face, edge, tree edge/hinge, loop edge, root face, Pattern, sliver face, MJCF/MjModel/MjData, mj_step, position actuator, stretch joints, equality connect, vibrating table, paper centroid, forward displacement, trajectory, individual/population, Miura-ori, solref/solimp, condim, MJX, origami-render, JSON Schema, Warp backend)
- Swapped constants note referenced

### `docs/CONTRIBUTING.md` — Enriched
- `[dev]` extras, `[jax]`, `[warp]` extras documented
- Makefile targets (`make test`, `make lint`, `make format`, `make typecheck`)
- Test suite notes (tiny 2-face patterns; slow tests excluded by default)
- CI config reference
- Documentation contribution guidelines (nav line, code accuracy, link style)
- Project layout table

### `docs/ARCHITECTURE.md` — Enriched
- Data flow diagram updated for MJX
- Module responsibility table (shared vs backend-specific)
- Key MJX differences from classic MuJoCo (5 bullet points)
- `Crease` → `Pattern` → Simulation pipeline steps
- See-also cross-references

### `docs/BACKENDS.md` — Enriched and tabularized
- MJX-JAX: hardware, transformations, batching, performance, install
- MJX-Warp: hardware, transformations, batching, status
- Backend selection logic (`auto`, explicit)
- Feature/limitation comparison table (Classic vs MJX-JAX vs MJX-Warp across 8 dimensions)

### `docs/COLAB_GPU.md` — Enriched with checklist
- Validation checklist (9 items from `colab_gpu_validation.py`)
- Setup instructions (pip, checkout)
- What to record (9 benchmark metrics)
- Manual run steps
- Troubleshooting GPU issues

### `docs/MIGRATION.md` — Enriched with full API mapping
- Old vs. new entry points table (12 mappings)
- API mapping examples (old script → new script)
- Behavioral notes table (state mutability, step function, control arrays, name lookups, population workers, render video, constants sign convention, origami-render entry point)
- `origami-render` entry point removal documented

### `examples/README.md` — Migrated
- All 7 example scripts documented with Key APIs
- Old→new mapping table (example → tutorial → deeper doc)
- Legacy `make render`/`origami-render` references removed

### `README.md` — Adjusted Documentation section (lines 157–167)
- 9 doc links added; legacy `origami-engine` links removed; same bullet style preserved
- Core modules section (lines 180–192) reorganized

---

## 2. Code Fixes

### Commit `1702eba`: "fix: drop broken origami-render-mjx entry point; mypy the MJX package in CI"
- Removed `[project.scripts] origami-render-mjx` from `pyproject.toml` (module doesn't exist)
- CI now runs `mypy origami_engine_mjx --config-file pyproject.toml`
- Fixed lazy-import mypy errors in `backends/jax_backend.py` and `backends/warp_backend.py` (import-first, assign `None` in `except` with `# type: ignore[assignment]`)
- `engine.py` auto-resolution `# type: ignore[assignment]`
- `evaluation.py` return annotation fixed to `list[float]` (dropped unused `PopulationResult` import)
- Stale `origami_engine.*` → `origami_engine_mjx.*` in `geometry.py` and `plotting.py` docstrings and schema description
- After: mypy clean (15 files), ruff check/format clean (62 files), `pytest tests_mjx -q` → 13 passed, 3 skipped

### Additional fixes during docs work
- Updated `.pre-commit-config.yaml` mypy hook regex from `^origami_engine/` to `^origami_engine(_mjx)?/` so pre-commit actually type-checks the shipped package
- Updated `Makefile` `typecheck` target from `origami_engine` to `origami_engine_mjx --config-file pyproject.toml`

### Sign convention note (documented, not changed)
- Runtime convention: negative=mountain, positive=valley
- `constants.py` has `V = deg2rad(-45)`, `M = deg2rad(45)` (names swapped vs. runtime convention)
- Per decision: **do NOT change values**; document current behavior in CREASE_FORMAT.md, BACKENDS.md, MIGRATION.md, and this report

---

## 3. Issues Not Fixed (Requiring Decision)

| Issue | Status | Reason |
|---|---|---|
| `M = +π/4` / `V = -π/4` constant names swapped vs. runtime sign convention | Documented, values unchanged | Per decision; inherited from upstream `origami-engine`; changing would break reference pattern geometry |
| MJX-Warp GPU execution / benchmarks | Blocked on Colab T4 access | No GPU numbers may be claimed; `examples/colab_gpu_validation.py` intended for Colab T4 only |
| `origami-render` CLI entry point removal | Documented, not reverted | Module doesn't exist; removal from `pyproject.toml` already committed as `1702eba` |
| `workers` parameter in `evaluate_population` | Documented as absent | MJX backends do not support `workers`; batching handled natively (`batch_size`) |
| Numerical differences vs classic MuJoCo | Documented as expected | MJX uses float32 XLA kernels; classic uses float64 C++; investigate rather than forcing identical outputs |
| Rendering video in MJX package | Documented as not supported | Neither MJX backend renders; use classic MuJoCo or `mjx.get_data` + `mujoco.Renderer` |

---

## 4. Unverifiable Items (GPU-only)

- MJX-Warp GPU execution: `WarpBackend().is_available()` depends on CUDA-capable GPU presence; cannot be verified on CPU-only environment
- GPU benchmark numbers (steps/s, populations/s, memory consumption): require Colab T4 or similar NVIDIA GPU
- Numerical comparison vs classic MuJoCo on GPU: requires GPU execution
- `jax.jit` / `jax.vmap` / `jax.grad` on Warp backend: JAX transformations don't apply to Warp state (documented in BACKENDS.md)

---

## 5. Verification Results

All quality gates pass on the current state:
- `python3 -m pytest tests_mjx -q` → **13 passed, 3 skipped**
- `python3 -m ruff check .` → **All checks passed**
- `python3 -m ruff format --check .` → **62 files already formatted**
- `python3 -m mypy origami_engine_mjx --config-file pyproject.toml` → **Success: no issues found in 15 source files**
- Link checker script: **all relative markdown links resolve successfully**
- `git diff README.md` → only Documentation section (lines 157–167) and Core modules (lines 180–192) modified, exactly as required

---

## 6. Commits (local, not pushed)

| Commit | Message |
|---|---|
| `1702eba` | fix: drop broken origami-render-mjx entry point; mypy the MJX package in CI |
| `8a2ead4` | fix: point pre-commit mypy hook and 'make typecheck' at origami_engine_mjx |
| `ee4399a` | docs: migrate and enrich documentation for origami_engine_mjx |

Total: 3 commits on `feat/mjx-dual-backend`, all local.