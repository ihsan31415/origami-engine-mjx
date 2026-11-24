# Changelog

All notable changes to Origami Engine (formerly the MuJoCo Origami Simulator)
are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> This changelog covers the `origami-engine-mjx` package only. The legacy
> classic MuJoCo engine (`origami-engine`) is a separate project with its own
> history, so its entries are not listed here.

## [Unreleased]

### Added
- `integrations/design_space.py` — `DesignSpace` / `DesignSpaceBuilder`, a
  **configurable** search space (fold angles, loop-edge angles, crease
  endpoints/geometry, crease presence/count) with the presets `angles_only`,
  `angles_and_geometry`, `geometry_only` and `everything`, selection by
  group/index/slice/list/bool-mask/name/predicate, equality/symmetry constraints
  that reduce the genome dimension, a normalized `[-1, 1]` view, per-variable
  bounds, validity/repair/rejection sampling (with crease-count bounds honoured
  via `presence_count` / `count_ok`), introspection and dict/JSON
  round-tripping. The default `angles_only` preset reproduces `SearchSpace`
  exactly (same vectors, bounds and decoded creases).
- `integrations/population.py` — population **seeding / initialization only**
  (no optimizer): seed-file loading (`load_seeds`), interpolation
  (`interpolate`, `piecewise_path`, `random_barycentric`), random
  initialization (`random_population`, `lhs_population`, `sobol_population`,
  `gaussian_population`), `jitter`, `deduplicate`, `diversity_filter`,
  `mixed_population` with provenance, population I/O (`save_population`,
  `load_population`, `export_creases_dir`) and `population_stats`.
- `OrigamiEvaluator` accepts a `DesignSpace` with no API change: when the
  topology varies between individuals it buckets by `topology_signature` and
  reports the per-individual scoring path (`EvalRecord.path` /
  `BatchResult.paths`: `batched` / `bucketed` / `single` / `invalid` / `cached`).
- `space_spec` adapter helper; `DesignSpace`, `DesignSpaceBuilder`, `Variable`,
  `ANGLE`/`COORD`/`PRESENCE`, `topology_signature` and the population helpers are
  exported from `origami_engine_mjx.integrations`.
- Documentation `docs/SEARCH_SPACE.md` and `docs/POPULATIONS.md`; examples
  `examples/06_design_space.py` and `examples/07_populations.py`; tests
  `tests_mjx/test_design_space.py` and `tests_mjx/test_population.py`.
- `notebook_examples/` — Colab-ready, self-contained integration notebooks:
  black-box optimizers (`nevergrad` / CMA-ES / `pymoo`), a single generic driver
  for every `evosax` strategy, Gymnasium + pure-JAX RL, and an AI/LLM agent
  loop, plus an index README with CPU/GPU cost guidance. The notebooks consume
  the library; they add no algorithms to it.
- `prompts/origami-optimizer-integration.md` — a reusable, algorithm-agnostic
  integration recipe (space selection, evaluator direction, adapters, seeding,
  logging, checkpoints and a verification checklist).
- Trajectory recording: `OrigamiEngine.run(..., record=True, record_every=1)`
  stores `time`/`qpos`/`qvel`/`centroid`/`hinge_qposadr` in
  `result.extra["trajectory"]` (physics and score unchanged), and
  `OrigamiEngine.evaluate(..., return_trajectory=True)` returns
  `(score, trajectory)` from one simulation. Both backends also report
  `info["table_qposadr"]`.
- `origami_engine_mjx/rendering/` — rendering of recorded runs: 26 named
  camera presets (`top`, `side`, `side_left`, `top_side_left`, `iso_top`, …)
  plus manual POVs `(azimuth, elevation[, distance[, zoom]])`, prefixed
  namespaces and JSON camera sidecars (`<stem>.camera.json`); multi-view
  concatenation (`views="all_sides"` = four views in one video); movement
  graphs beside the video (`plot="x"|"y"|"z"|"xyz"` or a list); a freq/amp
  overlay in the bottom-left of **every** frame; MP4 (h264 via `ffmpeg`) or
  GIF output; standalone `plot_trajectory(...)` and `plot_history(...)`.
- `integrations/generation_log.py` — `GenerationLogger` (`end_generation`,
  optional per-individual eval rows, best/mean/std/worst/median, global best,
  per-generation wall time, `RunLogger`-compatible JSONL/CSV, `run.summary.json`,
  `run.best.json` and best-per-generation checkpoints), `GenerationSummary`
  and `load_generation_history`.
- `RunLogger(..., include_generation=True)` adds an optional `generation`
  column/key (default schema unchanged).
- Documentation: rendering section in `docs/SIMULATION.md`, generation logging
  in `docs/EVALUATION.md`, rendering/logging API in `docs/API_REFERENCE.md`;
  tests `tests_mjx/test_rendering_generation.py` (fast) and
  `tests_mjx/test_rendering_video.py` (slow, needs GL + `ffmpeg`).
- Per-hinge actuation: `generate_xml(..., actuated=None)`, `OrigamiEngine.run` /
  `run_batch` and both backends accept `actuated` (`None` = all, a bool, or one
  bool per hinge) and `actuation_schedule=f(t)` evaluated every physics step, so
  hinges can be driven, left passive, or switched on/off during a run. Non-driven
  hinges are not braked to zero; the mask is reported in `info["actuated"]` /
  `info["actuator_hinges"]`, the per-frame trajectory records `"actuated"`, and
  `info["hinge_qpos"]` holds the achieved angles.
- `origami_engine_mjx/fitness/` — `fold_score`, `fold_report`,
  `hinge_fold_error`, `hinge_qpos`, `FitnessWeights`, `blend_scores`,
  `blend_fitness` and `FitnessBreakdown`: a tolerance-normalised **paper-folding
  (hinge-target)** score and a weighted blend with the movement score.
  `fold_weight=0` (default) reproduces the historical movement score. New
  `movement_weight` / `fold_weight` / `movement_scale` arguments on
  `OrigamiEngine.evaluate` / `evaluate_population`, `evaluate_individual` /
  `evaluate_population`, and `OrigamiEvaluator`; descriptors gain `fold_score`
  and `fold_mean_abs_error_deg`; population results carry
  `extra["movement_scores"]` / `extra["fold_scores"]`.
- Documentation `docs/FITNESS.md`; activation/fitness guidance in
  `docs/SIMULATION.md`, `docs/EVALUATION.md`, `docs/INTEGRATIONS.md` and
  `docs/API_REFERENCE.md`; tests `tests_mjx/test_actuation_fitness.py`; example
  `examples/08_fold_fitness.py`.
- `notebook_examples/05_evosax_switchable.ipynb` — one beginner-friendly notebook
  that switches between `evosax` strategies via a single `STRATEGY` variable and
  demonstrates the movement/folding blend; the existing notebooks now install
  from a pinned commit and link to the main repository.

### Changed
- `OrigamiEnv` / `OrigamiVectorEnv` accept a `SearchSpace` **or** an angle-only,
  unconstrained `DesignSpace`; a geometry/presence/constrained design space is
  rejected with a clear error (RL needs one shared, fixed topology).
- Corrected the documented `evosax` direction: it **minimizes**
  `state.best_fitness`, so `to_evosax_fitness` output must be negated before
  `tell` (`docs/OPTIMIZATION_ES.md` and the adapter docstring).

## [1.0.0] - 2026-11-24

### Added
- `origami_engine_mjx/integrations/` — ES/RL/AI **integration interfaces**
  (no optimization or learning algorithms shipped): `SearchSpace` (fixed-
  topology fold-angle genome), `OrigamiEvaluator` (stateless, picklable
  objective + behavior descriptors), optimizer adapters for `pymoo` /
  `nevergrad` / `cma` / `evosax`, Gymnasium `OrigamiEnv` / `OrigamiVectorEnv`,
  pure-JAX `reset` / `step` (jittable / vmappable), and plumbing (seeding,
  genome/checkpoint I/O, JSONL/CSV run logging, throughput benchmark).
- Optional extras `[opt]` (optimizer adapters) and `[rl]` (gymnasium); `[all]`
  now includes them.
- Documentation: `docs/README.md` (index), `docs/INTEGRATIONS.md`,
  `docs/OPTIMIZATION_ES.md`, `docs/RL.md`, and examples
  `examples/04_optimize_es.py` and `examples/05_rl_env.py`.
- Tests: `tests_mjx/test_integrations.py` and `tests_mjx/test_rl_env.py`.

### Removed
- Legacy classic-MuJoCo package `origami_engine/`, its test suite `tests/`,
  the `experiments/` archive, and the five classic example scripts
  (`01_evaluate_reference.py`, `02_json_input.py`, `03_inspect_pattern.py`,
  `04_evaluate_population.py`, `05_plot_pattern.py`). The package has been
  MJX-only since the wheel only ever shipped `origami_engine_mjx*`; the classic
  code remains available on PyPI as `origami-engine` and in this repository's
  git history at the `legacy-classic-engine` tag.
- `make render` (the classic renderer was removed with the legacy package);
  `make example` now runs the MJX CPU quick-start.

## [0.3.0] - 2026-11-23

### Changed
- Bumped the `origami-engine-mjx` package version from 0.1.0 to 0.3.0.

## [0.1.0] - 2026-11-23

### Added
- Initial `origami-engine-mjx` package: the `origami_engine_mjx` dual-backend
  MuJoCo MJX simulator (MJX-JAX + MJX-Warp) with the `OrigamiEngine` facade,
  geometry / model / serialization / plotting modules, the packaged Miura-ori
  reference pattern, the `tests_mjx/` suite, and CPU / Warp examples.

[Unreleased]: https://github.com/ihsan31415/origami-engine-mjx/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/ihsan31415/origami-engine-mjx/compare/v0.3.0...v1.0.0
[0.3.0]: https://github.com/ihsan31415/origami-engine-mjx/compare/v0.1.0...v0.3.0
[0.1.0]: https://github.com/ihsan31415/origami-engine-mjx/releases/tag/v0.1.0
