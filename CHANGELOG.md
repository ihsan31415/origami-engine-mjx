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

### Changed
- `OrigamiEnv` / `OrigamiVectorEnv` accept a `SearchSpace` **or** an angle-only,
  unconstrained `DesignSpace`; a geometry/presence/constrained design space is
  rejected with a clear error (RL needs one shared, fixed topology).

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
