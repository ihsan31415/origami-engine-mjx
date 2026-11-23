# Changelog

All notable changes to Origami Engine (formerly the MuJoCo Origami Simulator)
are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

## [1.1.0] - 2026-11-22

### Added
- Fragmented documentation set under `docs/`: pip/uv install guide, tutorials,
  crease format, simulation, evaluation, API reference, architecture,
  troubleshooting, glossary, and contributing guide
- Quick start guide for `origami-engine` installation
- README docs section linking every dedicated documentation file

### Changed
- Moved the documentation set into `docs/`; files are linked by bare filename
- Expanded the README with project details and features

### Fixed
- Reference crease JSON is now shipped in the wheel
- `origami-render` now writes its output to the current working directory
- Removed a stray empty `EOF` file that was committed by accident
- Bumped the package version (0.1.0 → 0.1.1 → 1.1.0) so the PyPI publish no
  longer collides with the existing 0.1.0 upload and matches the release tag

## [1.0.0] - 2026-11-20

### Added
- `origami_engine/plotting.py` with `plot_pattern(creases, pat=None)` for
  reference-style crease pattern plots (Mountain = crimson dashed,
  Valley = royalblue dash-dot, degree labels, face indices, loop-edge markers,
  legend at lower center)
- `examples/05_plot_pattern.py` runnable plotting snippet
- `docs/crease_pattern_plot.png` committed reference image
- README "Crease pattern visualization" section and reference crease pattern
  table (image on the left, description and JSON reference on the right)
- Quick start installation methods for pip and uv, since the package is on PyPI
- Modular docs split out of the README into `GETTING_STARTED.md` and
  `API_REFERENCE.md`
- `.github/workflows/ci.yml`: ruff, mypy and pytest on Python 3.10/3.11, plus a
  publish-to-PyPI job via OIDC
- `ARCHITECTURE.md` with module overview, data flow, design decisions and a
  public API table
- `CHANGELOG.md` in Keep a Changelog format
- `MANIFEST.in` so the source distribution includes schemas, the docs image,
  examples, experiments and LICENSE
- `tests/conftest.py` shared pytest fixtures
- `pytest-cov` coverage reporting
- `matplotlib` runtime dependency

### Changed
- Renamed the package and repository from `origami-simulator` to
  `origami-engine` (packaging, CI and README updated)
- `pyproject.toml`: classifiers, project URLs, authors, keywords, license-file
  reference, and `pytest-cov` dev dependency
- README rewritten to be library-focused and minimal: banner, badges (release,
  PyPI version, CI, license, Python, mypy, ruff, pre-commit), release link,
  prominent Getting Started, Installation, Quick Start and API Reference
  sections, with the original video tables preserved
- `render_individual.py`: removed the `format="FFMPEG"` argument and let
  `imageio` infer the format

### Fixed
- `ruff format` issues in `README.md`, `plotting.py`, `example.py` and
  `render_individual.py`
- `mypy` error in `render_individual.py` caused by the `imageio` format
  argument type

## [0.8.0-alpha] - 2026-11-20

### Added
- Initial modularized `origami_engine` package (refactored from `origami_sim`):
  `constants.py`, `types.py`, `backend.py`, `geometry.py`, `model.py`,
  `serialization.py`, `dynamics.py`, `simulation.py`, `evaluation.py`,
  `patterns.py`, `render_individual.py`, `example.py`
- `Makefile`, `.pre-commit-config.yaml`, `CONTRIBUTING.md`
- `pyproject.toml` with dev extras (`pytest`, `ruff`, `mypy`, `pre-commit`)
- `examples/` with 4 runnable snippets and a README
- `experiments/` archive: `TIMELINE.md` and `prototype_notebooks.zip`
- README badge row (CI, Python 3.10+, Apache-2.0, ruff, pre-commit, MuJoCo)

[Unreleased]: https://github.com/ihsan31415/origami-engine/compare/1.1.0...HEAD
[1.1.0]: https://github.com/ihsan31415/origami-engine/compare/v1.0.0...1.1.0
[1.0.0]: https://github.com/ihsan31415/origami-engine/compare/v0.8.0-alpha...v1.0.0
[0.8.0-alpha]: https://github.com/ihsan31415/origami-engine/releases/tag/v0.8.0-alpha
