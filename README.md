<h1 align="center">MuJoCo MJX Origami Engine</h1>

<div align="center">
  <a href="https://github.com/ihsan31415/origami-engine-mjx/releases"><img src="https://img.shields.io/github/v/release/ihsan31415/origami-engine-mjx?display_name=tag&logo=github" alt="Latest Release"></a>
  <a href="https://github.com/ihsan31415/origami-engine-mjx/actions/workflows/ci.yml"><img src="https://github.com/ihsan31415/origami-engine-mjx/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue?logo=python&logoColor=white" alt="Python"></a>
  <a href="https://mujoco.readthedocs.io/en/stable/mjx.html"><img src="https://img.shields.io/badge/Physics-MuJoCo%20MJX-blue" alt="MuJoCo MJX"></a>
  <a href="https://github.com/jax-ml/jax"><img src="https://img.shields.io/badge/Backend-JAX-9cf?logo=google&logoColor=white" alt="JAX"></a>
  <a href="https://github.com/NVIDIA/warp"><img src="https://img.shields.io/badge/Backend-NVIDIA%20Warp-76B900?logo=nvidia&logoColor=white" alt="NVIDIA Warp"></a>
  <br>
  <a href="https://numpy.org/"><img src="https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?logo=numpy&logoColor=white" alt="NumPy"></a>
  <a href="https://shapely.readthedocs.io/"><img src="https://img.shields.io/badge/Geometry-Shapely-3776AB" alt="Shapely"></a>
  <a href="https://docs.pytest.org/"><img src="https://img.shields.io/badge/Tests-pytest-0A9EDC?logo=pytest&logoColor=white" alt="pytest"></a>
  <a href="origami_engine_mjx/schemas/crease_pattern.schema.json"><img src="https://img.shields.io/badge/Validation-JSON%20Schema-green" alt="JSON Schema"></a>
  <br>
  <a href="https://github.com/ihsan31415/origami-engine-mjx/commits/main"><img src="https://img.shields.io/github/last-commit/ihsan31415/origami-engine-mjx" alt="Last Commit"></a>
  <a href="https://github.com/ihsan31415/origami-engine-mjx/issues"><img src="https://img.shields.io/github/issues/ihsan31415/origami-engine-mjx" alt="Issues"></a>
  <a href="https://github.com/ihsan31415/origami-engine-mjx/stargazers"><img src="https://img.shields.io/github/stars/ihsan31415/origami-engine-mjx" alt="Stars"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License"></a>
</div>

<br><br>

This project is a dual-backend **MuJoCo MJX** origami folding simulator — a migration of [`origami-engine`](https://github.com/ihsan31415/origami-engine) from the classic MuJoCo C++ runtime to MJX, with two interchangeable backends in one package. The crease-pattern parsing, geometry reconstruction, and MJCF model generation are **shared** between backends; only the simulation execution differs.

## What this project does
- Keeps the same crease → face → MJCF pipeline as the classic engine (`p1`, `p2`, `angle` data in, MuJoCo model out)
- Runs the simulation on **MJX-JAX** (`mujoco.mjx` on JAX/XLA) or **MJX-Warp** (`mujoco.warp` on NVIDIA Warp) behind a single `OrigamiEngine` facade
- Supports differentiable simulation and `jax.jit` / `jax.vmap` / `jax.lax.scan` on the JAX backend (CPU, NVIDIA GPU, or TPU)
- Supports high-throughput batched population evaluation on the Warp backend (NVIDIA CUDA GPU, e.g. Colab T4)
- Measures the forward displacement of the paper as a fitness signal, for single genomes or whole populations
- Exposes ES / RL / AI **integration interfaces** (a fixed-topology search space, a stateless evaluator, Gymnasium and JAX-functional environments, optimizer adapters) without shipping any optimization or learning algorithm
- Offers a **configurable design space** (`DesignSpace`: fold angles, crease endpoints/geometry, crease count) and **population seeding** helpers (seed files, interpolation, random and mixed initialization) on top of the default fixed-topology search space
- Auto-selects a backend with `backend="auto"` (`warp` if a CUDA GPU is available, otherwise `jax`)

## Backends

| Backend | Flag | Engine | Hardware | Best for |
|---|---|---|---|---|
| **MJX-JAX** | `backend="jax"` | `mujoco.mjx` on JAX/XLA | CPU (local dev), NVIDIA GPU, TPU (Colab) | Differentiable simulation, `jax.jit` / `jax.vmap` / `jax.lax.scan`, CPU testing |
| **MJX-Warp** | `backend="warp"` | `mujoco.warp` on NVIDIA Warp | NVIDIA CUDA GPU only (Colab T4) | High-throughput batched population evaluation |

## Simulation Output

Below is the physical rendering of the simulation output, demonstrating how the paper layout reacts to active hinge joint torques and support table vibrations:

<!-- Image left + description right referencing JSON -->
<table>
  <tr>
    <td style="vertical-align: top; padding-right: 20px;">
      <p align="center">
        <img src="docs/crease_pattern_plot.png" alt="Reference crease pattern" width="300" style="border:1px solid #d0d0d0; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.08);" />
      </p>
      <p align="center"><sub>Reference Miura-ori 4x4 | 16 faces</sub></p>
    </td>
    <td>
      <strong>Reference Crease Pattern</strong><br>
      The default Miura-ori crease pattern shared by both backends, tiling a square sheet into 16 polygonal faces. <br><br>
      <a href="origami_engine_mjx/schemas/crease_pattern.schema.json"><strong>Crease Pattern JSON Schema</strong></a> — validates the structure of crease input files.<br>
      <a href="origami_engine_mjx/crease_patterns/reference_miura_4x4.json"><strong>reference_miura_4x4.json</strong></a> — the reference crease pattern data (p1, p2, and angle per crease).<br><br>
      <em>Mountain folds are negative angles (creasing "up"), Valley folds are positive angles (creasing "down").</em>
    </td>
  </tr>
</table>

<!-- Video table (eyecatching) -->
<table>
  <tr>
    <th>Vibrating Table: Amplitude & Frequency</th>
    <th>Fold Crease</th>
    <th>Video Preview</th>
  </tr>
  <tr>
    <td><strong>Lower</strong> amplitude & frequency</td>
    <td>Partial / standard</td>
    <td>
      <video src="https://github.com/user-attachments/assets/15ff98b3-7a72-46c2-9303-e106c88bc71a"
             width="280" controls muted playsinline></video>
    </td>
  </tr>
  <tr>
    <td><strong>High</strong> amplitude & frequency</td>
    <td>Full</td>
    <td>
      <video src="https://github.com/user-attachments/assets/ef7d9808-d0a5-4c6b-b33c-2c6a81e11003"
             width="280" controls muted playsinline></video>
    </td>
  </tr>
  <tr>
    <td><strong>Higher</strong> amplitude & frequency</td>
    <td>Full</td>
    <td>
      <video src="https://github.com/user-attachments/assets/438714c1-f831-43cb-a729-027816fb3ea5"
             width="280" controls muted playsinline></video>
    </td>
  </tr>
</table>

*This repository is structured as a **library** for crease pattern simulation and analysis on MJX. For full documentation, see the dedicated files below.*

## Quick start

### Install from PyPI

```bash
# Core (geometry, patterns, XML generation — no GPU needed)
pip install origami-engine-mjx

# + MJX-JAX backend (CPU works out of the box)
pip install 'origami-engine-mjx[jax]'

# + MJX-Warp backend (requires NVIDIA CUDA GPU)
pip install 'origami-engine-mjx[warp]'

# Everything
pip install 'origami-engine-mjx[all]'
```

### Run a simulation

```python
from origami_engine_mjx import OrigamiEngine, load_default_creases

# CPU (local dev) — MJX-JAX
engine = OrigamiEngine(backend="jax")
creases = load_default_creases()

# Single simulation -> forward displacement (m)
score = engine.evaluate(creases, duration=2.0)
print(f"forward displacement: {score:.4f} m")

# Batch population evaluation
population = [creases, creases, creases]  # replace with your genomes
result = engine.evaluate_population(population, duration=2.0)
print(result.scores)
```

On Google Colab with an NVIDIA T4:

```python
engine = OrigamiEngine(backend="warp")  # CUDA GPU via MuJoCo Warp
result = engine.evaluate_population(population, duration=8.0, batch_size=128)
```

### Validate (source installation)

```bash
pip install -e '.[dev,jax]'
pytest tests_mjx -q              # fast tests (no long sims)
pytest tests_mjx -q -m slow      # includes short MJX-CPU simulations
```

## Performance notes

- **MJX-JAX on CPU is slow for mesh-contact models**: the 16-face reference Miura model costs ~70 s/step on CPU (mesh–mesh collision in XLA). Tiny models run at ~1.4 s/step. Use CPU only for correctness tests with short durations; use GPU for real workloads.
- **JIT compilation of `mjx.step` is expensive** for this model; the JAX backend runs eager by default (`jit=False`).
- On a Colab T4 GPU, both backends should be dramatically faster — see `examples/colab_gpu_validation.py` and the benchmark in [COLAB_GPU.md](docs/COLAB_GPU.md).

## 📖 Documentation

For in-depth information, see the dedicated documentation files:

- **[GETTING_STARTED.md](docs/GETTING_STARTED.md)** — installation and first run
- **[TUTORIALS.md](docs/TUTORIALS.md)** — nine step-by-step recipes
- **[CREASE_FORMAT.md](docs/CREASE_FORMAT.md)** — crease & JSON schema reference
- **[API_REFERENCE.md](docs/API_REFERENCE.md)** — complete API signatures
- **[SIMULATION.md](docs/SIMULATION.md)** — model structure, stepping loop, rendering
- **[EVALUATION.md](docs/EVALUATION.md)** — fitness functions and population evaluation
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** — package layout, data flow, backend responsibilities
- **[BACKENDS.md](docs/BACKENDS.md)** — MJX-JAX vs MJX-Warp: semantics, features, limitations
- **[COLAB_GPU.md](docs/COLAB_GPU.md)** — Colab T4 setup, validation checklist, benchmarking
- **[MIGRATION.md](docs/MIGRATION.md)** — migrating from `origami-engine` (classic MuJoCo)
- **[TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** — runtime problems and guidance
- **[GLOSSARY.md](docs/GLOSSARY.md)** — terminology used across the codebase
- **[CONTRIBUTING.md](docs/CONTRIBUTING.md)** — how to contribute, tests, CI
- **[INTEGRATIONS.md](docs/INTEGRATIONS.md)** — ES/RL/AI integration interfaces overview
- **[OPTIMIZATION_ES.md](docs/OPTIMIZATION_ES.md)** — search space, evaluator, optimizer adapters, QD descriptors
- **[SEARCH_SPACE.md](docs/SEARCH_SPACE.md)** — the configurable `DesignSpace` (angles, geometry, crease count)
- **[POPULATIONS.md](docs/POPULATIONS.md)** — population seeding / initialization
- **[RL.md](docs/RL.md)** — Gymnasium and JAX-functional reinforcement-learning environments
- **[notebook_examples/](notebook_examples/README.md)** — Colab-ready notebooks for external optimizers, RL and LLM agents
- `examples/` — runnable scripts (see the table below)

*All documentation lives in [`docs/`](docs).*

## Examples

| Script | Demonstrates |
|---|---|
| `examples/01_evaluate_cpu_jax.py` | Single sim + population on CPU (MJX-JAX) |
| `examples/02_evaluate_warp_gpu.py` | MJX-Warp on NVIDIA GPU |
| `examples/03_batch_population.py` | Batch population evaluation with configurable batch size |
| `examples/colab_gpu_validation.py` | Full GPU validation + benchmark script for Colab T4 |

## Core modules (brief)

- `origami_engine_mjx/constants.py` — physical / solver constants (shared)
- `origami_engine_mjx/types.py` — `Crease`, `Pattern`, `SimulationResult`, `PopulationResult`
- `origami_engine_mjx/geometry.py` — crease graph → faces → spanning tree (shared, Shapely)
- `origami_engine_mjx/model.py` — `generate_xml()`: Pattern → MJCF XML (shared, no MuJoCo needed)
- `origami_engine_mjx/serialization.py` — JSON crease I/O (shared)
- `origami_engine_mjx/patterns.py` — reference Miura patterns (shared)
- `origami_engine_mjx/plotting.py` — crease-pattern plots (shared)
- `origami_engine_mjx/backend.py` — lazy import guards for mujoco / mjx-jax / mjx-warp
- `origami_engine_mjx/engine.py` — `OrigamiEngine`, the public facade with backend selection
- `origami_engine_mjx/evaluation.py` — `evaluate_individual` / `evaluate_population`
- `origami_engine_mjx/integrations/` — ES/RL/AI interfaces (no algorithms): `search_space.py` (`SearchSpace`), `design_space.py` (`DesignSpace`, the configurable angles/geometry/crease-count space), `evaluator.py` (`OrigamiEvaluator`, descriptors), `population.py` (seed loading, interpolation, random / mixed initialization, population I/O), `adapters.py` (pymoo / nevergrad / cma / evosax), `env.py` (`OrigamiEnv`, `OrigamiVectorEnv`), `jax_env.py` (pure-JAX `reset` / `step`), `plumbing.py` (seeding, genome/checkpoint I/O, run logging, benchmark)
- `origami_engine_mjx/backends/` — `base.py` (`SimulationBackend` ABC), `jax_backend.py` (`put_model` / `make_data` / `mjx.step`, optional jit), `warp_backend.py` (`mjw.make_model` / `make_data` / `mjw.step`, GPU), and a registry with `get_backend(name)`
- `origami_engine_mjx/schemas/` — JSON schema for crease files
- `origami_engine_mjx/crease_patterns/` — `reference_miura_4x4.json`

## Library API

```bash
pip install -e ".[dev,jax]"

python -c "
from origami_engine_mjx import (
    OrigamiEngine,
    Crease,
    build_pattern,
    load_default_creases,
)
"
```

*For the complete layout and data flow, see [ARCHITECTURE.md](docs/ARCHITECTURE.md).*

## Optimization & RL interfaces

The library is the **environment / evaluator substrate** for external search and learning tools — it ships **no** optimization or learning algorithm. `origami_engine_mjx.integrations` provides a fixed-topology `SearchSpace` (15 actuated fold angles for the default Miura pattern), a stateless picklable `OrigamiEvaluator` with behavior descriptors for quality-diversity, a Gymnasium `OrigamiEnv` plus pure-JAX `reset`/`step` (jittable / vmappable), thin adapters for `pymoo` / `nevergrad` / `cma` / `evosax`, and plumbing (seeding, genome/checkpoint I/O, run logging, throughput benchmark). The search space is configurable — `DesignSpace` can also vary loop-edge angles, crease endpoints (geometry) and crease count, with presets, selection and equality/symmetry constraints — and initial populations can be seeded from crease JSON files or generated by interpolation, random and mixed sampling. See [INTEGRATIONS.md](docs/INTEGRATIONS.md), [OPTIMIZATION_ES.md](docs/OPTIMIZATION_ES.md), [SEARCH_SPACE.md](docs/SEARCH_SPACE.md), [POPULATIONS.md](docs/POPULATIONS.md) and [RL.md](docs/RL.md).

## License

Apache-2.0
