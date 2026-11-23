# origami-engine-mjx

Dual-backend **MuJoCo MJX** origami folding simulator — a migration of `origami-engine` from the classic MuJoCo C++ runtime to MJX, with two interchangeable backends in one package:

| Backend | Flag | Engine | Hardware | Best for |
|---|---|---|---|---|
| **MJX-JAX** | `backend="jax"` | `mujoco.mjx` on JAX/XLA | CPU (local dev), NVIDIA GPU, TPU (Colab) | Differentiable simulation, `jax.jit` / `jax.vmap` / `jax.lax.scan`, CPU testing |
| **MJX-Warp** | `backend="warp"` | `mujoco.warp` on NVIDIA Warp | NVIDIA CUDA GPU only (Colab T4) | High-throughput batched population evaluation |

The crease-pattern parsing, geometry reconstruction, and MJCF model generation are **shared** between backends — only the simulation execution differs.

## Install

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

## Quick start

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

`backend="auto"` picks `warp` if a CUDA GPU is available, otherwise `jax`.

## Architecture

```
origami_engine_mjx/
├── __init__.py            # public API: OrigamiEngine, Crease, build_pattern, ...
├── constants.py           # physical / solver constants (shared)
├── types.py               # Crease, Pattern, SimulationResult, PopulationResult
├── geometry.py            # crease graph -> faces -> spanning tree (shared, Shapely)
├── model.py               # generate_xml(): Pattern -> MJCF XML (shared, no MuJoCo needed)
├── serialization.py       # JSON crease I/O (shared)
├── patterns.py            # reference Miura patterns (shared)
├── plotting.py            # crease-pattern plots (shared)
├── backend.py             # lazy import guards for mujoco / mjx-jax / mjx-warp
├── engine.py              # OrigamiEngine — public facade, backend selection
├── evaluation.py          # evaluate_individual / evaluate_population
├── backends/
│   ├── base.py            # SimulationBackend ABC
│   ├── __init__.py        # registry + get_backend(name)
│   ├── jax_backend.py     # MJX-JAX: put_model / make_data / mjx.step, jit option
│   └── warp_backend.py    # MJX-Warp: mjw.make_model / make_data / mjw.step (GPU)
├── schemas/               # JSON schema for crease files
└── crease_patterns/       # reference_miura_4x4.json
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for details, [docs/BACKENDS.md](docs/BACKENDS.md) for a backend comparison, and [docs/COLAB_GPU.md](docs/COLAB_GPU.md) for GPU validation.

## Performance notes

- **MJX-JAX on CPU is slow for mesh-contact models**: the 16-face reference Miura model costs ~70 s/step on CPU (mesh–mesh collision in XLA). Tiny models run at ~1.4 s/step. Use CPU only for correctness tests with short durations; use GPU for real workloads.
- **JIT compilation of `mjx.step` is expensive** for this model; the JAX backend runs eager by default (`jit=False`).
- On a Colab T4 GPU, both backends should be dramatically faster — see `examples/colab_gpu_validation.py` and benchmark in `docs/COLAB_GPU.md`.

## Examples

| Script | Demonstrates |
|---|---|
| `examples/01_evaluate_cpu_jax.py` | Single sim + population on CPU (MJX-JAX) |
| `examples/02_evaluate_warp_gpu.py` | MJX-Warp on NVIDIA GPU |
| `examples/03_batch_population.py` | Batch population evaluation with configurable batch size |
| `examples/colab_gpu_validation.py` | Full GPU validation + benchmark script for Colab T4 |

## Tests

```bash
pip install -e '.[dev,jax]'
pytest tests_mjx -q              # fast tests (no long sims)
pytest tests_mjx -q -m slow      # includes short MJX-CPU simulations
```

## Docs

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — package layout, data flow, backend responsibilities
- [docs/BACKENDS.md](docs/BACKENDS.md) — MJX-JAX vs MJX-Warp: semantics, features, limitations
- [docs/COLAB_GPU.md](docs/COLAB_GPU.md) — Colab T4 setup, validation checklist, benchmarking
- [docs/MIGRATION.md](docs/MIGRATION.md) — migrating from `origami-engine` (classic MuJoCo)

## License

Apache-2.0
