# Getting Started

> **Documentation map:** [README](../README.md) · **Getting Started** · [Tutorials](TUTORIALS.md) · [Crease Format](CREASE_FORMAT.md) · [Simulation](SIMULATION.md) · [Evaluation](EVALUATION.md) · [API Reference](API_REFERENCE.md) · [Architecture](ARCHITECTURE.md) · [Troubleshooting](TROUBLESHOOTING.md) · [Glossary](GLOSSARY.md) · [Contributing](CONTRIBUTING.md)

This guide covers how to install **origami-engine-mjx** and get started with the MJX-JAX (CPU/GPU/TPU) and MJX-Warp (NVIDIA CUDA GPU) backends.

## Quick start (Python)

```python
from origami_engine_mjx import OrigamiEngine, load_default_creases

# CPU via MJX-JAX (works out of the box, no GPU needed)
engine = OrigamiEngine(backend="jax")
creases = load_default_creases()

# Single simulation -> forward displacement (m)
score = engine.evaluate(creases, duration=2.0)
print(f"forward displacement: {score:.4f} m  (backend={engine.backend_name})")

# Batch population evaluation
population = [creases, creases, creases]
result = engine.evaluate_population(population, duration=2.0)
print(result.scores)
```

## Installation

The package is available from PyPI. Choose the installation that matches your needs:

| Goal | Command |
|---|---|
| Core (geometry, patterns, XML generation — no GPU needed) | `pip install origami-engine-mjx` |
| + MJX-JAX backend (CPU works out of the box) | `pip install 'origami-engine-mjx[jax]'` |
| + MJX-Warp backend (requires NVIDIA CUDA GPU) | `pip install 'origami-engine-mjx[warp]'` |
| Everything (JAX + Warp + rendering) | `pip install 'origami-engine-mjx[all]'` |

### Development install (editable, with all deps)

```bash
git clone https://github.com/ihsan31415/origami-engine-mjx.git
cd origami-engine-mjx
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,jax,warp,render]"
```

### Verifying the install

```bash
python -c "import origami_engine_mjx; print(origami_engine_mjx.__version__)"
# 0.1.0

python -c "
from origami_engine_mjx.backends import available_backends
print('available backends:', available_backends())
from origami_engine_mjx import OrigamiEngine
engine = OrigamiEngine(backend='auto')
print('selected backend:', engine.backend_name)
print('device info:', engine.device_info())
"
```

### Backend selection

```python
from origami_engine_mjx import OrigamiEngine

# Explicit JAX (CPU works, GPU/TPU if JAX devices available)
engine_j = OrigamiEngine(backend="jax")

# Explicit Warp (raises RuntimeError without CUDA GPU)
try:
    engine_w = OrigamiEngine(backend="warp")
except RuntimeError as e:
    print("Warp not available without GPU:", e)

# Auto-select: warp if CUDA GPU present, otherwise jax
engine_a = OrigamiEngine(backend="auto")
```

## Your first simulation (CPU, MJX-JAX)

```python
from origami_engine_mjx import OrigamiEngine, load_default_creases

engine = OrigamiEngine(backend="jax")
creases = load_default_creases()

# Single simulation -> forward displacement (m)
score = engine.evaluate(creases, duration=2.0, fold_duration=3.0, vibrate=False)
print(f"forward displacement: {score:.6f} m  (backend={engine.backend_name})")

# Or via the engine.run() / compute_forward_displacement() pipeline:
result = engine.run(creases, duration=2.0)
displacement = result.final_centroid[0] - result.initial_centroid[0]  # crude +x estimate
```

## Your first simulation (GPU, MJX-Warp on Google Colab T4)

```python
engine = OrigamiEngine(backend="warp")  # raises RuntimeError without CUDA GPU
creases = load_default_creases()
result = engine.run(creases, duration=8.0, vibrate=True)
score = engine.evaluate(creases, trajectory=(1.0, 0.0, 0.0))
print(f"forward displacement (warp/GPU): {score:.6f} m")
```

## Next steps

- Step-by-step recipes: [TUTORIALS.md](TUTORIALS.md)
- Crease input format and JSON Schema: [CREASE_FORMAT.md](CREASE_FORMAT.md)
- How the simulation is built and rendered: [SIMULATION.md](SIMULATION.md)
- Fitness / population evaluation: [EVALUATION.md](EVALUATION.md)
- Full API signatures: [API_REFERENCE.md](API_REFERENCE.md)
- Install/runtime problems: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- Package layout and data flow: [ARCHITECTURE.md](ARCHITECTURE.md)