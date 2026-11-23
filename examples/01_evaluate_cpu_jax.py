"""Single simulation + population evaluation on CPU using MJX-JAX.

Run:  python examples/01_evaluate_cpu_jax.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from origami_engine_mjx import Crease, OrigamiEngine


def main():
    engine = OrigamiEngine(backend="jax")
    print(f"backend: {engine.backend_name}  device: {engine.device_info()}")

    # Tiny 2-face pattern: MJX on CPU costs ~70 s/step for the 16-face
    # reference mesh model, so the CPU example uses a minimal pattern.
    # Use load_default_creases() on GPU (Colab) for the full model.
    creases = [
        Crease((0.2, 0.0), (0.5, 0.5), -0.7853981633974483),
        Crease((0.5, 0.5), (0.8, 0.0), -0.7853981633974483),
    ]

    t0 = time.time()
    score = engine.evaluate(creases, duration=0.005, fold_duration=0.003, vibrate=False)
    print(f"single individual forward displacement: {score:.6f} m  ({time.time() - t0:.1f}s)")

    population = [creases, creases]
    t0 = time.time()
    result = engine.evaluate_population(population, duration=0.005, fold_duration=0.003, vibrate=False)
    print(f"population scores: {result.scores}  ({time.time() - t0:.1f}s, backend={result.backend})")


if __name__ == "__main__":
    main()
