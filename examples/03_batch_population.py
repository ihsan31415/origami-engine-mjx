"""Batched population evaluation with a configurable batch size.

Works with either backend:
    python examples/03_batch_population.py jax
    python examples/03_batch_population.py warp   # on Colab GPU
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
    backend = sys.argv[1] if len(sys.argv) > 1 else "jax"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    engine = OrigamiEngine(backend=backend)
    print(f"backend: {engine.backend_name}  device: {engine.device_info()}")

    # Tiny 2-face pattern for CPU demo speed (16-face model is slow on CPU
    # MJX; on GPU use load_default_creases() for the full Miura pattern).
    creases = [
        Crease((0.2, 0.0), (0.5, 0.5), -0.7853981633974483),
        Crease((0.5, 0.5), (0.8, 0.0), -0.7853981633974483),
    ]
    population = [creases for _ in range(n)]

    t0 = time.time()
    result = engine.evaluate_population(
        population,
        duration=0.005,
        fold_duration=0.003,
        vibrate=False,
        batch_size=32,
    )
    dt = time.time() - t0
    print(f"evaluated {result.n_individuals} individuals in {dt:.1f}s")
    print(f"scores: {result.scores}")
    print(f"throughput: {result.n_individuals / dt:.2f} individuals/s")


if __name__ == "__main__":
    main()
