"""MJX-Warp single simulation on an NVIDIA GPU.

Run on Google Colab (T4) or any CUDA machine:
    python examples/02_evaluate_warp_gpu.py

Requires:  pip install 'origami-engine-mjx[warp]'
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from origami_engine_mjx import OrigamiEngine, load_default_creases


def main():
    engine = OrigamiEngine(backend="warp")  # raises RuntimeError without CUDA
    print(f"backend: {engine.backend_name}  device: {engine.device_info()}")

    creases = load_default_creases()
    score = engine.evaluate(creases, duration=1.0, fold_duration=0.3)
    print(f"forward displacement (warp/GPU): {score:.6f} m")


if __name__ == "__main__":
    main()
