"""Colab T4 GPU validation + benchmark script for origami-engine-mjx.

Run this entire file as a single Colab cell (after installing the package):

    !pip install 'origami-engine-mjx[all]'

It validates the Warp backend on the T4 GPU, benchmarks both backends at
several population sizes, and prints a summary. No results are pre-claimed —
the numbers you see here are real measurements from your run.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from origami_engine_mjx import OrigamiEngine, load_default_creases
from origami_engine_mjx.backends import available_backends
from origami_engine_mjx.backends.warp_backend import WarpBackend


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}", flush=True)


def main():
    section("0. Environment")
    print("available backends:", available_backends())
    warp = WarpBackend()
    print("warp is_available:", warp.is_available())
    print("warp device:", warp.device_info())
    import jax

    print("jax devices:", jax.devices(), "backend:", jax.default_backend())

    creases = load_default_creases()

    section("1. MJX-Warp: single simulation on GPU")
    engine_w = OrigamiEngine(backend="warp")
    t0 = time.time()
    res = engine_w.run(creases, duration=0.05, fold_duration=0.02, vibrate=True)
    t1 = time.time()
    print(f"warp sim 50ms: {t1 - t0:.1f}s, final centroid={res.final_centroid}")
    assert np.isfinite(np.asarray(res.final_centroid)).all()

    section("2. MJX-JAX on GPU: single simulation")
    engine_j = OrigamiEngine(backend="jax")
    t0 = time.time()
    res = engine_j.run(creases, duration=0.05, fold_duration=0.02, vibrate=True)
    t1 = time.time()
    print(f"jax(gpu) sim 50ms: {t1 - t0:.1f}s, final centroid={res.final_centroid}")

    section("3. Population throughput (short duration, warm-up excluded)")
    for backend in ("jax", "warp"):
        try:
            eng = OrigamiEngine(backend=backend)
        except RuntimeError as e:
            print(f"{backend}: unavailable ({e})")
            continue
        for n in (32, 128, 256):
            pop = [creases] * n
            eng.evaluate_population(pop, duration=0.005, fold_duration=0.002, vibrate=False)  # warm-up
            t0 = time.time()
            r = eng.evaluate_population(pop, duration=0.005, fold_duration=0.002, vibrate=False)
            dt = time.time() - t0
            print(f"  {backend:5s}  n={n:4d}  {dt:6.1f}s  {n / dt:8.2f} ind/s  "
                  f"score[0]={r.scores[0]:.6f}", flush=True)

    section("4. GPU memory")
    try:
        stats = jax.devices()[0].memory_stats()
        print("jax gpu memory_stats:", stats)
    except Exception as e:
        print("memory stats unavailable:", e)
    import subprocess

    subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv"])

    section("DONE — record these numbers; do not claim them without this run.")


if __name__ == "__main__":
    main()
