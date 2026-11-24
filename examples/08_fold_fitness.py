"""Movement vs. folding fitness and per-hinge actuation.

Shows the two fitness components on a real (short) simulation, how to blend them
with any percentage split, and how to drive only some hinges (or switch them
off part-way through a run).

The demo uses the small 2x2 Miura pattern (6 creases -> 4 hinges) because the
16-face reference pattern is very slow on a CPU. Every run is a few
milliseconds of simulated time, so the script finishes quickly; raise
``DURATION`` on a GPU.

Run:
    MUJOCO_GL=egl python examples/08_fold_fitness.py

See ``docs/FITNESS.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from origami_engine_mjx import (
    Crease,
    FitnessWeights,
    OrigamiEngine,
    blend_fitness,
    fold_report,
    fold_score,
)

DURATION = 0.01  # simulated seconds per run (short: CPU-friendly)
FOLD_DURATION = 0.006  # seconds spent ramping the hinges to their targets

# 2x2 Miura: 6 creases, of which 4 become hinges.
CREASES = [
    Crease((0.1, 0.1), (0.1, 0.5), 0.8),
    Crease((0.1, 0.5), (0.5, 0.5), -0.8),
    Crease((0.5, 0.5), (0.5, 0.1), 0.8),
    Crease((0.5, 0.1), (0.1, 0.1), -0.8),
    Crease((0.1, 0.1), (0.5, 0.5), -0.8),
    Crease((0.1, 0.5), (0.5, 0.1), 0.8),
]


def main():
    engine = OrigamiEngine(backend="jax")

    # 1. One short run. `fold_report` explains a low fold score.
    result = engine.run(CREASES, duration=DURATION, fold_duration=FOLD_DURATION, vibrate=False)
    report = fold_report(result, percent=True)
    movement = engine.evaluate(CREASES, duration=DURATION, fold_duration=FOLD_DURATION, vibrate=False)
    print(f"movement (m)     : {movement:+.5f}")
    print(f"fold score       : {report['score']:.1f} / 100   ({report['n_hinges']} hinges)")
    print(f"mean hinge error : {report['mean_abs_error_deg']:.2f} deg")
    print(f"per hinge error  : {np.round(report['per_hinge_error_deg'], 2).tolist()}")
    print("(a few milliseconds is far too short to fold: raise DURATION to watch the score climb)")

    # 2. Blend the two components — weights are relative (70 / 30 = 70 % / 30 %).
    weights = FitnessWeights(movement=70, fold=30)
    breakdown = blend_fitness(result, weights=weights)
    print(
        "\nblend            : "
        f"combined={breakdown.combined:+.5f}  movement={breakdown.movement:+.5f}  fold={breakdown.fold:.4f}"
    )
    print(f"split            : {breakdown.movement_fraction:.0%} movement / {breakdown.fold_fraction:.0%} folding")

    # 3. The same split through the engine facade. fold_weight=0 is the old score.
    blended = engine.evaluate(
        CREASES,
        duration=DURATION,
        fold_duration=FOLD_DURATION,
        vibrate=False,
        movement_weight=70,
        fold_weight=30,
    )
    print(f"\nmovement only    : {movement:+.5f}")
    print(f"70/30 blend      : {blended:+.5f}")

    # 4. Drive only the first two hinges; the rest are passive (still scored).
    n_hinges = len(result.info["hinges"])
    actuated = [True, True] + [False] * (n_hinges - 2)
    passive_run = engine.run(
        CREASES,
        duration=DURATION,
        fold_duration=FOLD_DURATION,
        vibrate=False,
        actuated=actuated,
    )
    all_hinges = fold_score(passive_run, percent=True)
    driven = fold_score(passive_run, actuated=actuated, percent=True)
    print(f"\nhinges driven    : {sum(actuated)} / {n_hinges}")
    print(f"fold score (all hinges)   : {all_hinges:.1f} / 100")
    print(f"fold score (driven hinges): {driven:.1f} / 100")

    # 5. Switch actuation off half-way: fold for a while, then let the paper relax.
    scheduled = engine.run(
        CREASES,
        duration=DURATION,
        fold_duration=FOLD_DURATION,
        vibrate=False,
        actuation_schedule=lambda t: t < FOLD_DURATION,
        record=True,
    )
    mask = scheduled.extra["trajectory"]["actuated"]
    print(f"\nscheduled mask   : {mask.shape[0]} frames x {mask.shape[1]} hinges")
    print(f"driven at start  : {int(mask[0].sum())} hinges")
    print(f"driven at end    : {int(mask[-1].sum())} hinges")


if __name__ == "__main__":
    main()
