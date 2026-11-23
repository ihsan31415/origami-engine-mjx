"""Black-box optimization over the fold-angle search space.

Demonstrates the *integration interfaces*: a third-party optimizer drives the
simulation through :class:`SearchSpace` + :class:`OrigamiEvaluator`.  This
library ships **no** optimizer — the algorithm below comes from ``nevergrad``
(install with ``pip install 'origami-engine-mjx[opt]'``).  If ``nevergrad`` is
not installed, the script falls back to a tiny random search so it still runs.

Run:
    python examples/04_optimize_es.py            # nevergrad if available
    python examples/04_optimize_es.py --budget 8

The tiny 2-face pattern is used for CPU demo speed; on a GPU use
``load_default_creases()`` for the full 16-face Miura pattern.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from origami_engine_mjx import Crease
from origami_engine_mjx.integrations import OrigamiEvaluator, SearchSpace
from origami_engine_mjx.integrations.adapters import to_nevergrad_function


def main():
    budget = 8
    if "--budget" in sys.argv:
        budget = int(sys.argv[sys.argv.index("--budget") + 1])

    creases = [
        Crease((0.2, 0.0), (0.5, 0.5), -0.7853981633974483),
        Crease((0.5, 0.5), (0.8, 0.0), -0.7853981633974483),
    ]
    space = SearchSpace.from_creases(creases)
    print(f"search space: {space}")

    # A fast, deterministic evaluation for the demo.
    evaluator = OrigamiEvaluator(
        space,
        backend="jax",
        duration=0.003,
        fold_duration=0.002,
        vibrate=False,
    )
    objective = to_nevergrad_function(evaluator)  # f(x) -> float, minimized

    best_x, best_value = None, float("inf")
    try:
        import nevergrad as ng

        optimizer = ng.optimizers.OnePlusOne(
            parametrization=ng.p.Array(init=np.zeros(space.dim)).set_bounds(space.lower, space.upper),
            budget=budget,
        )
        recommendation = optimizer.minimize(objective, verbosity=0)
        best_x = np.asarray(recommendation.value, dtype=float)
        best_value = float(recommendation.loss)
        print(f"optimizer: nevergrad.OnePlusOne (budget={budget})")
    except ImportError:
        print("nevergrad not installed; falling back to a random search")
        rng = np.random.default_rng(0)
        for _ in range(budget):
            x = space.sample(rng)
            value = objective(x)
            if value < best_value:
                best_x, best_value = x, value
        print(f"optimizer: random search (budget={budget})")

    best_raw = -best_value  # evaluator.maximize=True, so minimized objective == -raw
    print(f"best minimized objective: {best_value:.6g}")
    print(f"best raw forward displacement: {best_raw:.6g} m")
    print(f"best genome: {np.round(space.repair(best_x), 6).tolist()}")


if __name__ == "__main__":
    main()
