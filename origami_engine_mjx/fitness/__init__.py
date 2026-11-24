from __future__ import annotations

"""Fitness building blocks: movement, hinge-target folding and their blend.

The engine's historical score is *forward displacement* (movement).  These
modules add an independent *hinge-target* score — how closely the paper folds
to the angles the design asks for — and a weighted blend of the two:

>>> from origami_engine_mjx.fitness import blend_fitness, FitnessWeights
>>> result = engine.run(creases, duration=2.0)
>>> breakdown = blend_fitness(result, weights=FitnessWeights(movement=70, fold=30))
>>> breakdown.combined, breakdown.movement, breakdown.fold

``fold_weight=0`` (the default) reproduces the movement score exactly, so
existing callers are unaffected.
"""

from ..engine import compute_forward_displacement
from .blend import FitnessBreakdown, FitnessWeights, blend_fitness, blend_scores
from .folding import fold_report, fold_score, hinge_fold_error, hinge_qpos

__all__ = [
    "FitnessBreakdown",
    "FitnessWeights",
    "blend_fitness",
    "blend_scores",
    "compute_forward_displacement",
    "fold_report",
    "fold_score",
    "hinge_fold_error",
    "hinge_qpos",
]
