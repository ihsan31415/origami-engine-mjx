from __future__ import annotations

"""Weighted blend of movement and hinge-target fitness.

The two raw components live on different scales:

* **movement** — forward centroid displacement in metres (can be negative);
* **fold** — a tolerance-normalised hinge score in ``[0, 1]``.

:func:`blend_scores` divides movement by ``movement_scale`` (default ``1.0``,
the sheet side) and then takes a weighted average.  The weights are ordinary
non-negative numbers, so ``movement_weight=70, fold_weight=30`` is the same as
``0.7`` / ``0.3``: only their ratio matters.  With the default
``fold_weight=0`` the blended score is exactly the movement score, preserving
the historical behaviour.
"""

from dataclasses import dataclass, field
from typing import Any

from ..engine import compute_forward_displacement
from ..types import SimulationResult
from .folding import fold_report

__all__ = ["FitnessBreakdown", "FitnessWeights", "blend_fitness", "blend_scores"]


@dataclass(frozen=True)
class FitnessWeights:
    """Relative weights for the movement and folding components.

    Parameters
    ----------
    movement : float
        Weight of the movement (centroid displacement) component.
    fold : float
        Weight of the hinge-target (paper-folding) component.
    movement_scale : float
        Movement is divided by this before blending.  ``1.0`` puts movement in
        units of the sheet side, so a 0.2 m displacement scores ``0.2``.
    """

    movement: float = 1.0
    fold: float = 0.0
    movement_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.movement < 0 or self.fold < 0:
            raise ValueError("movement and fold weights must be non-negative.")
        if self.movement == 0 and self.fold == 0:
            raise ValueError("at least one of movement/fold weights must be positive.")
        if self.movement_scale <= 0:
            raise ValueError("movement_scale must be positive.")

    @property
    def total(self) -> float:
        """Sum of the movement and fold weights."""
        return float(self.movement + self.fold)

    def normalized(self) -> tuple[float, float]:
        """``(movement_fraction, fold_fraction)`` in ``[0, 1]`` summing to 1."""
        total = self.total
        return float(self.movement / total), float(self.fold / total)


@dataclass
class FitnessBreakdown:
    """A blended fitness value plus its components and weights."""

    combined: float
    movement: float
    fold: float
    weights: FitnessWeights
    fold_report: dict[str, Any] = field(default_factory=dict)

    @property
    def movement_fraction(self) -> float:
        return self.weights.normalized()[0]

    @property
    def fold_fraction(self) -> float:
        return self.weights.normalized()[1]

    def as_dict(self) -> dict[str, Any]:
        """Flat dict suitable for logs and JSON checkpoints."""
        return {
            "combined": self.combined,
            "movement": self.movement,
            "fold": self.fold,
            "movement_fraction": self.movement_fraction,
            "fold_fraction": self.fold_fraction,
            "movement_scale": self.weights.movement_scale,
            "fold_mean_abs_error_deg": self.fold_report.get("mean_abs_error_deg"),
            "fold_max_abs_error_deg": self.fold_report.get("max_abs_error_deg"),
        }


def blend_scores(
    movement: float,
    fold: float,
    weights: FitnessWeights | None = None,
    *,
    movement_weight: float = 1.0,
    fold_weight: float = 0.0,
    movement_scale: float = 1.0,
) -> float:
    """Weighted average of a movement score and a fold score in ``[0, 1]``.

    ``weights`` takes precedence when supplied; otherwise the keyword weights
    are used.  Movement is divided by ``movement_scale`` first, so the default
    ``fold_weight=0`` returns the movement score unchanged.
    """
    w = weights if weights is not None else FitnessWeights(movement_weight, fold_weight, movement_scale)
    m_frac, f_frac = w.normalized()
    return float(m_frac * (float(movement) / w.movement_scale) + f_frac * float(fold))


def blend_fitness(
    result: SimulationResult,
    trajectory: tuple[float, ...] = (1.0, 0.0, 0.0),
    weights: FitnessWeights | None = None,
    *,
    movement_weight: float = 1.0,
    fold_weight: float = 0.0,
    movement_scale: float = 1.0,
    actuated=None,
    fold_percent: bool = False,
    zero_tolerance_deg: float = 2.0,
) -> FitnessBreakdown:
    """Compute the movement, fold and blended score for one simulation.

    ``movement`` is :func:`origami_engine_mjx.compute_forward_displacement`;
    ``fold`` is :func:`fold_report`'s ``score``.  Pass ``fold_weight > 0`` to
    optimise for reaching the target fold angles as well as travelling.
    """
    w = weights if weights is not None else FitnessWeights(movement_weight, fold_weight, movement_scale)
    movement = float(compute_forward_displacement(result, trajectory=trajectory))
    report = fold_report(
        result,
        actuated=actuated,
        percent=fold_percent,
        zero_tolerance_deg=zero_tolerance_deg,
    )
    fold = float(report["score"])
    # blend_scores expects fold in [0, 1]; convert if the caller asked for %.
    fold_unit = fold / 100.0 if fold_percent else fold
    combined = blend_scores(movement, fold_unit, w)
    return FitnessBreakdown(combined=combined, movement=movement, fold=fold, weights=w, fold_report=report)
