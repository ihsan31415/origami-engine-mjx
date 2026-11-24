from __future__ import annotations

"""Hinge-target (paper-folding) fitness.

The movement score asks *how far did the sheet travel*; this module asks *did
the paper actually reach the fold angles it was told to fold to*.  Both are
useful, and :mod:`origami_engine_mjx.fitness.blend` combines them with weights.

All scores are higher-is-better.  ``fold_score`` is normalised so that ``1.0``
(or ``100`` with ``percent=True``) means every selected hinge is within a small
tolerance of its target angle, and ``0.0`` means no hinge made progress.
"""

from typing import Any

import numpy as np

from ..model import fold_score as _fold_score
from ..model import hinge_fold_error as _hinge_fold_error
from ..types import SimulationResult

__all__ = [
    "fold_report",
    "fold_score",
    "hinge_fold_error",
    "hinge_qpos",
]


def hinge_qpos(result: SimulationResult) -> np.ndarray:
    """Achieved hinge angles (radians) from a finished simulation.

    Reads ``info["hinge_qpos"]`` when the backend stored it, otherwise falls
    back to indexing the final ``qpos`` with ``info["hinge_qposadr"]``.
    """
    stored = result.info.get("hinge_qpos")
    if stored is not None:
        return np.asarray(stored, dtype=float).reshape(-1)
    adr = np.asarray(result.info.get("hinge_qposadr", []), dtype=int).reshape(-1)
    qpos = np.asarray(result.data.qpos, dtype=float).reshape(-1)
    return qpos[adr]


def fold_score(
    result: SimulationResult, actuated=None, percent: bool = False, zero_tolerance_deg: float = 2.0
) -> float:
    """Tolerance-normalised hinge-target score for a finished simulation.

    Returns a single number (``0..1``, or ``0..100`` with ``percent=True``).
    See :func:`origami_engine_mjx.model.fold_score` for the exact formula and
    :func:`fold_report` for the per-hinge breakdown.  ``actuated`` selects
    hinges (``None`` scores every hinge, including passive ones — their target
    is still the angle the design wants).
    """
    report = _fold_score(
        hinge_qpos(result),
        result.info,
        actuated=actuated,
        percent=percent,
        zero_tolerance_deg=zero_tolerance_deg,
    )
    return float(report["score"])


def hinge_fold_error(result: SimulationResult, actuated=None) -> np.ndarray:
    """Signed ``achieved - target`` hinge error in radians."""
    return _hinge_fold_error(hinge_qpos(result), result.info, actuated=actuated)


def fold_report(
    result: SimulationResult,
    actuated=None,
    percent: bool = False,
    zero_tolerance_deg: float = 2.0,
) -> dict[str, Any]:
    """Full folding report: score, errors and the achieved/target angles.

    Returns
    -------
    dict
        ``score``, ``mean_abs_error_deg``, ``max_abs_error_deg``,
        ``rms_error_deg``, ``n_hinges``, ``per_hinge_error_deg``,
        ``achieved_deg`` and ``target_deg`` (the last two over *all* hinges).
    """
    report = _fold_score(
        hinge_qpos(result),
        result.info,
        actuated=actuated,
        percent=percent,
        zero_tolerance_deg=zero_tolerance_deg,
    )
    report["achieved_deg"] = np.degrees(hinge_qpos(result))
    report["target_deg"] = np.degrees(np.asarray(result.info["targets"], dtype=float))
    return report
