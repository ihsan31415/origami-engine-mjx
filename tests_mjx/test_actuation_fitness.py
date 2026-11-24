"""Actuation control (per-hinge, per-step) and fold/movement fitness blend.

The fast, physics-free tests cover MJCF generation and the scoring maths; the
slow tests run the tiny 2-crease pattern through MJX-JAX on CPU.
"""

import numpy as np
import pytest

from origami_engine_mjx import (
    Crease,
    FitnessWeights,
    blend_scores,
    build_pattern,
    compute_forward_displacement,
    fold_report,
    fold_score,
    generate_xml,
    hinge_fold_error,
)
from origami_engine_mjx.model import resolve_actuated
from origami_engine_mjx.types import SimulationResult


@pytest.fixture(scope="module")
def tiny_creases():
    """Minimal valid pattern: one mountain fold -> 2 faces, 1 hinge."""
    return [
        Crease((0.2, 0.0), (0.5, 0.5), -0.7853981633974483),
        Crease((0.5, 0.5), (0.8, 0.0), -0.7853981633974483),
    ]


@pytest.fixture(scope="module")
def miura_creases():
    """2x2 Miura: 6 creases -> 4 actuated tree hinges."""
    return [
        Crease((0.1, 0.1), (0.1, 0.5), 0.8),
        Crease((0.1, 0.5), (0.5, 0.5), -0.8),
        Crease((0.5, 0.5), (0.5, 0.1), 0.8),
        Crease((0.5, 0.1), (0.1, 0.1), -0.8),
        Crease((0.1, 0.1), (0.5, 0.5), -0.8),
        Crease((0.1, 0.5), (0.5, 0.1), 0.8),
    ]


def _fake_result(qpos_by_hinge, targets):
    """Build a SimulationResult with only the fields the fold score reads."""
    n = len(targets)
    hinge_qposadr = np.arange(1, n + 1)
    qpos = np.zeros(n + 1)
    qpos[hinge_qposadr] = qpos_by_hinge
    return SimulationResult(
        model=None,
        data={"qpos": qpos},
        pattern=None,  # type: ignore[arg-type]
        info={
            "targets": np.asarray(targets, dtype=float),
            "hinge_qposadr": hinge_qposadr,
            "hinge_qpos": np.asarray(qpos_by_hinge, dtype=float),
        },
        initial_centroid=np.zeros(2),
        final_centroid=np.zeros(2),
    )


# ---------------------------------------------------------------------------
# XML generation: which hinges get actuators
# ---------------------------------------------------------------------------


def test_default_actuates_every_hinge(miura_creases):
    pat = build_pattern(miura_creases)
    xml, info = generate_xml(pat, miura_creases)
    assert info["n_hinges"] == 4
    assert len(info["actuator_hinges"]) == 4
    assert xml.count('<position name="fold_') == 4
    assert info["actuated"].all()


def test_actuated_mask_selects_hinges(miura_creases):
    pat = build_pattern(miura_creases)
    xml, info = generate_xml(pat, miura_creases, actuated=[True, False, True, False])
    assert list(info["actuator_hinges"]) == [0, 2]
    assert list(info["actuated"]) == [True, False, True, False]
    assert '<position name="fold_0"' in xml
    assert '<position name="fold_1"' not in xml
    assert '<position name="fold_2"' in xml
    assert '<position name="fold_3"' not in xml


def test_actuated_false_emits_no_actuator_block(miura_creases):
    pat = build_pattern(miura_creases)
    xml, info = generate_xml(pat, miura_creases, actuated=False)
    assert "<actuator>" not in xml
    assert info["actuator_hinges"].size == 0
    # Targets are still recorded, so passive hinges can still be scored.
    assert info["targets"].shape == (4,)


def test_actuated_wrong_length_raises(miura_creases):
    pat = build_pattern(miura_creases)
    with pytest.raises(ValueError, match="expected one per hinge"):
        generate_xml(pat, miura_creases, actuated=[True, False])


def test_resolve_actuated_forms():
    assert resolve_actuated(None, 3).tolist() == [True, True, True]
    assert resolve_actuated(True, 2).tolist() == [True, True]
    assert resolve_actuated(False, 2).tolist() == [False, False]
    assert resolve_actuated([1, 0, 1], 3).tolist() == [True, False, True]


# ---------------------------------------------------------------------------
# Fold score maths (no physics)
# ---------------------------------------------------------------------------


def test_fold_score_perfect_fold():
    result = _fake_result([0.5, -0.5], [0.5, -0.5])
    report = fold_report(result)
    assert report["score"] == pytest.approx(1.0)
    assert report["mean_abs_error_deg"] == pytest.approx(0.0)
    assert fold_score(result, percent=True) == pytest.approx(100.0)


def test_fold_score_unfolded_paper_is_zero():
    result = _fake_result([0.0, 0.0], [0.5, -0.5])
    assert fold_score(result) == pytest.approx(0.0)


def test_fold_score_never_rewards_overshoot():
    result = _fake_result([1.0], [0.2])  # folded far past the target
    assert fold_score(result) == pytest.approx(0.0)


def test_fold_score_matches_model_helper():
    from origami_engine_mjx.model import fold_score as model_fold_score

    result = _fake_result([0.5], [0.5])
    expected = model_fold_score(result.info["hinge_qpos"], result.info)["score"]
    assert fold_score(result) == pytest.approx(expected)


def test_hinge_fold_error_is_signed():
    result = _fake_result([0.4, -0.1], [0.5, -0.2])
    err = hinge_fold_error(result)
    assert err == pytest.approx([-0.1, 0.1])


def test_fold_score_actuated_subset():
    result = _fake_result([0.5, 0.0], [0.5, -0.5])
    assert fold_score(result, actuated=[True, False]) == pytest.approx(1.0)
    assert fold_score(result, actuated=[False, True]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Weights and blending (no physics)
# ---------------------------------------------------------------------------


def test_fitness_weights_fractions():
    w = FitnessWeights(movement=70, fold=30)
    assert w.normalized() == pytest.approx((0.7, 0.3))
    # Only the ratio matters.
    assert FitnessWeights(movement=0.7, fold=0.3).normalized() == pytest.approx(w.normalized())


def test_fitness_weights_validation():
    with pytest.raises(ValueError):
        FitnessWeights(movement=-1)
    with pytest.raises(ValueError):
        FitnessWeights(movement=0, fold=0)
    with pytest.raises(ValueError):
        FitnessWeights(movement_scale=0)


def test_blend_scores_is_weighted_average():
    assert blend_scores(0.4, 0.8, movement_weight=1.0, fold_weight=1.0) == pytest.approx(0.6)
    assert blend_scores(0.4, 0.8, movement_weight=3.0, fold_weight=1.0) == pytest.approx((3.0 * 0.4 + 1.0 * 0.8) / 4.0)


def test_blend_scores_zero_fold_weight_is_movement():
    assert blend_scores(0.123, 0.0) == pytest.approx(0.123)


def test_blend_scores_movement_scale():
    # 0.2 m over a 0.5 m scale == 0.4 of the sheet.
    assert blend_scores(0.2, 0.0, movement_weight=1.0, fold_weight=1.0, movement_scale=0.5) == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Physics: actuation flags and the fold score on a real run
# ---------------------------------------------------------------------------

jax = pytest.importorskip("jax")
pytest.importorskip("mujoco.mjx")

from origami_engine_mjx import OrigamiEngine, blend_fitness
from origami_engine_mjx.integrations import DesignSpace, OrigamiEvaluator

RUN = {"duration": 0.003, "fold_duration": 0.002, "vibrate": False}


@pytest.mark.slow
def test_passive_run_reports_fold_state(tiny_creases):
    engine = OrigamiEngine(backend="jax")
    result = engine.run(tiny_creases, actuated=False, **RUN)
    assert result.info["actuated"].tolist() == [False]
    assert np.asarray(result.info["hinge_qpos"]).shape == (1,)
    report = fold_report(result, percent=True)
    assert 0.0 <= report["score"] <= 100.0


@pytest.mark.slow
def test_actuation_schedule_switches_off(miura_creases):
    engine = OrigamiEngine(backend="jax")
    # Driven for the first millisecond only, then left passive.
    result = engine.run(miura_creases, actuation_schedule=lambda t: t < 0.001, record=True, **RUN)
    traj = result.extra["trajectory"]
    assert "actuated" in traj
    masks = traj["actuated"]
    assert masks.shape[1] == 4
    assert masks[0].all()
    assert not masks[-1].any()
    assert list(result.info["actuated"]) == [False] * 4


@pytest.mark.slow
def test_blend_fitness_components(tiny_creases):
    engine = OrigamiEngine(backend="jax")
    result = engine.run(tiny_creases, **RUN)
    movement_only = compute_forward_displacement(result)
    breakdown = blend_fitness(result, weights=FitnessWeights(movement=1.0, fold=0.0))
    assert breakdown.combined == pytest.approx(movement_only)
    blended = blend_fitness(result, movement_weight=70, fold_weight=30)
    assert blended.movement_fraction == pytest.approx(0.7)
    assert 0.0 <= blended.fold <= 100.0
    assert np.isfinite(blended.combined)


@pytest.mark.slow
def test_engine_evaluate_fold_weight(miura_creases):
    engine = OrigamiEngine(backend="jax")
    plain = engine.evaluate(miura_creases, **RUN)
    blended = engine.evaluate(miura_creases, fold_weight=0.3, **RUN)
    assert isinstance(plain, float) and isinstance(blended, float)
    assert np.isfinite(blended)
    pop = engine.evaluate_population([miura_creases, miura_creases], fold_weight=0.3, **RUN)
    assert pop.scores.shape == (2,)
    assert pop.extra["movement_scores"].shape == (2,)
    assert pop.extra["fold_scores"].shape == (2,)
    assert np.allclose(pop.extra["fold_scores"], pop.extra["fold_scores"][0])


@pytest.mark.slow
def test_evaluator_fold_weight_and_actuation(miura_creases):
    space = DesignSpace.angles_only(miura_creases, name="miura-2x2")
    ev = OrigamiEvaluator(space, backend="jax", fold_weight=0.5, actuated=[True, True, False, False], **RUN)
    assert ev.fold_fraction == pytest.approx(0.5 / 1.5)
    rec = ev.evaluate(space.base_vector())
    assert rec.status == "ok"
    assert "fold_score" in rec.descriptors
    assert 0.0 <= rec.descriptors["fold_score"] <= 100.0

    # fold_weight=0 keeps the historical movement-only objective.
    ev0 = OrigamiEvaluator(space, backend="jax", **RUN)
    assert ev0.fold_weight == 0.0
    assert ev0.movement_fraction == 1.0
