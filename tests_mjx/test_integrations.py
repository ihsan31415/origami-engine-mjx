"""Compliance tests for the optimization/RL integration interfaces.

These tests exercise the *interfaces* in
:mod:`origami_engine_mjx.integrations` only.  They contain no search algorithm,
matching the package contract.

Simulation-backed tests are marked ``slow`` (MJX-JAX on CPU costs seconds per
tiny step) and are skipped by default; run them with ``pytest -m slow``.

Tests that construct a JAX-backed evaluator (even without stepping the physics)
need the ``[jax]`` extra and are skipped with ``requires_jax`` when it is absent,
so the suite is green on a NumPy-only install (as in CI).
"""

import importlib.util
import json

import numpy as np
import pytest

from origami_engine_mjx import Crease
from origami_engine_mjx.integrations import (
    DESCRIPTOR_NAMES,
    BatchResult,
    EvalRecord,
    OrigamiEvaluator,
    RunLogger,
    SearchSpace,
    benchmark,
    load_best,
    load_genome,
    make_rng,
    save_best,
    save_genome,
    set_global_seed,
)
from origami_engine_mjx.integrations.adapters import (
    to_evosax_fitness,
    to_nevergrad_function,
    to_pymoo_problem,
)


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover - defensive
        return False


#: Skip guard for tests that build a JAX-backed engine (needs the ``[jax]`` extra).
requires_jax = pytest.mark.skipif(
    not (_module_available("jax") and _module_available("mujoco.mjx")),
    reason="requires the [jax] extra (mujoco-mjx + jax)",
)


@pytest.fixture(scope="module")
def tiny_creases():
    """Minimal valid pattern: one mountain fold -> 2 faces, 1 hinge."""
    return [
        Crease((0.2, 0.0), (0.5, 0.5), -0.7853981633974483),
        Crease((0.5, 0.5), (0.8, 0.0), -0.7853981633974483),
    ]


# ----------------------------------------------------------------------
# SearchSpace (fast: pure geometry, no simulation)
# ----------------------------------------------------------------------


def test_default_search_space_shape():
    space = SearchSpace.from_default_pattern()
    assert space.dim == 15
    assert space.n_creases == 24
    assert len(space) == 15
    assert len(space.parameter_names) == 15
    assert (space.lower < space.upper).all()


def test_search_space_active_only_vs_full(tiny_creases):
    active = SearchSpace.from_creases(tiny_creases, active_only=True)
    full = SearchSpace.from_creases(tiny_creases, active_only=False)
    assert active.dim == 1
    assert full.dim == 2
    assert active.active_indices.tolist() == [1]


def test_search_space_encode_decode_round_trip(tiny_creases):
    space = SearchSpace.from_creases(tiny_creases)
    genome = space.sample(0)
    creases = space.decode(genome)
    assert len(creases) == space.n_creases
    assert np.allclose(space.encode(creases), genome)
    # Inactive crease keeps its base angle.
    assert creases[0].angle == pytest.approx(tiny_creases[0].angle)


def test_search_space_normalized_round_trip():
    space = SearchSpace.from_default_pattern()
    genome = space.sample(1)
    z = space.to_normalized(genome)
    assert np.all(z >= -1.0) and np.all(z <= 1.0)
    assert np.allclose(space.from_normalized(z), genome)


def test_search_space_clip_and_repair():
    space = SearchSpace.from_default_pattern()
    clipped = space.clip(np.full(space.dim, 1e9))
    assert np.allclose(clipped, space.upper)
    repaired = space.repair(np.full(space.dim, np.nan))
    assert np.isfinite(repaired).all()


def test_search_space_wrong_length_raises():
    space = SearchSpace.from_default_pattern()
    with pytest.raises(ValueError):
        space.decode(np.zeros(space.dim + 1))


def test_search_space_is_picklable():
    import pickle

    space = SearchSpace.from_default_pattern()
    restored = pickle.loads(pickle.dumps(space))
    assert restored.dim == space.dim
    assert np.allclose(restored.lower, space.lower)


def test_search_space_value_equality():
    a = SearchSpace.from_default_pattern()
    b = SearchSpace.from_default_pattern()
    assert a == b
    assert a != SearchSpace.from_default_pattern(active_only=False)
    assert a != "not a space"


def test_search_space_to_dict_is_json_serializable():
    space = SearchSpace.from_default_pattern()
    text = json.dumps(space.to_dict())
    assert json.loads(text)["dim"] == 15


def test_search_space_invalid_pattern_raises():
    bad = [
        Crease((0.0, 0.0), (0.01, 0.0), 0.0),
        Crease((0.01, 0.0), (0.01, 0.0001), 0.0),
        Crease((0.01, 0.0001), (0.0, 0.0001), 0.0),
    ]
    with pytest.raises(ValueError):
        SearchSpace.from_creases(bad)


# ----------------------------------------------------------------------
# Evaluator (fast paths only; simulation tests are marked slow)
# ----------------------------------------------------------------------


class _ExplodingEvaluator(OrigamiEvaluator):
    """Evaluator whose simulation always fails, to exercise error handling."""

    def _simulate(self, engine, creases):  # type: ignore[override]
        raise RuntimeError("boom")


class _FixedEvaluator(OrigamiEvaluator):
    """Evaluator with a constant raw score, to test sign handling without a sim."""

    def _simulate(self, engine, creases):  # type: ignore[override]
        return 2.0, {"displacement_x": 2.0}


@requires_jax
def test_evaluator_penalty_on_failure(tiny_creases):
    space = SearchSpace.from_creases(tiny_creases)
    ev = _ExplodingEvaluator(space, error_policy="penalty")
    rec = ev.evaluate(np.zeros(space.dim))
    assert rec.status == "error"
    assert rec.fitness == pytest.approx(-1e6)


@requires_jax
def test_evaluator_raise_on_failure(tiny_creases):
    space = SearchSpace.from_creases(tiny_creases)
    ev = _ExplodingEvaluator(space, error_policy="raise")
    with pytest.raises(RuntimeError):
        ev.evaluate(np.zeros(space.dim))


@requires_jax
def test_evaluator_fast_batch_penalty_on_population_failure(tiny_creases):
    """A population-level failure is converted to penalty records, not raised."""
    space = SearchSpace.from_creases(tiny_creases)
    ev = OrigamiEvaluator(space, trajectory=(0.0, 0.0, 0.0), error_policy="penalty")
    res = ev.evaluate_batch(space.sample(0, size=2), return_details=False)
    assert all(s != "ok" for s in res.status)
    assert np.all(res.fitness == pytest.approx(-1e6))


@requires_jax
def test_evaluator_fast_batch_raise_on_population_failure(tiny_creases):
    space = SearchSpace.from_creases(tiny_creases)
    ev = OrigamiEvaluator(space, trajectory=(0.0, 0.0, 0.0), error_policy="raise")
    with pytest.raises(ValueError):
        ev.evaluate_batch(space.sample(0, size=2), return_details=False)


def test_evaluator_bad_error_policy_raises(tiny_creases):
    space = SearchSpace.from_creases(tiny_creases)
    with pytest.raises(ValueError):
        OrigamiEvaluator(space, error_policy="ignore")


def test_evaluator_wrong_genome_length_raises(tiny_creases):
    space = SearchSpace.from_creases(tiny_creases)
    ev = OrigamiEvaluator(space)
    with pytest.raises(ValueError):
        ev.evaluate(np.zeros(space.dim + 3))


def test_evaluator_is_picklable(tiny_creases):
    import pickle

    space = SearchSpace.from_creases(tiny_creases)
    ev = OrigamiEvaluator(space)
    restored = pickle.loads(pickle.dumps(ev))
    assert restored.dim == space.dim


@requires_jax
def test_signed_and_minimizer_helpers(tiny_creases):
    space = SearchSpace.from_creases(tiny_creases)
    ev = _FixedEvaluator(space, maximize=True)
    assert ev.signed(2.0) == 2.0
    assert ev.evaluate(np.zeros(space.dim)).fitness == pytest.approx(2.0)
    assert ev.to_minimizer()(np.zeros(space.dim)) == pytest.approx(-2.0)
    ev_min = _FixedEvaluator(space, maximize=False)
    assert ev_min.signed(2.0) == -2.0
    assert ev_min.evaluate(np.zeros(space.dim)).fitness == pytest.approx(-2.0)


# ----------------------------------------------------------------------
# Adapters (construction only; no simulation)
# ----------------------------------------------------------------------


def test_nevergrad_adapter_is_callable(tiny_creases):
    space = SearchSpace.from_creases(tiny_creases)
    fn = to_nevergrad_function(OrigamiEvaluator(space))
    assert callable(fn)


def test_evosax_adapter_is_callable(tiny_creases):
    space = SearchSpace.from_creases(tiny_creases)
    fn = to_evosax_fitness(OrigamiEvaluator(space))
    assert callable(fn)


def test_pymoo_problem_shape(tiny_creases):
    pytest.importorskip("pymoo")
    space = SearchSpace.from_creases(tiny_creases)
    problem = to_pymoo_problem(OrigamiEvaluator(space))
    assert problem.n_var == space.dim
    assert problem.n_obj == 1
    assert np.allclose(problem.xl, space.lower)


# ----------------------------------------------------------------------
# Plumbing (fast: no simulation)
# ----------------------------------------------------------------------


def test_genome_save_load_round_trip(tiny_creases, tmp_path):
    space = SearchSpace.from_creases(tiny_creases)
    genome = space.sample(3)
    path = save_genome(genome, space, tmp_path / "genome.json")
    assert path.exists()
    assert np.allclose(load_genome(path, space), genome)


def test_best_checkpoint_round_trip(tiny_creases, tmp_path):
    space = SearchSpace.from_creases(tiny_creases)
    genome = space.sample(4)
    path = save_best(genome, space, tmp_path / "best.json", fitness=1.25, extra={"cycle": 2})
    loaded, meta = load_best(path, space)
    assert np.allclose(loaded, genome)
    assert meta["fitness"] == pytest.approx(1.25)
    assert meta["meta"] == {"cycle": 2}


def test_run_logger_jsonl(tmp_path):
    path = tmp_path / "run.jsonl"
    with RunLogger(path) as log:
        log.log(EvalRecord(fitness=1.0, raw=1.0, descriptors={"displacement_x": 0.5}), genome=[0.1, 0.2])
        log.log(EvalRecord(fitness=2.0, raw=2.0, status="ok"), genome=[0.3, 0.4])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["fitness"] == pytest.approx(1.0)
    assert rows[0]["genome"] == [0.1, 0.2]


def test_run_logger_csv(tmp_path):
    path = tmp_path / "run.csv"
    with RunLogger(path) as log:
        log.log(EvalRecord(fitness=1.0, raw=1.0), genome=[0.1, 0.2])
    text = path.read_text()
    assert "fitness" in text.splitlines()[0]
    assert "genome" in text.splitlines()[0]


def test_run_logger_csv_infers_descriptor_columns(tmp_path):
    path = tmp_path / "run.csv"
    with RunLogger(path) as log:
        log.log(EvalRecord(fitness=1.0, raw=1.0, descriptors={"displacement_x": 0.5}), genome=[0.1, 0.2])
    lines = path.read_text().splitlines()
    assert "displacement_x" in lines[0]
    assert "0.5" in lines[1]


def test_run_logger_rejects_bad_format(tmp_path):
    with pytest.raises(ValueError):
        RunLogger(tmp_path / "run.txt", format="xml")


def test_seed_and_rng_are_reproducible():
    set_global_seed(123)
    a = make_rng(7).uniform(size=5)
    b = make_rng(7).uniform(size=5)
    assert np.allclose(a, b)
    space = SearchSpace.from_default_pattern()
    assert np.allclose(space.sample(11), space.sample(11))


# ----------------------------------------------------------------------
# Simulation-backed tests (slow on CPU)
# ----------------------------------------------------------------------


@pytest.mark.slow
def test_evaluator_single_and_batch(tiny_creases):
    space = SearchSpace.from_creases(tiny_creases)
    ev = OrigamiEvaluator(
        space, backend="jax", duration=0.003, fold_duration=0.002, vibrate=False, return_descriptors=True
    )
    rec = ev.evaluate(np.array([0.3]))
    assert rec.status == "ok"
    assert np.isfinite(rec.raw)
    assert set(rec.descriptors) == set(DESCRIPTOR_NAMES)

    X = np.array([[0.0], [0.3], [-0.3]])
    result = ev.evaluate_batch(X)
    assert isinstance(result, BatchResult)
    assert result.n == 3
    assert result.ok.all()
    assert result.fitness.shape == (3,)


@pytest.mark.slow
def test_evaluator_cache_returns_cached(tiny_creases):
    space = SearchSpace.from_creases(tiny_creases)
    ev = OrigamiEvaluator(space, backend="jax", duration=0.003, fold_duration=0.002, vibrate=False, use_cache=True)
    first = ev.evaluate(np.array([0.3]))
    second = ev.evaluate(np.array([0.3]))
    assert not first.cached
    assert second.cached
    assert second.fitness == pytest.approx(first.fitness)


@pytest.mark.slow
def test_benchmark_returns_metrics(tiny_creases):
    space = SearchSpace.from_creases(tiny_creases)
    ev = OrigamiEvaluator(space, backend="jax", duration=0.003, fold_duration=0.002, vibrate=False)
    metrics = benchmark(ev, n=2)
    assert metrics["n"] == 2
    assert metrics["wall_time"] >= 0.0
    assert "evals_per_sec" in metrics


def test_benchmark_rejects_nonpositive_n(tiny_creases):
    """A non-positive count is rejected before any sampling or simulation."""
    space = SearchSpace.from_creases(tiny_creases)
    ev = OrigamiEvaluator(space, backend="jax")
    with pytest.raises(ValueError):
        benchmark(ev, n=0)


@requires_jax
def test_benchmark_warmup_not_counted_as_cache_hit(tiny_creases):
    """The warm-up genome is kept out of the timed set even with caching on."""
    space = SearchSpace.from_creases(tiny_creases)
    ev = _FixedEvaluator(space, use_cache=True)
    benchmark(ev, n=2)
    assert ev.n_cache_hits == 0


@requires_jax
def test_batch_result_reports_cached_and_errors(tiny_creases):
    space = SearchSpace.from_creases(tiny_creases)
    ev = _FixedEvaluator(space, use_cache=True)
    X = space.sample(0, size=2)
    ev.evaluate_batch(X, return_details=True)  # populate the cache
    res = ev.evaluate_batch(X, return_details=True)  # all cache hits
    assert res.cached.all()
    assert all(err == "" for err in res.errors)
    assert all(rec.cached for rec in res.as_records())
