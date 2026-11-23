"""Tests for the configurable design space (Task A) and its integrations (Task C).

Everything here is fast: it exercises the geometry/encoding substrate and the
evaluator's bucketing logic with a stub engine, so no simulation runs.  The one
test that needs JAX is guarded with ``requires_jax``.
"""

import json
import pickle

import numpy as np
import pytest

from origami_engine_mjx.integrations import (
    ANGLE,
    COORD,
    DesignSpace,
    DesignSpaceBuilder,
    OrigamiEvaluator,
    SearchSpace,
    Variable,
    space_spec,
    topology_signature,
)
from origami_engine_mjx.integrations.env import _as_rl_space
from origami_engine_mjx.patterns import load_default_creases

# ----------------------------------------------------------------------
# Default is byte-for-byte the old SearchSpace
# ----------------------------------------------------------------------


def test_angles_only_reproduces_search_space():
    ss = SearchSpace.from_default_pattern()
    ds = DesignSpace.from_default_pattern()
    assert ds.dim == ss.dim == 15
    assert ds.n_creases == ss.n_creases == 24
    assert np.array_equal(ds.lower, ss.lower)
    assert np.array_equal(ds.upper, ss.upper)
    assert ds.parameter_names == ss.parameter_names

    genome = ss.sample(0)
    assert ds.decode(genome) == ss.decode(genome)
    assert np.array_equal(ds.encode(ds.decode(genome)), ss.encode(ss.decode(genome)))


def test_to_search_space_round_trip_is_identical():
    ss = SearchSpace.from_default_pattern()
    back = DesignSpace.from_default_pattern().to_search_space()
    assert back.topology == ss.topology
    assert np.array_equal(back.base_angles, ss.base_angles)
    assert np.array_equal(back.active_indices, ss.active_indices)
    assert np.array_equal(back.lower, ss.lower)
    assert np.array_equal(back.upper, ss.upper)
    assert back.parameter_names == ss.parameter_names
    assert back == ss


def test_from_search_space_round_trip():
    ss = SearchSpace.from_default_pattern()
    assert DesignSpace.from_search_space(ss).to_search_space() == ss


# ----------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------


def test_angles_only_full_includes_loop_edges():
    space = DesignSpace.angles_only(active_only=False)
    assert space.dim == 24
    # Base vector reproduces the template exactly (including inert angles).
    assert space.decode(space.base_vector()) == space.base_creases


def test_loop_angles_selects_only_loop_edges():
    space = DesignSpaceBuilder(load_default_creases()).loop_angles().build()
    # 24 creases, 15 actuated -> 9 inert loop edges.
    assert space.dim == 9
    assert set(space.groups) == {"loop_angles"}


def test_builder_select_by_group_name_index_mask_predicate():
    space = DesignSpace.angles_and_geometry()
    assert space.group_slice("angles") == list(range(15))
    assert len(space.group_slice("geometry")) == 42
    assert space.select(group="angles") == list(range(15))
    assert space.select(name=space.parameter_names[0]) == [0]
    assert space.select(indices=slice(0, 3)) == [0, 1, 2]
    mask = np.zeros(space.dim, dtype=bool)
    mask[[1, 4, 9]] = True
    assert space.select(indices=mask) == [1, 4, 9]
    assert space.select(predicate=lambda v: v.kind == COORD) == space.group_slice("geometry")


def test_geometry_vertex_vs_segment_dims():
    vertex = DesignSpace.geometry_only(geometry_mode="vertex", axes="xy")
    segment = DesignSpace.geometry_only(geometry_mode="segment", axes="xy")
    assert vertex.dim < segment.dim
    assert set(vertex.variable_kinds) == {COORD}
    assert vertex.is_valid(vertex.base_vector())
    assert vertex.decode(vertex.base_vector()) == vertex.base_creases


def test_geometry_single_axis_halves_dim():
    xy = DesignSpace.geometry_only(geometry_mode="vertex", axes="xy")
    x = DesignSpace.geometry_only(geometry_mode="vertex", axes="x")
    assert x.dim * 2 == xy.dim


# ----------------------------------------------------------------------
# Constraints reduce dimension
# ----------------------------------------------------------------------


def test_symmetry_constraint_reduces_dim_and_re_expands():
    base = DesignSpace.from_default_pattern()
    linked = base.with_constraints(symmetries=[(0, 1), (2, 3)], equalities=[(4, 5)])
    assert linked.dim == base.dim - 3
    assert linked.n_fixed == 3
    full = linked._expand(linked.sample(0))
    assert full[1] == pytest.approx(-full[0])
    assert full[3] == pytest.approx(-full[2])
    assert full[5] == pytest.approx(full[4])


def test_constraint_errors():
    base = DesignSpace.from_default_pattern()
    with pytest.raises(ValueError):
        base.with_constraints(equalities=[(0, 0)])
    with pytest.raises(ValueError):
        base.with_constraints(equalities=[(0, 999)])


def test_constrained_space_rejects_rl_conversion():
    base = DesignSpace.from_default_pattern()
    with pytest.raises(ValueError):
        base.with_constraints(symmetries=[(0, 1)]).to_search_space()


def test_dependent_source_must_be_free():
    """A dependent variable may not point at another dependent variable."""
    base = load_default_creases()
    variables = [
        Variable(ANGLE, [(0, "")], lower=-1.0, upper=1.0, name="a"),
        Variable(ANGLE, [(1, "")], lower=-1.0, upper=1.0, name="b"),
        Variable(ANGLE, [(2, "")], lower=-1.0, upper=1.0, name="c"),
    ]
    with pytest.raises(ValueError, match="dependent sources"):
        DesignSpace(base_creases=base, variables=variables, dependent={1: (0, 1.0), 2: (1, 1.0)})
    # with_constraints resolves the chain so only free sources remain
    resolved = DesignSpace(base_creases=base, variables=variables).with_constraints(equalities=[(0, 1), (1, 2)])
    assert resolved.dependent == {1: (0, 1.0), 2: (0, 1.0)}


# ----------------------------------------------------------------------
# Presence / variable crease count
# ----------------------------------------------------------------------


def test_presence_mask_and_list_view():
    space = DesignSpace.everything(presence=[0, 1, 2])
    assert space.has_presence
    assert space.count_mode == "free"
    base = space.base_vector()
    assert space.presence_mask(base).all()
    assert len(space.decode(base)) == space.n_slots

    dropped = base.copy()
    dropped[space.select(group="presence")] = 0.0  # below threshold -> creases absent
    mask = space.presence_mask(dropped)
    assert mask.sum() == space.n_slots - 3
    assert len(space.decode(dropped)) == space.n_slots - 3


def test_count_mode_range_validates_bounds():
    with pytest.raises(ValueError):
        DesignSpace(
            base_creases=load_default_creases(),
            variables=DesignSpace.from_default_pattern().variables,
            count_mode="range",
            count_bounds=None,
        )
    space = DesignSpace.everything(presence=[0, 1], count_mode="range", count_bounds=(10, 24))
    assert space.count_bounds == (10, 24)


def test_presence_count_and_count_ok():
    space = DesignSpace.everything(presence=[0, 1, 2], count_mode="range", count_bounds=(23, 24))
    base = space.base_vector()
    assert space.presence_count(base) == space.n_slots
    assert space.count_ok(base)
    dropped = base.copy()
    dropped[space.select(group="presence")] = 0.0  # drop all three -> count 21
    assert space.presence_count(dropped) == space.n_slots - 3
    assert not space.count_ok(dropped)


def test_rejection_sample_respects_count_bounds():
    # angles + a single presence slot: geometry validity is unaffected by angles,
    # so this isolates the count-bound check in rejection_sample.
    space = (
        DesignSpaceBuilder(load_default_creases())
        .angles(active_only=True)
        .presence([0])
        .build(count_mode="range", count_bounds=(24, 24))
    )
    sample = space.rejection_sample(0, size=3, max_tries=500)
    assert sample.shape == (3, space.dim)
    assert all(space.count_ok(row) and space.presence_count(row) == 24 for row in sample)


# ----------------------------------------------------------------------
# Validity, repair, sampling
# ----------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_geometry_validity_and_repair_never_raise():
    space = DesignSpace.geometry_only(geometry_mode="segment")
    assert space.is_valid(space.base_vector())
    assert space.validity_reason(space.base_vector()) == ""
    assert not space.is_valid(np.full(space.dim, np.nan))  # degenerate geometry
    assert isinstance(space.validity_reason(np.full(space.dim, np.nan)), str)
    repaired = space.repair(np.full(space.dim, np.nan))
    assert np.isfinite(repaired).all()


def test_rejection_sample_returns_valid_genomes():
    space = DesignSpace.angles_and_geometry()
    sample = space.rejection_sample(0, size=2)
    assert sample.shape == (2, space.dim)
    assert all(space.is_valid(row) for row in sample)


def test_rejection_sample_size_zero_is_empty():
    space = DesignSpace.from_default_pattern()
    sample = space.rejection_sample(0, size=0)
    assert sample.shape == (0, space.dim)


def test_normalized_round_trip_for_geometry():
    space = DesignSpace.angles_and_geometry()
    v = space.sample(2)
    z = space.to_normalized(v)
    assert np.all(z >= -1.0) and np.all(z <= 1.0)
    assert np.allclose(space.from_normalized(z), v)


# ----------------------------------------------------------------------
# Introspection, serialization, errors
# ----------------------------------------------------------------------


def test_describe_and_counts():
    space = DesignSpace.angles_and_geometry()
    info = space.describe()
    assert info["dim"] == space.dim == space.n_free
    assert info["n_fixed"] == 0
    assert info["kinds"] == {"angle": 15, "coord": 42}
    assert info["has_presence"] is False


def test_empty_and_conflicting_selections_error():
    with pytest.raises(ValueError):
        DesignSpace(base_creases=load_default_creases(), variables=[])
    with pytest.raises(ValueError):
        DesignSpaceBuilder(load_default_creases()).build()  # no variables


def test_out_of_range_selection_errors():
    with pytest.raises(ValueError):
        DesignSpaceBuilder(load_default_creases()).angles(indices=[0, 999]).build()


def test_dict_and_pickle_round_trip():
    space = DesignSpace.angles_and_geometry()
    restored = DesignSpace.from_dict(json.loads(json.dumps(space.to_dict())))
    assert restored == space
    assert restored.dim == space.dim
    assert pickle.loads(pickle.dumps(space)) == space


def test_variable_bounds_must_increase():
    with pytest.raises(ValueError):
        Variable(ANGLE, [(0, "")], lower=1.0, upper=0.0)
    with pytest.raises(ValueError):
        Variable("bogus", [(0, "")], lower=0.0, upper=1.0)


def test_design_space_rejects_non_angle_rl_conversion():
    with pytest.raises(ValueError):
        DesignSpace.angles_and_geometry().to_search_space()
    with pytest.raises(ValueError):
        DesignSpace.everything(presence=[0]).to_search_space()


# ----------------------------------------------------------------------
# Evaluator bucketing / per-individual path reporting
# ----------------------------------------------------------------------


class _FakeEngine:
    """Records the size of every ``evaluate_population`` call."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def evaluate_population(self, pop, **kwargs):
        self.calls.append(len(pop))

        class _R:
            scores = np.arange(len(pop), dtype=float)

        return _R()


class _StubEvaluator(OrigamiEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eng = _FakeEngine()

    def _engine(self):
        return self.eng


def test_evaluator_fixed_topology_reports_batched():
    space = DesignSpace.from_default_pattern()
    ev = _StubEvaluator(space)
    res = ev.evaluate_batch(np.tile(space.base_vector(), (3, 1)), return_details=False)
    assert res.paths == ["batched", "batched", "batched"]
    assert ev.eng.calls == [3]


def test_evaluator_mixed_topology_buckets_by_signature():
    space = DesignSpace.angles_and_geometry()
    ev = _StubEvaluator(space)
    X = np.tile(space.base_vector(), (3, 1))
    first_coord = space.select(group="geometry")[0]
    X[0, first_coord] = 0.3
    X[1, first_coord] = 0.3
    X[2, first_coord] = 0.7
    res = ev.evaluate_batch(X, return_details=False)
    assert len({topology_signature(space.decode(X[i])) for i in range(3)}) == 2
    assert sorted(res.paths) == ["bucketed", "bucketed", "single"]
    assert sorted(ev.eng.calls) == [1, 2]


class _RaisingSpace:
    """Duck-typed space whose decode fails for a sentinel row."""

    def __init__(self, inner: DesignSpace) -> None:
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def decode(self, row):
        if np.asarray(row)[0] > 100.0:
            raise ValueError("sentinel row")
        return self._inner.decode(row)


def test_evaluator_reports_invalid_path():
    inner = DesignSpace.from_default_pattern()
    ev = _StubEvaluator(_RaisingSpace(inner), error_policy="penalty")
    X = np.tile(inner.base_vector(), (2, 1))
    X[1, 0] = 1000.0
    res = ev.evaluate_batch(X, return_details=False)
    assert "invalid" in res.paths
    assert any(s == "invalid" for s in res.status)


class _AngleSumEngine:
    """Deterministic engine: score = sum of the decoded crease angles."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def evaluate_population(self, pop, **kwargs):
        self.calls.append(len(pop))
        scores = np.asarray([sum(c.angle for c in creases) for creases in pop], dtype=float)

        class _R:
            pass

        r = _R()
        r.scores = scores
        return r


def test_evaluator_mixed_topology_scores_match_one_by_one():
    """A bucketed mixed-topology batch must score each row exactly like a
    single-row evaluation of the same genome."""
    space = DesignSpace.angles_and_geometry()
    rng = np.random.default_rng(3)
    X = np.vstack([space.repair(rng.uniform(space.lower, space.upper)) for _ in range(6)])

    ev = _StubEvaluator(space)
    ev.eng = _AngleSumEngine()
    batch = ev.evaluate_batch(X, return_details=False)

    singles = []
    for i in range(6):
        ev_i = _StubEvaluator(space)
        ev_i.eng = _AngleSumEngine()
        singles.append(float(ev_i.evaluate_batch(X[i : i + 1], return_details=False).raw[0]))

    assert np.allclose(batch.raw, np.asarray(singles))
    assert sum(ev.eng.calls) == sum(1 for p in batch.paths if p != "invalid")


# ----------------------------------------------------------------------
# Adapters / RL coercion (Task C)
# ----------------------------------------------------------------------


def test_space_spec_exposes_dim_bounds_and_presence():
    ss = SearchSpace.from_default_pattern()
    spec = space_spec(ss)
    assert spec["dim"] == 15
    assert spec["presence"] == []
    assert len(spec["lower"]) == 15

    geo = DesignSpace.angles_and_geometry()
    assert space_spec(geo)["dim"] == geo.dim
    pres = DesignSpace.everything(presence=[0, 1, 2])
    assert space_spec(pres)["presence"] == pres.select(group="presence")


def test_as_rl_space_converts_angle_only_design_space():
    ds = DesignSpace.from_default_pattern()
    converted = _as_rl_space(ds)
    assert isinstance(converted, SearchSpace)
    assert converted == SearchSpace.from_default_pattern()
    ss = SearchSpace.from_default_pattern()
    assert _as_rl_space(ss) is ss


def test_as_rl_space_rejects_geometry_design_space():
    with pytest.raises(ValueError):
        _as_rl_space(DesignSpace.angles_and_geometry())
