"""Tests for population seeding / initialization (Task B).

Fast, NumPy-only tests of the initializers: no optimizer and no simulation.
"""

import numpy as np
import pytest

from origami_engine_mjx import Crease
from origami_engine_mjx.integrations import (
    DesignSpace,
    Population,
    SearchSpace,
    deduplicate,
    diversity_filter,
    export_creases_dir,
    gaussian_population,
    interpolate,
    jitter,
    lhs_population,
    load_population,
    load_seeds,
    mixed_population,
    piecewise_path,
    population_stats,
    random_barycentric,
    random_population,
    save_population,
)
from origami_engine_mjx.serialization import save_creases_to_json


@pytest.fixture
def space():
    return DesignSpace.from_default_pattern()


# ----------------------------------------------------------------------
# Random initialization
# ----------------------------------------------------------------------


def test_random_population_methods_and_shapes(space):
    for method in ("uniform", "gaussian", "lhs", "sobol"):
        X = random_population(space, 8, rng=0, method=method)
        assert X.shape == (8, space.dim)
        assert np.all(X >= space.lower - 1e-9) and np.all(X <= space.upper + 1e-9)


def test_random_population_is_seed_reproducible(space):
    assert np.allclose(random_population(space, 5, rng=7), random_population(space, 5, rng=7))
    assert np.allclose(lhs_population(space, 5, rng=7), lhs_population(space, 5, rng=7))


def test_random_population_rejects_bad_method(space):
    with pytest.raises(ValueError):
        random_population(space, 3, method="bogus")


def test_gaussian_population_centres_on_base(space):
    X = gaussian_population(space, 200, sigma=0.05, rng=0)
    assert np.allclose(X.mean(axis=0), space.base_vector(), atol=0.1)


def test_jitter_clips_to_bounds(space):
    X = np.tile(space.upper, (3, 1))
    jittered = jitter(X, sigma=1.0, space=space, rng=0)
    assert np.all(jittered <= space.upper + 1e-12)


def test_jitter_normalized_requires_space(space):
    with pytest.raises(ValueError):
        jitter(space.sample(0, size=2), sigma=0.1, space=None, rng=0, normalized=True)
    # With a space the normalized path is fine and stays in bounds.
    jittered = jitter(space.sample(0, size=2), sigma=0.1, space=space, rng=0, normalized=True)
    assert np.all(jittered <= space.upper + 1e-12)
    assert np.all(jittered >= space.lower - 1e-12)


def test_search_space_is_accepted(space):
    """The helpers duck-type SearchSpace too."""
    ss = SearchSpace.from_default_pattern()
    X = random_population(ss, 4, rng=0)
    assert X.shape == (4, ss.dim)
    assert gaussian_population(ss, 4, rng=0).shape == (4, ss.dim)


# ----------------------------------------------------------------------
# Interpolation
# ----------------------------------------------------------------------


def test_interpolate_endpoints_and_midpoint(space):
    a = space.sample(0)
    b = space.sample(1)
    assert np.allclose(interpolate(a, b, 0.0), a)
    assert np.allclose(interpolate(a, b, 1.0), b)
    assert np.allclose(interpolate(a, b, 0.5), 0.5 * (a + b))


def test_interpolate_shape_mismatch_raises():
    with pytest.raises(ValueError):
        interpolate([0.0, 0.0], [0.0, 0.0, 0.0], 0.5)


def test_piecewise_and_barycentric(space):
    seeds = random_population(space, 5, rng=2)
    assert piecewise_path(seeds, 10, rng=0).shape == (10, space.dim)
    assert piecewise_path(seeds, 10, rng=0, closed=True).shape == (10, space.dim)
    assert random_barycentric(seeds, 10, rng=0).shape == (10, space.dim)


def test_piecewise_closed_stays_within_seed_hull():
    # A closed path visits the wrap-around segment (last -> first) but must not
    # extrapolate past the final vertex. Regression: the parameter bound used to
    # be ``m`` instead of ``m - 1``, overshooting by one full segment.
    seeds = np.array([[0.0], [1.0], [2.0]])
    out = piecewise_path(seeds, 5000, rng=0, closed=True)
    assert out.min() >= 0.0
    assert out.max() <= 2.0


def test_piecewise_needs_two_seeds(space):
    with pytest.raises(ValueError):
        piecewise_path(space.sample(0)[None, :], 4)


# ----------------------------------------------------------------------
# Dedup / diversity / stats
# ----------------------------------------------------------------------


def test_deduplicate_keeps_first_occurrence():
    X = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]])
    unique, keep = deduplicate(X)
    assert unique.shape == (2, 2)
    assert keep.tolist() == [0, 2]


def test_diversity_filter_selects_far_points(space):
    X = np.vstack([space.base_vector(), space.sample(1), space.sample(2)])
    selected = diversity_filter(X, 2, space=space, rng=0)
    assert selected.shape == (2, space.dim)


def test_diversity_filter_k_ge_n_returns_all(space):
    X = random_population(space, 3, rng=0)
    assert diversity_filter(X, 5, space=space).shape == X.shape


def test_diversity_filter_returns_genome_space_rows(space):
    """Distances use the normalized view, but the returned rows are genomes."""
    X = random_population(space, 6, rng=0)
    selected = diversity_filter(X, 3, space=space, rng=0)
    for row in selected:
        assert any(np.allclose(row, x) for x in X)


def test_population_stats(space):
    X = random_population(space, 8, rng=0)
    stats = population_stats(X, space)
    assert stats["n"] == 8
    assert stats["dim"] == space.dim
    assert stats["valid_fraction"] == pytest.approx(1.0)
    assert stats["mean_pairwise_distance"] > 0.0
    assert len(stats["mean"]) == space.dim


def test_population_stats_empty(space):
    assert population_stats([])["n"] == 0
    empty = population_stats([], space=space)
    assert empty["n"] == 0
    assert empty["valid_fraction"] is None
    # an empty (0, dim) input keeps the declared dimension
    shaped = population_stats(np.zeros((0, space.dim)), space=space)
    assert shaped["n"] == 0 and shaped["dim"] == space.dim


def test_deduplicate_with_space_returns_genome_rows(space):
    X = np.vstack([space.base_vector(), space.base_vector(), space.sample(1)])
    unique, keep = deduplicate(X, space=space)
    assert keep.tolist() == [0, 2]
    assert unique.shape == (2, space.dim)
    assert np.allclose(unique[0], space.base_vector())


# ----------------------------------------------------------------------
# Mixed populations
# ----------------------------------------------------------------------


def test_mixed_population_fractions_and_provenance(space):
    seeds = random_population(space, 5, rng=1)
    pop = mixed_population(space, 30, seeds=seeds, fractions={"seed": 0.2, "interp": 0.5, "random": 0.3}, rng=0)
    assert isinstance(pop, Population)
    assert pop.genomes.shape[1] == space.dim
    assert len(pop.genomes) == len(pop.provenance)
    methods = {p["method"] for p in pop.provenance}
    assert {"seed", "interp", "random"} <= methods


def test_mixed_population_without_seeds_falls_back_to_random(space):
    pop = mixed_population(space, 10, seeds=None, fractions={"seed": 0.5, "interp": 0.5}, rng=0)
    assert all(p["method"] == "random" for p in pop.provenance)


def test_mixed_population_valid_only_filters(space):
    # A geometry space: random genomes are often invalid, so valid_only must keep
    # only valid ones (and never raise).
    geo = DesignSpace.angles_and_geometry()
    pop = mixed_population(geo, 12, fractions={"random": 1.0}, rng=0, valid_only=True)
    assert all(geo.is_valid(row) for row in pop.genomes)


# ----------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------


def test_population_json_and_npz_round_trip(space, tmp_path):
    pop = mixed_population(space, 12, seeds=random_population(space, 4, rng=3), rng=0)
    for suffix in (".json", ".npz"):
        path = save_population(tmp_path / f"pop{suffix}", pop)
        restored = load_population(path)
        assert isinstance(restored.space, DesignSpace)
        assert restored.space == space
        assert np.allclose(restored.genomes, pop.genomes)
        assert restored.provenance == pop.provenance


def test_export_creases_dir_and_load_seeds_round_trip(space, tmp_path):
    pop = mixed_population(space, 6, fractions={"random": 1.0}, rng=0)
    paths = export_creases_dir(tmp_path / "creases", pop, space)
    assert len(paths) == len(pop.genomes)
    seeds = load_seeds(paths, space)
    assert len(seeds) == len(pop.genomes)
    assert seeds.rejected == []
    assert np.allclose(seeds.genomes, pop.genomes)


def test_load_seeds_reports_rejects(space, tmp_path):
    good = tmp_path / "good.json"
    save_creases_to_json(space.decode(space.base_vector()), good)
    bad_topology = tmp_path / "bad_topology.json"
    save_creases_to_json([Crease((0.0, 0.0), (0.1, 0.1), 0.0)], bad_topology)
    garbage = tmp_path / "garbage.json"
    garbage.write_text("not json at all")

    seeds = load_seeds(tmp_path, space)
    assert len(seeds) == 1
    assert str(good) in seeds.sources
    reasons = dict(seeds.rejected)
    assert str(bad_topology) in reasons
    assert str(garbage) in reasons


def test_load_seeds_directory_and_list_forms(space, tmp_path):
    d = tmp_path / "seeds"
    d.mkdir()
    for i in range(3):
        save_creases_to_json(space.decode(space.base_vector()), d / f"seed_{i}.json")
    assert len(load_seeds(d, space)) == 3
    assert len(load_seeds([d], space)) == 3
