"""Population seeding & initialization.

Builds the *initial* population an external optimizer starts from: random
sampling, seed files, interpolation, mixtures, de-duplication, diversity
selection, I/O and statistics.  This library ships **no** mutation, crossover or
selection algorithm — only seeding.

Run:
    python examples/07_populations.py

See ``docs/POPULATIONS.md``.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from origami_engine_mjx import save_creases_to_json
from origami_engine_mjx.integrations import (
    DesignSpace,
    deduplicate,
    diversity_filter,
    export_creases_dir,
    interpolate,
    jitter,
    load_population,
    load_seeds,
    mixed_population,
    piecewise_path,
    population_stats,
    random_barycentric,
    random_population,
    save_population,
)


def main():
    space = DesignSpace.from_default_pattern()  # the 15-d default genome
    print(f"space: dim={space.dim}, n_slots={space.n_slots} ({space.name})")

    # 1. Random initialization (uniform / Gaussian / Latin hypercube / Sobol').
    print("\nrandom initialization")
    for method in ("uniform", "gaussian", "lhs", "sobol"):
        pop = random_population(space, 8, rng=0, method=method)
        print(f"  {method:8s} -> {pop.shape}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # 2. Write a few seed files, then load them back into the space.
        seed_dir = tmp / "seeds"
        seed_dir.mkdir()
        rng = np.random.default_rng(0)
        base = space.base_vector()
        for i in range(3):
            genome = space.repair(base + rng.normal(0.0, 0.15, space.dim))
            save_creases_to_json(space.decode(genome), seed_dir / f"seed_{i}.json")
        seeds = load_seeds(seed_dir, space)
        print(f"\nloaded {len(seeds)} seeds ({len(seeds.rejected)} rejected)")

        # A malformed file is reported, not fatal.
        (seed_dir / "bad.json").write_text("{ this is not json")
        seeds = load_seeds(seed_dir, space)
        print(f"after adding a bad file: {len(seeds)} seeds, {len(seeds.rejected)} rejected")
        for path, reason in seeds.rejected:
            print(f"  rejected {Path(path).name}: {reason.split(':')[0]}")

        # 3. Interpolation between seeds.
        a, b = seeds.genomes[0], seeds.genomes[1]
        print(f"\ninterpolate midpoint -> {np.round(interpolate(a, b, 0.5)[:3], 4).tolist()} ...")
        print(f"piecewise_path(seeds, 8) -> {piecewise_path(seeds.genomes, 8, rng=0).shape}")
        print(f"random_barycentric(seeds, 8) -> {random_barycentric(seeds.genomes, 8, rng=0).shape}")
        print(f"jitter(..., sigma=0.05) -> {jitter(seeds.genomes, sigma=0.05, space=space, rng=0).shape}")

        # 4. Mixed population with provenance (20% seeds / 50% interp / 30% random).
        pop = mixed_population(
            space,
            24,
            seeds=seeds.genomes,
            fractions={"seed": 0.2, "interp": 0.5, "random": 0.3},
            rng=0,
        )
        counts: dict[str, int] = {}
        for entry in pop.provenance:
            counts[entry["method"]] = counts.get(entry["method"], 0) + 1
        print(f"\nmixed population -> {pop.genomes.shape}, provenance={counts}")

        # 5. De-duplication and diversity-aware selection.
        unique, _ = deduplicate(pop.genomes)
        print(f"deduplicate -> {unique.shape} unique of {pop.genomes.shape[0]}")
        print(f"diversity_filter(8) -> {diversity_filter(pop.genomes, 8, space=space, rng=0).shape}")

        # 6. Population I/O and crease export.
        save_population(tmp / "pop.json", pop, space, meta={"name": "demo"})
        restored = load_population(tmp / "pop.json")
        print(f"\nsave/load JSON -> {restored.genomes.shape}, space restored={restored.space is not None}")
        save_population(tmp / "pop.npz", pop, space)
        print(f"save/load npz  -> {load_population(tmp / 'pop.npz').genomes.shape}")
        print(f"export_creases_dir -> {len(export_creases_dir(tmp / 'creases', pop, space))} files")

        # 7. Statistics.
        stats = population_stats(pop.genomes, space)
        print(
            f"\npopulation_stats: n={stats['n']} dim={stats['dim']} "
            f"valid_fraction={stats['valid_fraction']:.3f} "
            f"mean_pairwise_distance={stats['mean_pairwise_distance']:.3f}"
        )


if __name__ == "__main__":
    main()
