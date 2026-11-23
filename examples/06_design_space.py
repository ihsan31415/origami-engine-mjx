"""Configurable search spaces with ``DesignSpace``.

``DesignSpace`` lets a pipeline choose *what* it optimizes: fold angles (the
default), the inert loop-edge angles, crease endpoints (geometry) and crease
presence (a variable crease count).  It also supports selection, equality /
symmetry constraints and a JSON form.  This script ships **no** optimizer — it
only builds the genome spec that the evaluator consumes.

Run:
    python examples/06_design_space.py

The default preset reproduces ``SearchSpace`` exactly; the short simulation at
the end uses a tiny 2-crease pattern for CPU speed.  See ``docs/SEARCH_SPACE.md``.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from origami_engine_mjx import Crease
from origami_engine_mjx.integrations import DesignSpace, OrigamiEvaluator, SearchSpace, topology_signature


def main():
    # 1. The default preset reproduces the fixed-topology SearchSpace exactly.
    ds = DesignSpace.from_default_pattern()
    ss = SearchSpace.from_default_pattern()
    assert ds.dim == ss.dim == 15
    assert ds.to_search_space() == ss
    print(f"default DesignSpace dim={ds.dim} == SearchSpace dim={ss.dim} (identical)")

    # 2. Presets: pick which quantities vary.
    print("\npreset dimensions")
    presets = [
        ("angles_only", DesignSpace.angles_only()),
        ("angles_only(active_only=False)", DesignSpace.angles_only(active_only=False)),
        ("angles_and_geometry", DesignSpace.angles_and_geometry()),
        ("geometry_only(vertex)", DesignSpace.geometry_only()),
        ("geometry_only(segment)", DesignSpace.geometry_only(geometry_mode="segment")),
        ("everything(presence=[0,1,2])", DesignSpace.everything(presence=[0, 1, 2])),
    ]
    for label, space in presets:
        print(f"  {label:32s} dim={space.dim:3d}  kinds={space.describe()['kinds']}")

    # 3. Selection by group / index / slice / mask / predicate.
    combo = DesignSpace.angles_and_geometry()
    mask = np.zeros(combo.dim, dtype=bool)
    mask[[1, 4, 9]] = True
    print("\nselection on angles_and_geometry")
    print(f"  group 'angles' -> {combo.group_slice('angles')[:5]} ...")
    print(f"  group 'geometry' -> {len(combo.select(group='geometry'))} variables")
    print(f"  slice(0, 3) -> {combo.select(indices=slice(0, 3))}")
    print(f"  bool mask -> {combo.select(indices=mask)}")
    print(f"  predicate kind=='coord' -> {combo.select(predicate=lambda v: v.kind == 'coord')[:5]} ...")

    # 4. Equality / symmetry constraints reduce the dimension.
    linked = ds.with_constraints(symmetries=[(0, 1)], equalities=[(4, 5)])
    print(f"\nconstrained dim {linked.dim} (was {ds.dim}), n_fixed={linked.n_fixed}")
    print(f"  a {linked.dim}-d genome still decodes to {len(linked.decode(linked.sample(0)))} creases")

    # 5. Presence gives a variable crease count (padded mask + list view).
    counted = DesignSpace.everything(presence=[0, 1, 2], count_mode="free")
    base = counted.base_vector()
    dropped = base.copy()
    dropped[counted.select(group="presence")] = 0.0  # below the 0.5 threshold
    print(
        f"\npresence: {int(counted.presence_mask(base).sum())}/{counted.n_slots} slots present "
        f"-> {len(counted.decode(base))} creases; dropping 3 -> {len(counted.decode(dropped))}"
    )

    # 6. JSON and pickle round-trip.
    restored = DesignSpace.from_dict(json.loads(json.dumps(counted.to_dict())))
    assert restored == counted
    assert pickle.loads(pickle.dumps(counted)) == counted
    print("\nJSON and pickle round-trip: equal")

    # 7. Validity and repair never crash on a bad genome.
    repaired = ds.repair(np.full(ds.dim, np.nan))
    print(f"\nrepair(NaN genome) -> finite={bool(np.isfinite(repaired).all())}, valid={ds.is_valid(repaired)}")
    print(f"validity_reason(base): {ds.validity_reason(ds.base_vector())!r}")

    # 8. Evaluate a design space; the per-individual scoring path is reported.
    tiny = [
        Crease((0.2, 0.0), (0.5, 0.5), -0.7853981633974483),
        Crease((0.5, 0.5), (0.8, 0.0), -0.7853981633974483),
    ]
    tiny_angles = DesignSpace.angles_only(tiny)
    batch = _evaluator(tiny_angles).evaluate_batch(tiny_angles.sample(0, size=3), return_details=False)
    print("\ntiny angles-only space (one fixed topology)")
    print(f"  fitness={np.round(batch.fitness, 6).tolist()}")
    print(f"  paths  ={batch.paths}")

    # Geometry variables can change the topology between individuals, so the
    # evaluator buckets by topology signature.  Same endpoints -> one bucket;
    # perturbed endpoints -> several buckets.
    tiny_geo = DesignSpace.angles_and_geometry(tiny, geometry_mode="segment")
    same_topo = np.vstack([tiny_geo.base_vector(), tiny_geo.base_vector()])
    same_topo[1, 0] += 0.1  # change an angle only: same endpoints, same model
    sigs = {topology_signature(tiny_geo.decode(row)) for row in same_topo}
    print(f"\ntiny geometry space: {len(sigs)} topology signature(s) for the 2 angle-only rows")
    gbatch = _evaluator(tiny_geo).evaluate_batch(same_topo, return_details=False)
    print(f"  status ={gbatch.status}")
    print(f"  paths  ={gbatch.paths}")


def _evaluator(space):
    return OrigamiEvaluator(space, backend="jax", duration=0.003, fold_duration=0.002, vibrate=False)


if __name__ == "__main__":
    main()
