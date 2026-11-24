# Glossary

> **Documentation map:** [README](../README.md) · [Getting Started](GETTING_STARTED.md) · [Tutorials](TUTORIALS.md) · [Crease Format](CREASE_FORMAT.md) · [Simulation](SIMULATION.md) · [Evaluation](EVALUATION.md) · [Fitness](FITNESS.md) · [API Reference](API_REFERENCE.md) · [Architecture](ARCHITECTURE.md) · [Troubleshooting](TROUBLESHOOTING.md) · **Glossary** · [Integrations](INTEGRATIONS.md) · [Optimization (ES)](OPTIMIZATION_ES.md) · [Search space](SEARCH_SPACE.md) · [Populations](POPULATIONS.md) · [RL](RL.md) · [Contributing](CONTRIBUTING.md)

Terminology used across the documentation and the code.

| Term | Meaning |
|---|---|
| **Crease** | A line segment on the flat sheet with a signed target fold angle (`Crease(p1, p2, angle)`). Negative angle = mountain fold, positive = valley fold (radians in the Python API). |
| **Mountain / Valley fold** | Fold direction convention: mountain = crease folds "up" (negative angle), valley = folds "down" (positive angle). In `plot_pattern`: mountain = crimson dashed, valley = royalblue dash-dot. |
| **Sheet** | The unit square `[0, S]²` with `S = 1.0` (1 m × 1 m) the paper is cut from. |
| **Face** | A polygon of the crease graph — a rigid panel of the folded sheet. `build_pattern` reconstructs faces by polygonizing the crease arrangement. |
| **Edge** | A shared boundary segment between two faces (`{"i", "j", "p", "q", "crease"}` in `Pattern.edges`). |
| **Tree edge / hinge** | An edge in the spanning tree of the face graph (BFS from the largest-area root face). Tree edges become **hinge joints** in the MuJoCo / MJX model and get a position actuator. |
| **Loop edge** | An edge **not** in the spanning tree (`Pattern.loop_edges`). Closed in the physical model with an equality `<connect>` constraint so the sheet stays connected. The reference pattern has 9 loop edges. |
| **Root face** | The largest-area face (`Pattern.root`); carries the `<freejoint>` and is the base of the kinematic tree. |
| **Pattern** | The validated face/edge graph returned by `build_pattern`: `valid`, `reason`, `faces`, `edges`, `root`, `parent`, `depth`, `tree_edge`, `loop_edges`, `n_faces`, `n_loops`. |
| **Sliver face** | A face smaller than `MIN_FACE_AREA = 1e-4`; causes `Pattern.valid = False` with reason `"sliver face …"`. |
| **MJCF / XML** | MuJoCo's model description format. `generate_xml` emits an MJCF string; `mujoco.MjModel.from_xml_string` compiles it. |
| **`MjModel` / `MjData`** | MuJoCo's compiled model and per-simulation state. `SimulationResult.model` is a backend bundle that keeps `classic_model` (an `MjModel`) for name lookups/rendering; `SimulationResult.data` is the backend-native state (`mjx.Data` or Warp data), not an `MjData`. |
| **`mj_step`** | One physics integration step (timestep 0.001 s, `implicitfast` integrator). MJX-JAX runs `mjx.step`; MJX-Warp runs `mjw.step`. |
| **Position actuator (`fold_{h}`)** | Drives hinge `crease_{h}` toward its target angle with gains `kp=KP=2.0`, `kv=KV=0.15`, force limit `FORCE_MAX=2.0`. Targets ramp in with a smoothstep over `fold_duration`. |
| **Actuated / passive hinge** | A hinge is **actuated** when an actuator drives it toward its target (`actuated=None` = every hinge). A **passive** hinge has no drive and is not braked: it moves only as the rest of the paper pushes it, but it still has a target and is still counted by the fold score. See [FITNESS.md](FITNESS.md). |
| **Stretch joints** | Two slide spring joints per hinge (`stretch_a`, `stretch_n`) that keep panel edges from stretching (`STRETCH_K=5e4`, range ±`STRETCH_MAX=1e-4`). |
| **Equality connect** | MJCF `<connect>` constraint welding the two endpoints of a loop edge between adjacent faces. |
| **Vibrating table** | The support table body (`table_x`/`table_z` slide joints) driven kinematically inside each backend's stepping loop: `qpos = amp·sin(2πft)` with matching `qvel` — defaults `amp_z=0.0013 m`, `freq=75 Hz`. There is no separate `drive_table` helper in this package. |
| **Paper centroid** | Mean XY of all non-table geom positions, computed by each backend's `_centroid_from_data` helper — the tracked point of the sheet. |
| **Forward displacement** | Fitness score: dot product of `final_centroid − initial_centroid` with `trajectory`. A 2-element trajectory is used unnormalized; a 3-element trajectory is normalized. See [EVALUATION.md](EVALUATION.md#forward-displacement). |
| **Trajectory** | Fitness direction: a 2- or 3-element vector (default `(1, 0, 0)` = +x). Waypoint arrays are not supported; population scoring currently ignores the vector and always uses +x. |
| **Individual / population** | Evaluation terminology borrowed from genetic algorithms: one crease pattern = one individual; `evaluate_population` scores a list of them sequentially (no process pool or `workers` argument; `batch_size` is currently ignored). |
| **Miura-ori** | The classic rigid-foldable origami tessellation used as the reference pattern (24 creases, 16 faces, 9 loop edges) — repo copy at `crease_patterns/reference_miura_4x4.json`, packaged copy at `origami_engine_mjx/crease_patterns/reference_miura_4x4.json` (shipped in the wheel, byte-identical). |
| **`solref` / `solimp`** | MuJoCo constraint solver softness parameters (time constant/damping and impedance) tuned in `constants.py` (`EQ_SOLREF`, `EQ_SOLIMP`, geom defaults). |
| **`condim`** | Contact dimensionality (3 = normal + 2 friction directions) used for paper geoms and the table. |
| **MJX** | MuJoCo's JAX backend (`mujoco.mjx`) — consumes the same MJCF XML and enables batched/GPU simulation. Two interchangeable backends in one package: MJX-JAX and MJX-Warp. See [BACKENDS.md](BACKENDS.md). |
| **`origami-render`** | Console script installed by the **legacy** `origami-engine` package (entry point `origami_engine.render_individual:main`). `origami-engine-mjx` ships **no** renderer and no CLI; see [MIGRATION.md](MIGRATION.md#origami-render-and-video-rendering) for the rendering options. |
| **JSON Schema** | `origami_engine_mjx/schemas/crease_pattern.schema.json` (draft 2020-12) validating crease input files; exposed as `origami_engine_mjx.CREASE_SCHEMA`. |
| **Warp backend** | MJX-Warp (`mujoco.warp` on NVIDIA Warp, CUDA GPUs only). See [BACKENDS.md](BACKENDS.md) and [COLAB_GPU.md](COLAB_GPU.md) for status and validation steps. |
| **Search space** | The fixed-topology fold-angle design space searched by an optimizer; `origami_engine_mjx.integrations.SearchSpace`. See [OPTIMIZATION_ES.md](OPTIMIZATION_ES.md). |
| **Design space** | The configurable generalization `origami_engine_mjx.integrations.DesignSpace`: fold angles, loop-edge angles, crease endpoints (geometry) and crease presence, with presets, selection and constraints. The `angles_only` preset reproduces `SearchSpace` exactly. See [SEARCH_SPACE.md](SEARCH_SPACE.md). |
| **Variable** | One optimized quantity in a `DesignSpace` (`Variable`): a fold angle (`ANGLE`), an endpoint coordinate (`COORD`) or a crease's presence (`PRESENCE`). |
| **Presence / crease count** | A `PRESENCE` variable that keeps or drops a crease (value `< presence_threshold` drops it), giving a variable crease count. See [SEARCH_SPACE.md](SEARCH_SPACE.md). |
| **Genome** | One point in the search/design space: the vector of *free* variables (the actuated fold angles by default). Decoded to a full `list[Crease]` by `decode`. |
| **Population seeding** | Producing the *initial* population for an optimizer (seed files, interpolation, random, mixed) — no mutation, crossover or selection. See [POPULATIONS.md](POPULATIONS.md). |
| **Seed** | A known crease pattern (or its genome) used as a starting point or interpolation endpoint when seeding a population. |
| **Descriptor** | A behavior characterization computed from the final simulation state (e.g. displacement direction), used by quality-diversity methods. See `DESCRIPTOR_NAMES`. |
| **Episode** | One RL rollout: `reset()` followed by repeated `step()` until `terminated`/`truncated`. See [RL.md](RL.md). |
| **Evaluator** | `OrigamiEvaluator`, the stateless, picklable black-box objective mapping a genome to a fitness. Ships no algorithm. |

## See also

- [CREASE_FORMAT.md](CREASE_FORMAT.md) — crease/JSON format reference
- [SIMULATION.md](SIMULATION.md) — model & loop internals
- [EVALUATION.md](EVALUATION.md) — fitness functions
- [INTEGRATIONS.md](INTEGRATIONS.md) — ES/RL/AI interfaces and terminology in context
- [API_REFERENCE.md](API_REFERENCE.md) — API signatures
- [BACKENDS.md](BACKENDS.md) — MJX-JAX vs MJX-Warp semantics