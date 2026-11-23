# Glossary

> **Documentation map:** [README](../README.md) · [Getting Started](GETTING_STARTED.md) · [Tutorials](TUTORIALS.md) · [Crease Format](CREASE_FORMAT.md) · [Simulation](SIMULATION.md) · [Evaluation](EVALUATION.md) · [API Reference](API_REFERENCE.md) · [Architecture](ARCHITECTURE.md) · [Troubleshooting](TROUBLESHOOTING.md) · **Glossary** · [Contributing](CONTRIBUTING.md)

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
| **`MjModel` / `MjData`** | MuJoCo's compiled model and per-simulation state. Returned inside `SimulationResult` (as backend-specific handles). |
| **`mj_step`** | One physics integration step (timestep 0.001 s, `implicitfast` integrator). MJX-JAX runs `mjx.step`; MJX-Warp runs `mjw.step`. |
| **Position actuator (`fold_{h}`)** | Drives hinge `crease_{h}` toward its target angle with gains `kp=KP=2.0`, `kv=KV=0.15`, force limit `FORCE_MAX=2.0`. Targets ramp in with a smoothstep over `fold_duration`. |
| **Stretch joints** | Two slide spring joints per hinge (`stretch_a`, `stretch_n`) that keep panel edges from stretching (`STRETCH_K=5e4`, range ±`STRETCH_MAX=1e-4`). |
| **Equality connect** | MJCF `<connect>` constraint welding the two endpoints of a loop edge between adjacent faces. |
| **Vibrating table** | The support table body (`table_x`/`table_z` slide joints) driven kinematically by `drive_table`: `qpos = amp·sin(2πft)` — defaults `amp_z=0.0013 m`, `freq=75 Hz`. |
| **Paper centroid** | Mean XY of all non-table geom positions (`paper_centroid`) — the tracked point of the sheet. |
| **Forward displacement** | Fitness score: `dot(final_centroid − initial_centroid, normalize(trajectory))` — how far the sheet moved along the requested direction while folding. |
| **Trajectory** | Fitness direction: a 2/3-vector, or an (N,2)/(N,3) waypoint array (direction = last − first), normalized. Default `(1, 0, 0)` = +x. |
| **Individual / population** | Evaluation terminology borrowed from genetic algorithms: one crease pattern = one individual; `evaluate_population` scores a list of them (optionally in a process pool via `evaluate_population(..., backend=..., batch_size=...)`). |
| **Miura-ori** | The classic rigid-foldable origami tessellation used as the reference pattern (24 creases, 16 faces, 9 loop edges) — repo copy at `crease_patterns/reference_miura_4x4.json`, packaged copy at `origami_engine_mjx/crease_patterns/reference_miura_4x4.json` (shipped in the wheel, byte-identical). |
| **`solref` / `solimp`** | MuJoCo constraint solver softness parameters (time constant/damping and impedance) tuned in `constants.py` (`EQ_SOLREF`, `EQ_SOLIMP`, geom defaults). |
| **`condim`** | Contact dimensionality (3 = normal + 2 friction directions) used for paper geoms and the table. |
| **MJX** | MuJoCo's JAX backend (`mujoco.mjx`) — consumes the same MJCF XML and enables batched/GPU simulation. Two interchangeable backends in one package: MJX-JAX and MJX-Warp. See [BACKENDS.md](BACKENDS.md). |
| **`origami-render`** | Console script installed by the **legacy** `origami-engine` package (entry point `origami_engine.render_individual:main`). The `origami-engine-mjx` package does **not** ship a renderer; use classic MuJoCo or `trace_simulation` + `export_video` directly. |
| **JSON Schema** | `origami_engine_mjx/schemas/crease_pattern.schema.json` (draft 2020-12) validating crease input files; exposed as `origami_engine_mjx.CREASE_SCHEMA`. |
| **Warp backend** | MJX-Warp (`mujoco.warp` on NVIDIA Warp, CUDA GPUs only). See [BACKENDS.md](BACKENDS.md) and [COLAB_GPU.md](COLAB_GPU.md) for status and validation steps. |

## See also

- [CREASE_FORMAT.md](CREASE_FORMAT.md) — crease/JSON format reference
- [SIMULATION.md](SIMULATION.md) — model & loop internals
- [EVALUATION.md](EVALUATION.md) — fitness functions
- [API_REFERENCE.md](API_REFERENCE.md) — API signatures
- [BACKENDS.md](BACKENDS.md) — MJX-JAX vs MJX-Warp semantics