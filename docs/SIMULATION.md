# Simulation Pipeline

> **Documentation map:** [README](../README.md) · [Getting Started](GETTING_STARTED.md) · [Tutorials](TUTORIALS.md) · [Crease Format](CREASE_FORMAT.md) · **Simulation** · [Evaluation](EVALUATION.md) · [API Reference](API_REFERENCE.md) · [Architecture](ARCHITECTURE.md) · [Troubleshooting](TROUBLESHOOTING.md) · [Glossary](GLOSSARY.md) · [Contributing](CONTRIBUTING.md)

This document explains what happens inside `OrigamiEngine.run(...)`, how the MuJoCo XML model is generated, what the physical constants mean, and how the two MJX backends differ in execution.

## Pipeline overview

```
creases (Crease list | dict list | JSON path)
   │  normalize_creases()                       (serialization.py — coerce + validate)
   ▼
build_pattern(creases) → Pattern               (geometry.py — Shapely face graph + spanning tree)
   │  generate_xml(pat, creases)                (model.py — MJCF XML string + info dict)
   ▼
backend.build_model(xml, info)                  (backends/* — MJX / Warp model conversion)
   │    jax: MjModel.from_xml_string → mjx.put_model → mjx.make_data
   │    warp: MjModel.from_xml_string → mjw.make_model → mjw.make_data
   ▼
backend.run(...) — step loop                    (actuator ramp + table vibration + mjx.step/mjw.step)
   ▼
SimulationResult(model, data, pattern, info, initial_centroid, final_centroid, backend)
```

## Model generation — `generate_xml(pat, creases, name="origami_free", coll_thick=None)`

Returns `(xml: str, info: dict)`. The XML is standard MJCF; you can save it and open it in the MuJoCo viewer. **No MuJoCo runtime is required for generation** — it is pure string building (see [API_REFERENCE.md](API_REFERENCE.md)).

For the reference 4×4 Miura pattern (16 faces / 15 hinges / 9 loop edges), compiling the emitted XML with classic MuJoCo gives `nq=54`, `nv=53`, `nu=15`, `neq=18`, `ngeom=33`, `nbody=18` — and `mjx.put_model` accepts the resulting model on CPU.

Global settings emitted into the XML:

```xml
<compiler angle="radian"/>
<option timestep="0.001" gravity="0 0 -9.81" integrator="implicitfast"/>
<default><geom solref="0.004 1" solimp="0.9 0.95 0.001"/></default>
```

- **Timestep** 1 ms with the `implicitfast` integrator; gravity −9.81 along z.
- Solver defaults `solref="0.004 1"`, `solimp="0.9 0.95 0.001"` for contact softness.

### Bodies and joints — one body per face

- Each face `i` becomes `<body name="face_{i}">` containing one triangular-prism **mesh geom per triangle** (visual `mesh` geoms with density `PAPER_MASS / (S² · 2·THICK) = 1000`, `contype=conaffinity=1`, `condim=3`, friction `"0.3 0.005 0.0001"` — i.e. `MU_PAPER = 0.3` plus tangential/rolling terms). Materials alternate `paper_A` / `paper_B` by tree depth for visualization.
- The **root face** gets a `<freejoint name="root"/>` and is placed at its centroid, offset to `z = Z0 = THICK + 0.001`.
- Every non-root face is attached to its parent through the tree edge:
  - `<joint name="crease_{h}" type="hinge">` — axis along the shared edge, `range="±π"`, damping `DAMPING=0.003`, armature 0.001. This is the fold hinge.
  - Two `<joint name="stretch_a_{h}" ...>` / `<joint name="stretch_n_{h}" ...>` slide joints along the edge direction and its in-plane normal (`stiffness=STRETCH_K=5e4`, `damping=STRETCH_DAMP=50`, `range="±STRETCH_MAX=±1e-4"`) that keep the panel edges from stretching.
- **Loop edges** (edges not in the spanning tree) are closed with `<equality><connect body1 body2 anchor=... solref="0.005 1" solimp="0.95 0.99 0.001"/></equality>` at both endpoints, so the paper stays a single connected sheet.

### Actuators — fold targets

One `<position name="fold_{h}" joint="crease_{h}" kp="2.0" kv="0.15" ctrlrange="±π" forcerange="±2.0"/>` per hinge (`KP`, `KV`, `FORCE_MAX`). The per-hinge target is `clip(sgn * crease.angle, ±MAX_FOLD)` where `sgn` mirrors the angle depending on which side of the hinge the child face lies (`MAX_FOLD = 0.98π`). Targets are returned in `info["targets"]` and the hinge metadata in `info["hinges"]` (`name`, `face`, `edge`, `target`).

`info` dict keys: `hinges`, `targets` (np.ndarray), `n_faces`, `n_hinges`, `n_loops`.

### Support table

```xml
<body name="table" pos="0 0 -0.05">
  <joint name="table_x" type="slide" axis="1 0 0" damping="0"/>
  <joint name="table_z" type="slide" axis="0 0 1" damping="0"/>
  <geom name="ground" type="box" size="5 5 0.05" ... friction="0.5 0.005 0.0001" mass="1000"/>
</body>
```

The table is kinematically vibrated (see below) and the paper rests on it via contacts (`MU_TABLE = 0.5`).

### Optional thin-collision mode

`coll_thick=<float>` splits each mesh into a visual geom (`contype=0`) and a thinner collision geom (transparent, `COLL_THICK = THICK·0.1 = 1e-5` by default) — useful when paper self-collision at full thickness causes jitter.

## The step loop — `OrigamiEngine.run(creases, duration=8.0, fold_duration=3.0, vibrate=True, amp_x=0.0, amp_z=0.0013, freq_hz=75.0, coll_thick=None)`

All backends share the same logical loop:

1. `normalize_creases` → `build_pattern`; raises `ValueError` if `duration <= 0` or the pattern is invalid (`"invalid crease pattern: {reason}"`).
2. `generate_xml` → backend `build_model` → backend `make_state`; all hinge qpos zeroed; `initial_centroid = mean XY of non-table geoms`.
3. For each step `i` in `round(duration / timestep)`:
   - `t = i * timestep`; `frac = min(t / fold_duration, 1.0)`, smoothstep `s = frac²(3 − 2·frac)`; actuator commands = `s * targets` (targets ramp from 0 to full over `fold_duration` seconds).
   - If `vibrate`: table slide joints are set kinematically:
     `qpos = amp·sin(2π f t)`, `qvel = amp·2π f·cos(2π f t)` for `table_x` and `table_z`.
   - `mjx.step(model, data)` (JAX) or `mjw.step(warp_model, data)` (Warp).
4. Stability check: non-finite `qpos` → `FloatingPointError("unstable simulation: non-finite joint coordinates produced.")`.
5. Returns `SimulationResult` with `final_centroid` computed from the paper geoms.

### Backend-specific state handling

- **MJX-JAX** (`backends/jax_backend.py`): state is an **immutable JAX pytree**. Writes use `data.replace(qpos=data.qpos.at[i].set(v))`; controls are `jax.numpy` arrays (`jnp.asarray(info["targets"], dtype=jnp.float32)`). Steps are executed eagerly (`jit=False`) by default — JIT compilation of `mjx.step` on mesh-contact models is prohibitively slow. Passing NumPy arrays into MJX controls causes `AttributeError: 'numpy.ndarray' object has no attribute 'at'`.
- **MJX-Warp** (`backends/warp_backend.py`): state is a Warp struct with `.numpy()` views; writes use `data.qpos.numpy()[i] = v` and `data.ctrl.numpy()[:] = s * targets`. Steps use `mjw.step(warp_model, data)`. Requires a CUDA-capable NVIDIA GPU.

For name lookups (`classic.joint("table_x").qposadr` etc.), both backends keep the classic `MjModel` in their model bundle — the MJX model has no `.joint()` views.

## Physical & solver constants

Defined in `origami_engine_mjx/constants.py` and re-exported at package level:

| Constant | Value | Role |
|---|---|---|
| `S` | `1.0` | Sheet side length (unit square) |
| `THICK` | `1e-4` | Paper thickness (m) |
| `PAPER_MASS` | `0.2` | Total sheet mass (kg) |
| `Z0` | `1.1e-3` | Initial paper height above world origin |
| `MIN_FACE_AREA` | `1e-4` | Reject faces smaller than this (sliver detection) |
| `MAX_FOLD` | `0.98π ≈ 3.079` | Clip for hinge target angles |
| `KP`, `KV` | `2.0`, `0.15` | Position-actuator gains per hinge |
| `DAMPING` | `0.003` | Hinge joint damping |
| `FORCE_MAX` | `2.0` | Actuator force limit |
| `MU_PAPER` / `MU_TABLE` | `0.3` / `0.5` | Friction coefficients |
| `VIB_FREQ` / `VIB_AMP_X` / `VIB_AMP_Z` | `75.0` / `0.0` / `0.0013` | Table vibration defaults |
| `EQ_SOLREF` / `EQ_SOLIMP` | `"0.005 1"` / `"0.95 0.99 0.001"` | Equality-constraint solver params |
| `STRETCH_K` / `STRETCH_DAMP` / `STRETCH_MAX` | `5e4` / `50` / `1e-4` | Panel stretch spring limits |
| `K_PANEL` / `D_PANEL` / `MAX_STRAIN` / `COLL_THICK` | `1e5` / `50` / `0.005` / `1e-5` | Panel stiffness/damping, strain cap, thin collision thickness |
| `M` / `V` | `+π/4` / `−π/4` | Canonical ±45° fold angles (names swapped vs. runtime sign convention — see [CREASE_FORMAT.md](CREASE_FORMAT.md)) |

## Performance notes

- MJX-JAX on CPU is slow for mesh-contact models: ~70 s/step for the 16-face Miura reference model, ~1.4 s/step for a 2-face model (measured locally). Use short durations on CPU; use GPU for real workloads.
- Cost scales with `duration / timestep` (1 s ⇒ 1000 steps) and with the number of faces (each face = one body + meshes).
- JIT compilation of `mjx.step` is expensive on CPU; the JAX backend runs eager by default (`jit=False`).
- `evaluate_population(..., batch_size=N)` delegates to the backend's `run_batch`, which currently loops per individual and ignores `batch_size`; see [EVALUATION.md](EVALUATION.md#current-behaviour-caveats).

## Rendering

Neither MJX backend renders directly — `mujoco.Renderer` works only with classic `MjModel`/`MjData`. For video rendering, install the classic `origami-engine` package (in a separate environment; both packages define the `origami_engine` module) or copy MJX state back with `mjx.get_data` into a classic `MjData`.

## See also

- [EVALUATION.md](EVALUATION.md) — turning a `SimulationResult` into a fitness score
- [API_REFERENCE.md](API_REFERENCE.md) — exact signatures of `OrigamiEngine.run`, `generate_xml`, …
- [ARCHITECTURE.md](ARCHITECTURE.md) — module responsibilities and data flow
- [BACKENDS.md](BACKENDS.md) — MJX-JAX vs MJX-Warp semantics
- [COLAB_GPU.md](COLAB_GPU.md) — GPU validation and benchmarking