# Simulation Pipeline

> **Documentation map:** [README](../README.md) · [Getting Started](GETTING_STARTED.md) · [Tutorials](TUTORIALS.md) · [Crease Format](CREASE_FORMAT.md) · **Simulation** · [Evaluation](EVALUATION.md) · [Fitness](FITNESS.md) · [API Reference](API_REFERENCE.md) · [Architecture](ARCHITECTURE.md) · [Troubleshooting](TROUBLESHOOTING.md) · [Glossary](GLOSSARY.md) · [Integrations](INTEGRATIONS.md) · [Optimization (ES)](OPTIMIZATION_ES.md) · [Search space](SEARCH_SPACE.md) · [Populations](POPULATIONS.md) · [RL](RL.md) · [Contributing](CONTRIBUTING.md)

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

## Model generation — `generate_xml(pat, creases, name="origami_free", coll_thick=None, actuated=None)`

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

One `<position name="fold_{h}" joint="crease_{h}" kp="2.0" kv="0.15" ctrlrange="±π" forcerange="±2.0"/>` per **actuated** hinge (`KP`, `KV`, `FORCE_MAX`). The per-hinge target is `clip(sgn * crease.angle, ±MAX_FOLD)` where `sgn` mirrors the angle depending on which side of the hinge the child face lies (`MAX_FOLD = 0.98π`). Targets are returned in `info["targets"]` and the hinge metadata in `info["hinges"]` (`name`, `face`, `edge`, `target`).

Every hinge of the pattern also exists as a hinge joint, so a pattern can have
passive hinges that nothing drives:

```python
xml, info = generate_xml(pat, creases, actuated=[True, False])  # drive hinge 0 only
```

`actuated=None` (default) drives all hinges, `True`/`False` drives all or none,
and a list of bools (one per hinge) picks individual hinges. The mask is
returned as `info["actuated"]` and `info["actuator_hinges"]` lists, in order,
the hinge index behind each actuator (element *i* of `ctrl` drives
`hinges[info["actuator_hinges"][i]]`). A list of the wrong length raises
`ValueError("actuated has N entries, expected one per hinge (M).")`.

`info` dict keys: `hinges`, `targets` (np.ndarray), `n_faces`, `n_hinges`, `n_loops`, `actuated`, `actuator_hinges`.

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

## The step loop — `OrigamiEngine.run(creases, duration=8.0, fold_duration=3.0, vibrate=True, amp_x=0.0, amp_z=0.0013, freq_hz=75.0, coll_thick=None, record=False, record_every=1, actuated=None, actuation_schedule=None)`

All backends share the same logical loop:

1. `normalize_creases` → `build_pattern`; raises `ValueError` if `duration <= 0` or the pattern is invalid (`"invalid crease pattern: {reason}"`).
2. `generate_xml` → backend `build_model` → backend `make_state`; all hinge qpos zeroed; `initial_centroid = mean XY of non-table geoms`.
3. For each step `i` in `round(duration / timestep)`:
   - `t = i * timestep`; `frac = min(t / fold_duration, 1.0)`, smoothstep `s = frac²(3 − 2·frac)`; actuator commands = `s * targets` (targets ramp from 0 to full over `fold_duration` seconds).
   - If `actuation_schedule` is given, `mask = actuation_schedule(t)` is evaluated **every step**; otherwise `mask` is the fixed `actuated` mask. An actuator whose hinge is off (or not actuated at all) is commanded to the hinge's **current angle**, which produces no servo torque, so that hinge is free — it is not braked towards 0.
   - If `vibrate`: table slide joints are set kinematically:
     `qpos = amp·sin(2π f t)`, `qvel = amp·2π f·cos(2π f t)` for `table_x` and `table_z`.
   - `mjx.step(model, data)` (JAX) or `mjw.step(warp_model, data)` (Warp).
4. Stability check: non-finite `qpos` → `FloatingPointError("unstable simulation: non-finite joint coordinates produced.")`.
5. Returns `SimulationResult` with `final_centroid` computed from the paper geoms.

### Recording a trajectory — `record=True`

Passing `record=True` (and optionally `record_every=N` steps) stores every
sampled frame in `result.extra["trajectory"]`:

| key | shape | meaning |
|---|---|---|
| `time` | `(T,)` | frame time in seconds (first frame `0.0`, last frame `duration`) |
| `qpos`, `qvel` | `(T, nq)` / `(T, nv)` | full state, same layout as the backend's `data` |
| `centroid` | `(T, 2)` | mean XY of the paper geoms — the scored quantity |
| `hinge_qposadr` | `(n_hinges,)` | fold-hinge `qpos` columns (excludes table/stretch/panel joints) |
| `actuated` | `(T, n_hinges)` | only with an `actuation_schedule`: which hinges were driven at each frame |

Recording does **not** change the physics: the final `qpos`, `centroid` and the
score are identical with and without it (tested). It costs one host copy per
sampled frame, so keep `record_every` ≥ 1 and thin long runs. To get a score
and a trajectory from a single simulation:

```python
score, trajectory = engine.evaluate(creases, duration=1.0, return_trajectory=True)
```

`result.info["hinge_qpos"]` holds the achieved hinge angles (in `info["hinges"]`
order) after the run, ready for
[`fold_score`](FITNESS.md) without touching `qpos` layouts.

`result.info["table_qposadr"]` holds the `qpos` indices of the `table_x` /
`table_z` slides, so
[`Trajectory.center_of_mass()`](API_REFERENCE.md#origamienginemjxrendering-video-cameras-plots)
can remove the table's own motion from the measured movement.

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

MJX states cannot be drawn directly — `mujoco.Renderer` only works with classic
`MjModel`/`MjData`. Both backends therefore keep the classic model in their
bundle (`result.model["classic_model"]`) and record the joint trajectory; the
`origami_engine_mjx.rendering` package replays each recorded frame into a
classic `MjData` (`mj_forward` + `update_scene`) and composites the result.

```python
from origami_engine_mjx.rendering import render_video

result = engine.run(creases, duration=1.0, record=True)

render_video(result, "default.mp4")  # MuJoCo default view
render_video(result, "top.mp4", camera="top")  # named preset
render_video(result, "manual.mp4", camera=(40, -20, 1.3))  # az, el, distance (deg/m)
render_video(result, "four.mp4", views="all_sides")  # 4 views, one video
render_video(result, "graphs.mp4", plot="xyz")  # x/y/z movement beside
render_video(result, "run7.mp4", prefix="run7", camera="run7_top", plot=["x", "y"])
```

Replays ignore actuation: the recorded `qpos`/`qvel` are the real physics, so a
video shows exactly which hinges were driven (the per-frame `"actuated"` mask is
available if you want to overlay that information yourself).

Rendering needs an offscreen GL context (`MUJOCO_GL=egl` or `osmesa`) and
`ffmpeg` for MP4 (a `.gif` path works without ffmpeg). Without ffmpeg a
`RuntimeWarning` is raised and frames are kept in memory instead of failing.

### Cameras

A camera is MuJoCo's free-camera spherical form (`lookat`, `distance`,
`azimuth`, `elevation`; angles in **degrees**). Three ways to pick one:

* **Preset name** — `top`, `side`, `side_left`, `top_side_left`, `iso_top`, …
  (26 presets; see `CAMERA_PRESETS`). `default` reproduces MuJoCo's own free
  camera, so it matches the old `OrigamiEnv.render` view.
* **Manual coordinates** — a tuple `(azimuth, elevation[, distance[, zoom]])`
  or a mapping `{"azimuth": 40, "elevation": -20, "distance": 1.4,
  "zoom": 1.0, "lookat": (0.5, 0.5, 0.0)}`. `zoom < 1` moves closer.
* **Prefix** — `prefix="run7"` namespaces preset names, so `"run7_top"` (or
  bare `"run7"` = `"run7_default"`) resolves without a global preset. The
  resolved cameras are saved next to the video as `<stem>.camera.json` (and
  `save_camera` / `load_camera` round-trip one explicitly).

The default look-at point is the centre of the pattern's bounding box. See the
preset reference grid in [TUTORIALS.md](TUTORIALS.md).

### Multi-view videos

`views=` / `cameras=` take a list of views and concatenate them into **one**
video, left to right, each letterboxed to `resolution` so nothing is distorted.
`views="all_sides"` expands to four look-down views (`iso_top`,
`iso_top_side`, `top`, `iso_top_left`) — one 4-view video instead of four
separate ones. With `plot=` the camera row and the graph column are placed side
by side (output width grows by 50 %).

### Movement graphs beside the video

`plot=` adds matplotlib panels to every frame, one per axis:

| `plot` | panels |
|---|---|
| `"x"` / `"y"` | centre-of-mass movement along that axis (relative to frame 0) |
| `"z"` | mean absolute fold-hinge angle, i.e. how far the folds have closed |
| `"xyz"` | all three, stacked |
| `["x", "y"]` | those panels, left to right |

The curve is drawn with a progress marker, so it grows with the video.
The same plots are available standalone as
`plot_trajectory(result, path="movement.png")` (no GL needed), and each axis'
series through `Trajectory.center_of_mass()`, `Trajectory.velocity()` and
`Trajectory.hinge_angles()`.

### The frequency/amplitude overlay

Every rendered frame carries a small log in its **bottom-left corner**
(`overlay="freq/amp"`, the default):

```
freq=75.0 Hz
amp x=0.0000 m
amp z=0.0013 m
```

Values come from the `excitation={"freq_hz", "amp_x", "amp_z", "vibrate"}`
mapping (defaults: the `VIB_*` constants). `vibrate=False` prints
`vibration off`. Pass `overlay="gen {index} | {freq_hz:.0f} Hz"` for a custom
format (fields: `t`, `frac`, `index`, `freq_hz`, `amp`), a callable
`(t, frac) -> str`, `overlay_extra="best gen 7"` to append a line, or
`overlay=None` to disable it.

### `render_video` reference

```python
render_video(result, path, *, camera=None, cameras=None, views=None, prefix=None,
             plot=None, overlay="freq/amp", resolution=(480, 360), fps=25,
             max_frames=None, lookat=None, zoom=1.0, excitation=None,
             overlay_extra=None, save_camera_json=True) -> Path
```

* `resolution=(width, height)` is **one camera tile**; the output width is
  `n_views × width` (plus a graph column when `plot=` is set).
* `max_frames=N` thins the trajectory (first and last frame kept) so short
  runs still give a sensible video length; `fps` is the playback rate. A run
  rendered with `max_frames=len(trajectory)` plays back in real time.
* `render_frames(...)` returns the frames without encoding (notebooks, tests).
* The classic `OrigamiEnv.render` contract is unchanged.

### Plotting the generation history

`origami_engine_mjx.rendering.plot_history(history)` draws best/mean per
generation (plus the running global best) from a `GenerationLogger`'s output —
see [EVALUATION.md](EVALUATION.md#logging-generations-generationlogger).

## See also

- [EVALUATION.md](EVALUATION.md) — turning a `SimulationResult` into a fitness score, run and generation logging
- [TUTORIALS.md](TUTORIALS.md) — camera presets, 4-view videos, graphs and overlays
- [API_REFERENCE.md](API_REFERENCE.md) — exact signatures of `OrigamiEngine.run`, `generate_xml`, …
- [ARCHITECTURE.md](ARCHITECTURE.md) — module responsibilities and data flow
- [BACKENDS.md](BACKENDS.md) — MJX-JAX vs MJX-Warp semantics
- [COLAB_GPU.md](COLAB_GPU.md) — GPU validation and benchmarking