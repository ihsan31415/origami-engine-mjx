# Architecture

`origami-engine-mjx` keeps all physics-independent logic in shared modules and isolates backend-specific execution in `origami_engine_mjx/backends/`.

## Data flow

```
Crease list / JSON
      │  serialization.normalize_creases()
      ▼
geometry.build_pattern()          # Shapely polygonize → faces → spanning tree → loop edges
      │
      ▼
model.generate_xml(pat, creases)  # → MJCF XML string + info dict (hinges, targets)
      │                              #   (no MuJoCo runtime needed — pure string building)
      ▼
backends.get_backend(name).build_model(xml, info)
      │   jax:  MjModel.from_xml_string → mjx.put_model → mjx.make_data
      │   warp: MjModel.from_xml_string → mjw.make_model → mjw.make_data
      ▼
backend.run(creases, ...)         # simulation loop (table vibration, actuator ramp, mjx.step)
      │
      ▼
SimulationResult(initial_centroid, final_centroid, ...) → fitness = projection on trajectory
```

## Shared vs backend-specific

| Shared (backend-independent) | Backend-specific |
|---|---|
| `constants.py`, `types.py` | `backends/jax_backend.py`, `backends/warp_backend.py` |
| `geometry.py`, `serialization.py`, `patterns.py`, `plotting.py` | model conversion (`put_model` vs `make_model`) |
| `model.py` (`generate_xml` — pure XML) | state init, stepping, actuation, vibration |
| `engine.py` (facade), `evaluation.py` | batch execution strategy |

## Key MJX differences from classic MuJoCo

1. **Immutable state**: MJX `Data` is a JAX pytree. Use `data = data.replace(qpos=data.qpos.at[i].set(v))` instead of `data.qpos[i] = v`.
2. **Step returns new state**: `data = mjx.step(model, data)` instead of `mujoco.mj_step(model, data)` (in-place).
3. **JAX arrays for controls**: control targets must be `jax.numpy` arrays, not plain NumPy (mixing causes `AttributeError: 'numpy.ndarray' has no attribute 'at'` inside MJX).
4. **Name lookups**: use the classic `MjModel` (kept in the model bundle) for `model.joint("table_x").qposadr` etc.; the MJX model has no `.joint()` views.
5. **Rendering**: `mujoco.Renderer` only works with classic `MjModel`/`MjData`. The Warp/JAX backends don't render; use `mjx.get_data` to copy state back to a classic `MjData` for rendering, or keep the legacy `render_individual.py`.
