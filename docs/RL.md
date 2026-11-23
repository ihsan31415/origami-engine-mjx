# Reinforcement Learning environments

> **Documentation map:** [README](../README.md) · [Docs index](README.md) · [Getting Started](GETTING_STARTED.md) · [Tutorials](TUTORIALS.md) · [Crease Format](CREASE_FORMAT.md) · [Simulation](SIMULATION.md) · [Evaluation](EVALUATION.md) · [API Reference](API_REFERENCE.md) · [Architecture](ARCHITECTURE.md) · [Backends](BACKENDS.md) · [Integrations](INTEGRATIONS.md) · [Optimization (ES)](OPTIMIZATION_ES.md) · **RL** · [Troubleshooting](TROUBLESHOOTING.md) · [Glossary](GLOSSARY.md) · [Contributing](CONTRIBUTING.md)

This guide covers the reinforcement-learning environments in
`origami_engine_mjx.integrations`. They expose the folding simulation as an
environment for any RL library; they implement **no** learning algorithm.

There are two ways to drive the simulation:

1. **Gymnasium** — `OrigamiEnv` (single) and `OrigamiVectorEnv` (batched),
   with `reset`/`step` and `Box` spaces. Hand `OrigamiEnv` to a
   Gymnasium-compatible trainer.
2. **Pure JAX** — `reset` / `step` in `integrations/jax_env.py`, which are
   pytree-in / pytree-out and work under `jax.jit` / `jax.vmap` /
   `jax.lax.scan`.

> Requires the `[rl]` extra (`gymnasium`) and the `[jax]` extra
> (`mujoco-mjx`, `jax`). The environments use the **MJX-JAX backend only**;
> the Warp backend is stateful and is not wrapped here.

## Gymnasium: `OrigamiEnv`

```python
import numpy as np
from origami_engine_mjx import Crease
from origami_engine_mjx.integrations import SearchSpace
from origami_engine_mjx.integrations.env import OrigamiEnv

creases = [
    Crease((0.2, 0.0), (0.5, 0.5), -0.7853981633974483),
    Crease((0.5, 0.5), (0.8, 0.0), -0.7853981633974483),
]
space = SearchSpace.from_creases(creases)

env = OrigamiEnv(space=space, max_steps=100, frame_skip=1, vibrate=False)
obs, info = env.reset(seed=0)

for _ in range(env.max_steps):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

### Spaces

| Space | Type | Shape | Meaning |
|---|---|---|---|
| Observation | `Box(float32)` | `(dim + 3,)` | `[normalized_target_angles (dim), centroid_x, centroid_y, progress]` |
| Action | `Box(float32)` | `(dim,)` | Fold-angle targets (or deltas); see below |

`dim = space.dim` is the number of actuated hinges (15 for the default pattern).

Action entry `j` (and observation angle `j`) corresponds to the same crease as
genome entry `j`, i.e. `space.active_indices[j]`. The MJCF model emits its
actuators in a depth-first traversal, which is generally **not** sorted crease
order, so the environments reindex the action into the model's actuator order
internally (`EnvParams.ctrl_perm`) and apply the per-hinge fold sign that
`generate_xml` folds into the joint target (`EnvParams.ctrl_sign`), matching
`OrigamiEngine`'s direction. You can therefore pass an environment action
straight to `space.decode` / `space.encode` and get the crease pattern you
expect. (A pattern where one crease maps to several spanning-tree hinges is not
supported: the space deduplicates creases while the model does not, so
`make_params` rejects the mismatch.)

### Action modes

- `normalized_action=True` (default): actions are in `[-1, 1]` and map linearly
  onto `[lower, upper]`.
- `normalized_action=False`: actions are already fold angles.
- `action_mode="absolute"` (default): the action sets the target angles.
- `action_mode="delta"`: the action is added to the current targets
  (scaled by `max_delta` when normalized).

The resulting targets are always clipped to the search-space bounds.

### Reward

By default the reward is the **change in forward (+x) centroid displacement**
produced by the step, so the episode return equals the total forward
displacement. Provide `reward_fn(prev_state, action, state, info) -> float` to
shape it. See [EVALUATION.md](EVALUATION.md) for the displacement signal.

### Termination

- `terminated` is `True` only if the simulation becomes non-finite
  (`info["status"] == "unstable"`).
- `truncated` is `True` once `step_count >= max_steps`.
- On a normal step `info` carries `step_count`, `status`,
  `forward_displacement`, `centroid` and `targets`. The terminal unstable step
  omits `centroid` and `targets`, and `reset` omits `targets`.

### Seeding and determinism

The physics is deterministic. `reset(seed=...)` seeds the per-episode NumPy
generator, which is only used when you pass
`options={"randomize_start": True}` (it then randomizes the initial target
angles). Identical actions give identical rollouts.

### `check_env`

`OrigamiEnv` is designed to pass Gymnasium's environment checker:

```python
from gymnasium.utils.env_checker import check_env

check_env(OrigamiEnv(space=space, max_steps=3), skip_render_check=True)
```

### Rendering

`render_mode="rgb_array"` uses MuJoCo's classic offscreen renderer, which needs
a working GL context (EGL/OSMesa). `render()` raises a clear `RuntimeError` if
rendering is unavailable on the machine. `close()` releases the renderer.

## Vectorized: `OrigamiVectorEnv`

`OrigamiVectorEnv` steps `N` environments together with `jax.vmap` over
`jax_env.step`, so they share one compiled model:

```python
from origami_engine_mjx.integrations.env import OrigamiVectorEnv

venv = OrigamiVectorEnv(space, n_envs=64, max_steps=100, frame_skip=1)
obs, info = venv.reset(seed=0)  # obs: (64, dim+3)
obs, reward, terminated, truncated, info = venv.step(actions)  # actions: (64, dim)
```

Every environment shares the same fixed crease topology (and therefore the same
model); only their states differ. The API follows the common vector-env
convention (`reset` returns `(obs, info)`; `step` returns
`(obs, rewards, terminated, truncated, info)`), with a leading `N` axis on every
array. `terminated` is all-`False` (only `truncated` ends episodes).

> Compiling the vmapped step is expensive; this path is intended for GPU
> workloads. On CPU it is dominated by `mjx.step`.

## Pure JAX: `reset` / `step`

For JAX-native pipelines, `integrations/jax_env.py` exposes the environment as
pure functions of pytrees:

```python
import jax, jax.numpy as jnp, numpy as np
from origami_engine_mjx.integrations import jax_env

model, classic, info = jax_env.build_jax_model(space.decode(np.zeros(space.dim)))
params = jax_env.make_params(space, classic, info, frame_skip=1, max_steps=100)
state = jax_env.reset(None, model, params)  # EnvState pytree

step = jax.jit(lambda s, a: jax_env.step(model, s, params, a))
state, obs, reward, done = step(state, jnp.zeros(space.dim))
```

- `EnvState` and `EnvParams` are `NamedTuple` pytrees; `model` is an MJX model
  pytree. All of `jit`, `vmap` and `scan` work.
- `reset` is deterministic; the `key` argument is accepted for API
  compatibility and ignored.
- `observe(state, params)` returns the same observation vector as the Gymnasium
  environment.
- `action` is a normalized `[-1, 1]` vector mapped onto the angle bounds; the
  reward is the change in forward (+x) displacement.

### Differentiability caveat

`mjx.step` is differentiable in principle, but this model is **not usefully
differentiable**: mesh–mesh contacts and the discrete face graph are
non-smooth, so gradients through `step` are unreliable. Use the black-box
scores from [OPTIMIZATION_ES.md](OPTIMIZATION_ES.md) for gradient-free
optimizers instead of trying to backpropagate through the physics.

### Cost caveat

JIT-compiling `mjx.step` for this contact model is expensive (the first trace
can take tens of seconds on CPU). Wrap `step` in `jax.jit` **once**, outside the
training loop, and reuse the compiled function.

## Extras and imports

```bash
pip install 'origami-engine-mjx[rl]'   # gymnasium
pip install 'origami-engine-mjx[jax]'  # mujoco-mjx, jax
```

`OrigamiEnv` / `OrigamiVectorEnv` (and the `jax_env` helpers `EnvState`,
`EnvParams`, `build_jax_model`, `make_params`, `observe`) are exposed lazily from
`origami_engine_mjx.integrations`, so importing the package never pulls in
Gymnasium or JAX unless you touch them. The pure-JAX `reset` / `step` functions
are reached through the `integrations.jax_env` submodule, since their names are
too generic to re-export safely.

## See also

- [INTEGRATIONS.md](INTEGRATIONS.md) — package overview and boundaries
- [OPTIMIZATION_ES.md](OPTIMIZATION_ES.md) — the black-box optimization guide
- [SIMULATION.md](SIMULATION.md) — the stepping loop the environments wrap
- [BACKENDS.md](BACKENDS.md) — why only the JAX backend is wrapped
- [`examples/05_rl_env.py`](../examples/05_rl_env.py) — runnable demo
