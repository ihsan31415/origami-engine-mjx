"""Compliance tests for the Gymnasium / JAX-functional RL environments.

Requires the ``[rl]`` (gymnasium) and ``[jax]`` extras.  Simulation-backed tests
are marked ``slow`` (MJX-JAX on CPU is seconds per tiny step).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
pytest.importorskip("jax")
pytest.importorskip("mujoco.mjx")

from origami_engine_mjx import Crease
from origami_engine_mjx.integrations import SearchSpace, jax_env
from origami_engine_mjx.integrations.env import OrigamiEnv, OrigamiVectorEnv
from origami_engine_mjx.integrations.jax_env import build_jax_model, make_params


@pytest.fixture(scope="module")
def tiny_space():
    creases = [
        Crease((0.2, 0.0), (0.5, 0.5), -0.7853981633974483),
        Crease((0.5, 0.5), (0.8, 0.0), -0.7853981633974483),
    ]
    return SearchSpace.from_creases(creases)


def test_env_spaces(tiny_space):
    env = OrigamiEnv(space=tiny_space)
    assert env.observation_space.shape == (tiny_space.dim + 3,)
    assert env.action_space.shape == (tiny_space.dim,)
    assert env.action_space.dtype == np.float32


def test_env_rejects_unknown_render_mode(tiny_space):
    with pytest.raises(ValueError):
        OrigamiEnv(space=tiny_space, render_mode="human")


def test_env_render_without_render_mode_returns_none(tiny_space):
    env = OrigamiEnv(space=tiny_space)
    assert env.render() is None
    env.close()


def test_env_rejects_bad_action_mode(tiny_space):
    with pytest.raises(ValueError):
        OrigamiEnv(space=tiny_space, action_mode="relative")


def test_env_rejects_non_jax_backend(tiny_space):
    with pytest.raises(ValueError):
        OrigamiEnv(space=tiny_space, backend="warp")


def test_env_step_before_reset_raises(tiny_space):
    env = OrigamiEnv(space=tiny_space)
    with pytest.raises(RuntimeError):
        env.step(np.zeros(tiny_space.dim, dtype=np.float32))


def test_action_order_aligns_with_search_space():
    """Genome entry j must drive the hinge of crease ``space.active_indices[j]``.

    The model emits hinges in a depth-first traversal, which for the default
    pattern differs from sorted crease order, so ``make_params`` must expose a
    non-identity permutation.  Regression test for a silent action/hinge swap.
    """
    space = SearchSpace.from_default_pattern()
    _, classic, info = build_jax_model(space.decode(np.zeros(space.dim)))
    params = make_params(space, classic, info)

    assert {int(c) for c in info["hinge_creases"]} == {int(i) for i in space.active_indices}
    ctrl_perm = np.asarray(params.ctrl_perm)
    for h, j in enumerate(ctrl_perm.tolist()):
        assert int(space.active_indices[j]) == int(info["hinge_creases"][h])
    assert not np.array_equal(ctrl_perm, np.arange(space.dim))  # genuinely permuted
    # hinge_adr is reordered to genome order too.
    hinge_adr = np.asarray(params.hinge_adr)
    model_adr = [classic.joint(h["name"]).qposadr[0] for h in info["hinges"]]
    for j in range(space.dim):
        h = int(np.where(ctrl_perm == j)[0][0])
        assert hinge_adr[j] == model_adr[h]


def test_make_params_rejects_mismatched_space(tiny_space):
    """A space whose crease set differs from the model must be rejected."""
    space = SearchSpace.from_default_pattern()
    _, classic, info = build_jax_model(tiny_space.decode(np.zeros(tiny_space.dim)))
    with pytest.raises(ValueError):
        make_params(space, classic, info)


def test_make_params_requires_hinge_crease_map(tiny_space):
    """``info`` without the hinge→crease map is rejected, not assumed identity."""
    _, classic, info = build_jax_model(tiny_space.decode(np.zeros(tiny_space.dim)))
    info.pop("hinge_creases")
    with pytest.raises(ValueError):
        make_params(tiny_space, classic, info)


@pytest.mark.slow
def test_step_ctrl_uses_search_space_order():
    """End-to-end: the control sent to the model is permuted into genome order.

    Uses a 3-strip pattern whose model hinge order is the reverse of its crease
    order, so a missing permutation would send each action to the wrong hinge.
    """
    creases = [
        Crease((0.0, 1.0 / 3), (1.0, 1.0 / 3), -0.3),
        Crease((0.0, 2.0 / 3), (1.0, 2.0 / 3), 0.4),
    ]
    space = SearchSpace.from_creases(creases)
    model, classic, info = build_jax_model(space.decode(np.zeros(space.dim)))
    params = make_params(space, classic, info)
    assert not np.array_equal(np.asarray(params.ctrl_perm), np.arange(space.dim))

    state = jax_env.reset(jax.random.key(0), model, params)
    action = np.array([1.0, -1.0], dtype=np.float32)
    state, _, _, _ = jax_env.step(model, state, params, jnp.asarray(action))
    angles = space.from_normalized(action)
    assert np.allclose(np.asarray(state.data.ctrl), angles[np.asarray(params.ctrl_perm)])


@pytest.mark.slow
def test_env_reset_and_step(tiny_space):
    env = OrigamiEnv(space=tiny_space, max_steps=2, frame_skip=1)
    obs, info = env.reset(seed=0)
    assert obs.shape == (tiny_space.dim + 3,)
    assert info["status"] == "ok"
    obs, reward, terminated, truncated, info = env.step(np.zeros(tiny_space.dim, dtype=np.float32))
    assert obs.shape == (tiny_space.dim + 3,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


@pytest.mark.slow
def test_env_passes_gymnasium_check_env(tiny_space):
    from gymnasium.utils.env_checker import check_env

    env = OrigamiEnv(space=tiny_space, max_steps=2, frame_skip=1)
    check_env(env, skip_render_check=True)


@pytest.mark.slow
def test_vector_env_reset_and_step(tiny_space):
    venv = OrigamiVectorEnv(tiny_space, n_envs=2, max_steps=2, frame_skip=1)
    obs, info = venv.reset(seed=0)
    assert obs.shape == (2, tiny_space.dim + 3)
    obs, reward, terminated, truncated, info = venv.step(np.zeros((2, tiny_space.dim), dtype=np.float32))
    assert obs.shape == (2, tiny_space.dim + 3)
    assert reward.shape == (2,)
    assert terminated.shape == (2,)
    assert truncated.shape == (2,)
    assert "step_count" in info
