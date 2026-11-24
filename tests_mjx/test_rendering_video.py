"""Recorded-trajectory and video tests (MJX-JAX on CPU; marked ``slow``).

These exercise the real physics loop, so they are deselected by default:

    pytest tests_mjx/test_rendering_video.py -m slow

The video tests need MuJoCo's offscreen GL context (``MUJOCO_GL=egl`` or
``osmesa``) and ``ffmpeg`` on ``PATH``; they skip cleanly otherwise.
"""

import os

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("mujoco.mjx")

from origami_engine_mjx import Crease, OrigamiEngine
from origami_engine_mjx.rendering import Trajectory, render_frames, render_video


@pytest.fixture(scope="module")
def tiny_creases():
    """Minimal valid pattern: one mountain fold -> 2 faces, 1 hinge."""
    return [
        Crease((0.2, 0.0), (0.5, 0.5), -0.7853981633974483),
        Crease((0.5, 0.5), (0.8, 0.0), -0.7853981633974483),
    ]


@pytest.fixture(scope="module")
def recorded(tiny_creases):
    engine = OrigamiEngine(backend="jax")
    return engine.run(tiny_creases, duration=0.004, fold_duration=0.002, vibrate=True, record=True, record_every=2)


@pytest.mark.slow
def test_record_does_not_change_physics(tiny_creases):
    engine = OrigamiEngine(backend="jax")
    plain = engine.run(tiny_creases, duration=0.004, fold_duration=0.002, vibrate=True)
    rec = engine.run(tiny_creases, duration=0.004, fold_duration=0.002, vibrate=True, record=True)
    assert "trajectory" not in plain.extra
    assert np.allclose(np.asarray(plain.data.qpos), np.asarray(rec.data.qpos))
    assert np.allclose(plain.final_centroid, rec.final_centroid)


@pytest.mark.slow
def test_evaluate_return_trajectory_matches_run(tiny_creases):
    engine = OrigamiEngine(backend="jax")
    score, traj = engine.evaluate(
        tiny_creases, duration=0.004, fold_duration=0.002, vibrate=False, return_trajectory=True
    )
    assert np.isfinite(score)
    assert traj["qpos"].shape[0] == traj["time"].size > 1
    assert traj["time"][-1] == pytest.approx(0.004, rel=0.2)


@pytest.mark.slow
def test_recorded_trajectory_endpoints(recorded):
    traj = Trajectory.from_result(recorded)
    assert traj.time[0] == pytest.approx(0.0)
    assert len(traj) >= 2
    assert np.allclose(traj.qpos[-1], np.asarray(recorded.data.qpos), atol=1e-5)
    assert np.allclose(traj.centroid[-1], recorded.final_centroid, atol=1e-6)
    assert traj.table_offset == recorded.info["table_qposadr"]
    assert traj.hinge_angles().shape[1] == recorded.info["n_hinges"]
    # Net displacement equals the scored quantity.
    assert np.allclose(traj.displacement, recorded.final_centroid - recorded.initial_centroid)


def _gl_and_ffmpeg_available() -> bool:
    from origami_engine_mjx.rendering import ffmpeg_available

    if ffmpeg_available() is None:
        return False
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        import mujoco

        model = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom size='.1'/></worldbody></mujoco>")
        renderer = mujoco.Renderer(model, height=32, width=32)
        renderer.close()
    except Exception:  # noqa: BLE001 - no GL context on this machine
        return False
    return True


requires_gl = pytest.mark.skipif(not _gl_and_ffmpeg_available(), reason="needs offscreen GL (EGL/OSMesa) and ffmpeg")


@pytest.mark.slow
@requires_gl
def test_render_video_all_sides_with_graphs(tmp_path, recorded):
    out = tmp_path / "four.mp4"
    path = render_video(
        recorded,
        out,
        views="all_sides",
        plot="xyz",
        max_frames=4,
        resolution=(160, 120),
        excitation={"freq_hz": 75.0, "amp_x": 0.0, "amp_z": 0.0013, "vibrate": True},
    )
    assert path == out and out.exists() and out.stat().st_size > 0
    # Four 120x90 views in one row (2x2 would need the graph layout).
    frames = render_frames(recorded, views="all_sides", max_frames=2, resolution=(120, 90))
    assert len(frames) == 2
    assert frames[0].shape == (90, 480, 3)
    # Cameras used are persisted for reproducibility.
    assert (tmp_path / "four.camera.json").exists()
