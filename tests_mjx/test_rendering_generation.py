"""Fast tests for generation logging and the rendering helpers (no GL needed)."""

import json

import matplotlib
import numpy as np
import pytest

from origami_engine_mjx.integrations import (
    BatchResult,
    EvalRecord,
    GenerationLogger,
    RunLogger,
    SearchSpace,
    load_best,
    load_generation_history,
)
from origami_engine_mjx.rendering import (
    CAMERA_PRESETS,
    CameraSpec,
    Trajectory,
    available_cameras,
    caption,
    cinematic_views,
    default_overlay,
    draw_text,
    hconcat,
    history_to_image,
    layout_frames,
    load_camera,
    overlay_text,
    plot_history,
    plot_trajectory,
    resolve_camera,
    resolve_cameras,
    save_camera,
    trajectory_to_image,
    vconcat,
    with_letterbox,
)

matplotlib.use("Agg")

import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Fixtures / fakes
# ----------------------------------------------------------------------


def _batch(fitness, status=None, cached=0.0):
    fit = np.asarray(fitness, dtype=float)
    n = fit.size
    return BatchResult(
        fitness=fit,
        raw=fit * 10,
        status=["ok"] * n if status is None else status,
        time=np.zeros(n),
        descriptors={"displacement_x": fit * 10},
        wall_time=0.01,
        n_cache_hits=int(cached),
    )


@pytest.fixture(scope="module")
def space():
    return SearchSpace.from_default_pattern()


class _FakeResult:
    """Looks enough like SimulationResult for Trajectory.from_result."""

    def __init__(self, traj, info=None):
        self.extra = {"trajectory": traj}
        self.info = info or {}


def _fake_traj(n=11, genes=4):
    t = np.linspace(0.0, 1.0, n)
    qpos = np.zeros((n, genes + 2))
    qpos[:, 0] = 0.001 * np.sin(2 * np.pi * t)  # table_x
    qpos[:, 1] = 0.0  # table_z
    qpos[:, 2:] = np.linspace(0.0, 0.5, n)[:, None] * np.ones(genes) * -1.0
    centroid = np.stack([0.5 + 0.1 * np.sin(np.pi * t), 0.5 + 0.05 * t], axis=1)
    return {"time": t, "qpos": qpos, "qvel": np.zeros_like(qpos), "centroid": centroid}


# ----------------------------------------------------------------------
# GenerationLogger
# ----------------------------------------------------------------------


def test_generation_logger_jsonl_maximising(tmp_path, space):
    log_path = tmp_path / "run.jsonl"
    rng = np.random.default_rng(0)
    with GenerationLogger(log_path, space) as log:
        for gen in range(4):
            genomes = space.sample(rng, size=5)
            batch = _batch(genomes.sum(axis=1))
            log.log_batch(batch, genomes, generation=gen)
            log.end_generation(gen, batch, genomes)

    assert len(log.history) == 4
    best_gen = int(np.argmax([s.best_fitness for s in log.history]))
    assert log.best_fitness == pytest.approx(log.history[best_gen].best_fitness)
    assert log.best_generation == best_gen
    assert log.evals_so_far == 20
    # improved flags trace the running best
    running = [s.best_fitness for s in log.history][: best_gen + 1]
    assert max(running) == pytest.approx(log.best_fitness)
    assert log.history[best_gen].improved

    # The single best checkpoint is loadable and equals the best genome.
    genome, meta = load_best(tmp_path / "run.best.json", space)
    assert np.allclose(genome, log.best_genome)
    assert meta["meta"]["generation"] == best_gen

    # Summary side file mirrors the history.
    payload = json.loads((tmp_path / "run.summary.json").read_text())
    assert len(payload["generations"]) == 4
    assert payload["best"]["generation"] == best_gen

    # Both record kinds share the file and eval rows carry the generation tag.
    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    kinds = [r["kind"] for r in rows]
    assert kinds.count("generation") == 4
    assert kinds.count("eval") == 20
    eval_rows = [r for r in rows if r["kind"] == "eval"]
    assert {r["generation"] for r in eval_rows} == {0, 1, 2, 3}

    hist = load_generation_history(log_path)
    assert [h["generation"] for h in hist] == [0, 1, 2, 3]


def test_generation_logger_minimising_csv_and_per_generation_checkpoints(tmp_path, space):
    log_path = tmp_path / "loss.csv"
    rng = np.random.default_rng(1)
    with GenerationLogger(log_path, space, maximising=False, keep_every_best=True) as log:
        for gen in range(3):
            genomes = space.sample(rng, size=4)
            batch = _batch(np.abs(genomes).sum(axis=1))  # a "loss"
            _ = log.end_generation(gen, batch, genomes)

    losses = [s.best_fitness for s in log.history]
    assert log.best_fitness == pytest.approx(min(losses))
    assert log.best_generation == int(np.argmin(losses))
    for gen in range(3):
        path = tmp_path / f"loss.best_gen_{gen:04d}.json"
        assert path.exists()
        loaded, meta = load_best(path, space)
        assert meta["meta"]["generation"] == gen
        assert np.allclose(loaded, log.history[gen].best_genome)

    header = log_path.read_text().splitlines()[0].split(",")
    assert "kind" in header and "best_fitness" in header
    hist = load_generation_history(log_path)
    assert len(hist) == 3
    assert hist[0]["best_genome"] is not None


def test_generation_logger_csv_mixes_record_kinds(tmp_path, space):
    log_path = tmp_path / "mixed.csv"
    with GenerationLogger(log_path, space) as log:
        genomes = space.sample(np.random.default_rng(3), size=2)
        batch = _batch(genomes.sum(axis=1))
        log.log_batch(batch, genomes, generation=0)
        log.end_generation(0, batch, genomes)
    rows = list(log_path.read_text().splitlines())
    header = rows[0].split(",")
    assert {"kind", "genome", "fitness", "best_fitness", "best_genome"} <= set(header)
    kinds = [row.split(",")[0] for row in rows[1:]]
    assert kinds.count("eval") == 2 and kinds.count("generation") == 1


def test_generation_logger_memory_only_and_curve(tmp_path, space):
    rng = np.random.default_rng(2)
    log = GenerationLogger(None, space)
    for gen in range(2):
        genomes = space.sample(rng, size=3)
        batch = _batch(genomes.sum(axis=1))
        log.end_generation(gen, batch, genomes)
    log.close()
    gens, best, mean, gbest = log.curve()
    assert gens.tolist() == [0, 1]
    assert np.all(gbest == np.maximum.accumulate(best))
    assert np.all(np.isfinite(mean))
    assert not list(tmp_path.iterdir())


def test_generation_logger_handles_nonfinite_and_failures(space):
    log = GenerationLogger(None, space)
    batch = _batch([np.nan, 2.0, -np.inf, 1.0], status=["nonfinite", "ok", "nonfinite", "invalid"])
    genomes = np.zeros((4, space.dim))
    summary = log.end_generation(7, batch, genomes)
    assert summary.best_index == 1
    assert summary.n_ok == 1
    assert summary.n_failed == 3  # three statuses other than "ok"
    assert summary.best_fitness == pytest.approx(2.0)
    assert summary.mean_fitness == pytest.approx(1.5)
    assert summary.worst_fitness == pytest.approx(1.0)


def test_generation_logger_validates_inputs(space):
    log = GenerationLogger(None, space)
    with pytest.raises(ValueError, match="at least one"):
        log.end_generation(0, _batch([]))
    with pytest.raises(ValueError, match="genomes"):
        log.end_generation(0, _batch([1.0, 2.0]), np.zeros((3, space.dim)))


def test_run_logger_include_generation_keeps_default_schema(tmp_path, space):
    default_path = tmp_path / "plain.jsonl"
    with RunLogger(default_path) as log:
        log.log(EvalRecord(fitness=1.0, raw=1.0), genome=[0.1])
        log.log_batch(_batch([2.0]), np.array([[0.2]]))
    rows = [json.loads(line) for line in default_path.read_text().splitlines()]
    assert all("generation" not in r for r in rows)

    tagged = tmp_path / "tagged.csv"
    with RunLogger(tagged, include_generation=True) as log:
        log.log(EvalRecord(fitness=1.0, raw=1.0), genome=[0.1], generation=3)
        log.log_batch(_batch([2.0]), np.array([[0.2]]), generation=4)
    lines = tagged.read_text().splitlines()
    assert lines[0].split(",")[0] == "generation"
    assert lines[1].split(",")[0] == "3"
    assert lines[2].split(",")[0] == "4"


# ----------------------------------------------------------------------
# Cameras
# ----------------------------------------------------------------------


def test_camera_presets_resolve():
    top = resolve_camera("top")
    assert isinstance(top, CameraSpec)
    assert top.azimuth == pytest.approx(90.0)
    assert top.elevation == pytest.approx(CAMERA_PRESETS["top"][1])
    assert resolve_camera(None).name == "default"
    assert resolve_camera("Default").name == "default"
    assert resolve_camera("Side Left").name == "side_left"
    assert resolve_camera("Top-Side-Left").name == "top_side_left"
    for name in ("top", "side", "top_side_left", "side_left", "iso_top"):
        assert resolve_camera(name).name == name


def test_camera_prefix_namespaces_views():
    spec = resolve_camera("cam_top", prefix="cam")
    assert spec.name == "top"
    assert resolve_camera("cam", prefix="cam").name == "default"
    with pytest.raises(ValueError, match="unknown camera preset"):
        resolve_camera("cam")  # a prefix alone is not a camera name
    views = cinematic_views("cam")
    assert views == ["cam_iso_top", "cam_iso_top_side", "cam_top", "cam_iso_top_left"]
    assert "cam_top" in available_cameras("cam")
    assert "top" in available_cameras()


def test_camera_manual_coordinates_and_zoom():
    manual = resolve_camera({"azimuth": 400, "elevation": -20, "distance": 1.2})
    assert manual.name == "manual"
    assert manual.azimuth == pytest.approx(40.0)
    assert manual.distance == pytest.approx(1.2)
    assert manual.lookat == (0.5, 0.5, 0.0)

    two = resolve_camera((0.0, -20.0))
    assert two.elevation == pytest.approx(-20.0)
    four = resolve_camera((10.0, -30.0, 1.0, 0.5))
    assert four.distance == pytest.approx(0.5)

    zoomed = resolve_camera("top", zoom=0.5, lookat=(1.0, 2.0, 0.3), bounds=(0, 0, 4, 4))
    assert zoomed.distance == pytest.approx(CAMERA_PRESETS["top"][2] * 0.5)
    assert zoomed.lookat == (1.0, 2.0, 0.3)
    assert resolve_camera("front", bounds=(0.0, 0.0, 1.0, 0.5)).lookat == (0.5, 0.25, 0.0)


def test_camera_errors_are_clear():
    with pytest.raises(ValueError, match="unknown camera preset"):
        resolve_camera("diagonal")
    with pytest.raises(ValueError, match="positive"):
        resolve_camera("top", zoom=0.0)
    with pytest.raises(ValueError):
        resolve_camera((1.0,))
    assert len(resolve_cameras("top")) == 1
    assert len(resolve_cameras(["top", (0, -20)])) == 2
    with pytest.raises(ValueError):
        resolve_cameras([])


def test_camera_persistence_roundtrip(tmp_path):
    spec = resolve_camera((137.0, 22.0, 1.45))
    path = save_camera(spec, tmp_path / "cam.json")
    loaded = load_camera(path)
    assert loaded == spec
    assert spec.as_dict()["azimuth"] == pytest.approx(137.0)


# ----------------------------------------------------------------------
# Trajectory
# ----------------------------------------------------------------------


def test_trajectory_from_result_and_motion():
    res = _FakeResult(_fake_traj(11), info={"table_qposadr": (0, 1)})
    traj = Trajectory.from_result(res)
    assert len(traj) == 11
    assert traj.duration == pytest.approx(1.0)
    assert traj.hinge_angles().shape == (11, 4)
    com = traj.center_of_mass(relative=True)
    assert np.allclose(com[0], 0.0)
    vel = traj.velocity(com)
    assert vel.shape == com.shape
    assert traj.subsample(5).tolist() == [0, 2, 5, 7, 10]
    sub = traj.subset([0, 5, 10])
    assert len(sub) == 3 and sub.time[-1] == pytest.approx(1.0)
    assert traj.as_dict()["centroid"].shape == (11, 2)


def test_trajectory_without_table_offset_keeps_all_joints():
    traj = Trajectory.from_dict(_fake_traj(4, genes=3))
    assert traj.hinge_angles().shape == (4, 5)


def test_trajectory_uses_recorded_hinge_columns():
    data = _fake_traj(4, genes=3)
    data["hinge_qposadr"] = np.array([2, 4])
    traj = Trajectory.from_dict(data)
    assert traj.hinge_angles().shape == (4, 2)
    assert np.allclose(traj.hinge_angles(), data["qpos"][:, [2, 4]])
    assert np.array_equal(traj.as_dict()["hinge_qposadr"], [2, 4])
    assert traj.subset([1, 2]).hinge_qposadr is not None


def test_trajectory_requires_recording():
    class Bare:
        def __init__(self):
            self.extra = {}

    with pytest.raises(ValueError, match="record=True"):
        Trajectory.from_result(Bare())


def test_trajectory_rejects_bad_shapes():
    with pytest.raises(ValueError):
        Trajectory([0, 1], np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((1, 2)))


# ----------------------------------------------------------------------
# Compose helpers
# ----------------------------------------------------------------------


def test_compose_concat_letterbox_and_text():
    red = np.zeros((40, 60, 3), dtype=np.uint8)
    red[..., 0] = 255
    blue = np.zeros((40, 60, 3), dtype=np.uint8)
    blue[..., 2] = 255

    side = hconcat([red, blue])
    assert side.shape == (40, 120, 3)
    assert side[0, 0, 0] == 255 and side[0, -1, 2] == 255

    stacked = vconcat([red, blue])
    assert stacked.shape == (80, 60, 3)

    wide = with_letterbox(np.zeros((20, 80, 3), dtype=np.uint8), (40, 40), background=7)
    assert wide.shape == (40, 40, 3)
    assert wide[0, 0, 0] == 7  # letterbox padding
    assert wide[20, 20, 0] == 0  # image content

    white = np.full((80, 120, 3), 255, dtype=np.uint8)
    marked = caption(white, "freq=75.0 Hz")
    assert marked.shape == white.shape
    assert not np.array_equal(marked, white)
    assert np.array_equal(marked[:30, -10:], white[:30, -10:])  # top-right untouched
    assert draw_text(white, "x", (4, 4), anchor="bottom-left").shape == white.shape
    assert overlay_text(white, "x").shape == white.shape


def test_layout_frames_grid_and_overlay():
    tiles = [np.full((90, 120, 3), 90, dtype=np.uint8) for _ in range(4)]
    frame = layout_frames(
        tiles,
        overlay="freq={freq_hz} Hz",
        overlay_values={"freq_hz": 75.0},
        resolution=(120, 90),
    )
    assert frame.shape == (90, 480, 3)  # one row of four views
    assert frame[-20:, :80].min() < 90  # overlay box darkens the bottom-left
    assert frame[:40, :80].min() == 90  # rest of the tile untouched

    single = layout_frames(tiles[:1], resolution=(120, 90))
    assert single.shape == (90, 120, 3)

    with_graph = layout_frames(tiles[:1], plot="x", plot_images=[np.zeros((90, 60, 3), np.uint8)], resolution=(120, 90))
    assert with_graph.shape == (90, 180, 3)

    with pytest.raises(ValueError):
        layout_frames([])


def test_overlay_helpers():
    text = default_overlay(75.0, 0.0, 0.0013)(0.0, 0.0)
    assert "freq=75.0 Hz" in text and "amp z=0.0013 m" in text
    assert default_overlay(75.0, vibrate=False)(0.0, 0.0) == "vibration off"
    assert "gen 3" in default_overlay(75.0, extra="gen 3")(0.0, 0.0)


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------


def test_trajectory_to_image_and_progress():
    traj = Trajectory.from_dict(_fake_traj(9))
    img = trajectory_to_image(traj, size=(200, 140), axis="x")
    assert img.shape == (140, 200, 3) and img.dtype == np.uint8
    half = trajectory_to_image(traj, size=(200, 140), axis="x", progress=0.5, style="dark")
    assert half.shape == (140, 200, 3)
    # The progress marker changes pixels near the curve, not the frame size.
    assert not np.array_equal(img, half)
    for axis in ("y", "z"):
        assert trajectory_to_image(traj, size=(120, 90), axis=axis).shape == (90, 120, 3)
    with pytest.raises(ValueError, match="axis"):
        trajectory_to_image(traj, axis="w")


def test_plot_trajectory_writes_file(tmp_path):
    traj = Trajectory.from_dict(_fake_traj(9))
    path = tmp_path / "movement.png"
    fig, axes = plot_trajectory(traj, axis="all", path=path, dpi=60)
    assert path.exists() and path.stat().st_size > 0
    assert len(axes) >= 3  # x, y, z panels (+ twin rate axes for x/y)
    fig.clf()
    fig, axes = plot_trajectory(traj, axis="y", path=tmp_path / "y.png", dpi=60)
    assert path.parent.joinpath("y.png").exists()
    fig.clf()


def test_plot_history_from_logger_and_rows(tmp_path, space):
    rng = np.random.default_rng(4)
    log = GenerationLogger(None, space)
    for gen in range(3):
        genomes = space.sample(rng, size=4)
        batch = _batch(genomes.sum(axis=1))
        log.end_generation(gen, batch, genomes)

    fig, ax = plot_history(log, path=tmp_path / "history.png", dpi=60)
    assert (tmp_path / "history.png").stat().st_size > 0
    assert "higher is better" in ax.get_ylabel()
    fig.clf()
    plt.close(fig)

    rows = [s.as_dict() for s in log.history]
    img = history_to_image(rows, size=(240, 160), dpi=50)
    assert img.shape == (160, 240, 3) and img.dtype == np.uint8

    loss = GenerationLogger(None, space, maximising=False)
    loss.end_generation(0, _batch([3.0, 1.0]), np.zeros((2, space.dim)))
    fig, ax = plot_history(loss.history, dpi=60, maximising=loss.maximising)
    assert "lower is better" in ax.get_ylabel()
    fig.clf()
    plt.close(fig)

    with pytest.raises(ValueError, match="empty"):
        plot_history([])


def test_trajectory_tolerates_single_frame():
    traj = Trajectory.from_dict(_fake_traj(1))
    assert traj.velocity().shape == (1, 2)
    assert trajectory_to_image(traj, size=(80, 60), axis="x").shape == (60, 80, 3)
