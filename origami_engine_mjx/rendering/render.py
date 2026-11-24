from __future__ import annotations

"""Render a recorded simulation to video, with cameras, panels and overlays.

This module is the rendering entry point.  It is **additive**: it never changes
``OrigamiEnv.render`` or any existing behaviour; it draws a trajectory you
recorded with ``engine.run(..., record=True)`` (or ``engine.evaluate(...,
return_trajectory=True)``).

Quick start
-----------
>>> result = engine.run(creases, duration=1.0, record=True)   # doctest: +SKIP
>>> render_video(result, "out.mp4")                          # doctest: +SKIP
>>> render_video(result, "four_views.mp4", views="all_sides")  # doctest: +SKIP
>>> render_video(result, "with_graphs.mp4", plot="xyz")        # doctest: +SKIP

Cameras
-------
``camera=`` takes one view; ``cameras=``/``views=`` take several and lay them
out in one video:

* a **preset name** — ``"default"``, ``"top"``, ``"side"``, ``"side_left"``,
  ``"top_side_left"``, ``"iso_top"``, … (see
  :data:`~origami_engine_mjx.rendering.cameras.CAMERA_PRESETS`);
* **manual coordinates** — a mapping ``{"azimuth": 40, "elevation": -20,
  "distance": 1.4, "zoom": 1.0, "lookat": (0.5, 0.5, 0)}`` or a tuple
  ``(azimuth, elevation[, distance[, zoom]])``.  Angles in degrees;
* ``views="all_sides"`` — the four built-in look-down views.
* ``prefix=`` namespaces preset names (``prefix="run7"`` accepts
  ``"run7_top"``).  The resolved preset name is always saved next to the
  video in ``<stem>.camera.json`` so a render can be reproduced.

Overlay
-------
Every frame carries the current excitation in its bottom-left corner
(``freq`` and ``amp``), via ``overlay="freq/amp"`` (default) or a custom
format string / callable.  ``overlay=None`` disables it.

Layout
------
``plot=`` puts movement graphs beside the simulation:
``"x"``, ``"y"``, ``"z"``, ``"xyz"`` (stacked) or a sequence such as
``["x", "y"]`` (left to right).  Graphs are vector-quality matplotlib panels
rendered to the frame buffer, not GL.
"""

import json
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .cameras import ALL_SIDES_ALIASES, cinematic_views, resolve_cameras
from .compose import draw_text, hconcat, vconcat, with_letterbox
from .plots import trajectory_to_image
from .trajectory import Trajectory
from .video import VideoWriter, ffmpeg_available

__all__ = [
    "CameraSpec",
    "Overlay",
    "default_overlay",
    "layout_frames",
    "load_camera",
    "render_frames",
    "render_video",
    "save_camera",
    "trajectory_frames",
]


@dataclass(frozen=True)
class CameraSpec:
    """A resolved camera: MuJoCo free-camera parameters in degrees/metres."""

    name: str
    lookat: tuple[float, float, float]
    distance: float
    azimuth: float
    elevation: float

    def as_dict(self) -> dict[str, Any]:
        """Return the parameters as a plain, JSON-friendly dictionary."""
        return {
            "name": self.name,
            "lookat": [float(v) for v in self.lookat],
            "distance": float(self.distance),
            "azimuth": float(self.azimuth),
            "elevation": float(self.elevation),
        }

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.name}(az={self.azimuth:g}, el={self.elevation:g}, dist={self.distance:g})"


#: An overlay is either a format string (``"{freq:.1f} Hz"``) or a callable
#: ``(t, frac) -> text``.
Overlay = str | Callable[[float, float], str] | None

_PANEL_INDICES = {"x": 0, "y": 1, "z": 2}


# ----------------------------------------------------------------------
# Overlay
# ----------------------------------------------------------------------


def default_overlay(
    freq_hz: float,
    amp_x: float = 0.0,
    amp_z: float | None = None,
    *,
    vibrate: bool = True,
    extra: str | None = None,
) -> Overlay:
    """Return a callable overlay that prints the current freq/amp in the corner.

    The text is::

        freq=75.0 Hz
        amp x=0.0000 m
        amp z=0.0013 m

    ``t`` is the current simulated time; the values are constant for a run
    (the table is driven by one sinusoid, see ``constants.py``).  With
    ``vibrate=False`` the first line becomes ``freq=off`` and amplitudes are
    hidden.  ``extra`` is appended as a final line (e.g. ``"best gen 7"``).
    """
    if not vibrate:

        def _off(t: float, frac: float) -> str:
            del t, frac
            return "vibration off" + (f"\n{extra}" if extra else "")

        return _off

    amp_z_val = 0.0 if amp_z is None else float(amp_z)
    amp_x_val = float(amp_x)

    def _text(t: float, frac: float) -> str:
        del t, frac
        text = f"freq={float(freq_hz):.1f} Hz\namp x={amp_x_val:.4f} m\namp z={amp_z_val:.4f} m"
        if extra:
            text = f"{text}\n{extra}"
        return text

    return _text


def _overlay_text(overlay: Overlay, t: float, frac: float, values: Mapping[str, Any], index: int) -> str | None:
    if overlay is None:
        return None
    if callable(overlay):
        return str(overlay(t, frac))
    try:
        return str(overlay.format(t=t, frac=frac, index=index, **values))
    except (KeyError, IndexError, ValueError):
        return str(overlay)


# ----------------------------------------------------------------------
# Camera persistence
# ----------------------------------------------------------------------


def save_camera(camera: CameraSpec | str | Mapping[str, Any] | Sequence[float] | None, path: str | Path) -> Path:
    """Write a resolved camera (and preset lookup values) to ``path`` as JSON."""
    spec = camera if isinstance(camera, CameraSpec) else resolve_cameras(camera)[0]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"camera": spec.as_dict()}
    out.write_text(json.dumps(payload, indent=2))
    return out


def load_camera(path: str | Path) -> CameraSpec:
    """Read a camera saved by :func:`save_camera`."""
    payload = json.loads(Path(path).read_text())
    cam = payload["camera"]
    lookat = [float(v) for v in cam["lookat"]]
    if len(lookat) != 3:
        raise ValueError(f"{path}: saved camera lookat must have three components")
    return CameraSpec(
        name=str(cam.get("name", "saved")),
        lookat=(lookat[0], lookat[1], lookat[2]),
        distance=float(cam["distance"]),
        azimuth=float(cam["azimuth"]),
        elevation=float(cam["elevation"]),
    )


# ----------------------------------------------------------------------
# Frame generation
# ----------------------------------------------------------------------


def _require_mujoco():
    try:
        import mujoco
    except Exception as exc:
        raise RuntimeError(
            "rendering a video needs MuJoCo with an offscreen GL context (EGL or OSMesa); "
            "install 'origami-engine-mjx[jax]' or 'mujoco>=3' and set MUJOCO_GL=egl"
        ) from exc
    return mujoco


def _mjv_camera(mujoco, model, spec: CameraSpec):
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.lookat[:] = np.asarray(spec.lookat, dtype=float)
    cam.distance = float(spec.distance)
    cam.azimuth = float(spec.azimuth)
    cam.elevation = float(spec.elevation)
    return cam


def _scene_bounds(result: Any) -> tuple[float, float, float, float] | None:
    """Approximate XY extent of the paper from the pattern, when available."""
    pat = getattr(result, "pattern", None)
    faces = getattr(pat, "faces", None)
    if not faces:
        return None
    pts: list[np.ndarray] = []
    for face in faces:
        exterior = getattr(face, "exterior", None)
        if exterior is None:
            continue
        pts.append(np.asarray(exterior.coords, dtype=float))
    if not pts:
        return None
    allpts = np.concatenate(pts, axis=0)
    return (float(allpts[:, 0].min()), float(allpts[:, 1].min()), float(allpts[:, 0].max()), float(allpts[:, 1].max()))


def _render_frames(
    result: Any,
    trajectory: Trajectory,
    cameras: Sequence[CameraSpec],
    width: int,
    height: int,
) -> Iterable[list[np.ndarray]]:
    """Yield one list of camera tiles per trajectory frame."""
    mujoco = _require_mujoco()
    classic = (getattr(result, "model", None) or {})["classic_model"]
    data = mujoco.MjData(classic)
    renderer = mujoco.Renderer(classic, height=int(height), width=int(width))
    cams = [_mjv_camera(mujoco, classic, spec) for spec in cameras]
    try:
        for i in range(len(trajectory)):
            data.qpos[:] = trajectory.qpos[i]
            data.qvel[:] = trajectory.qvel[i]
            mujoco.mj_forward(classic, data)
            tiles = []
            for cam in cams:
                renderer.update_scene(data, camera=cam)
                tiles.append(renderer.render().copy())
            yield tiles
    finally:
        # Some MuJoCo builds raise from ``Renderer.__del__`` after ``close()``;
        # closing explicitly still frees the GL context early.
        renderer.close()


def trajectory_frames(
    result: Any,
    *,
    camera=None,
    cameras=None,
    views: str | Sequence[Any] | None = None,
    prefix: str | None = None,
    width: int = 480,
    height: int = 360,
    lookat: Sequence[float] | None = None,
    zoom: float = 1.0,
) -> list[np.ndarray]:
    """Render every recorded frame and return the composited RGB frames.

    ``camera`` / ``cameras`` take presets or manual coordinates (see the module
    docstring).  ``views="all_sides"`` uses :func:`cinematic_views`.  Multiple
    cameras are concatenated left-to-right; with two cameras they are wrapped
    into a 2×1 row as well, so four cameras form a 2×2 grid at the same total
    width.  Each camera tile is letterboxed to ``(width, height)`` so
    concatenation never distorts the image.
    """
    traj = _trajectory_of(result)
    if views is not None:
        if isinstance(views, str):
            views = cinematic_views(prefix) if views in ALL_SIDES_ALIASES else [views]
        camera = views
    if cameras is not None:
        camera = cameras
    bounds = _scene_bounds(result)
    if lookat is None and bounds is not None:
        lookat = ((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2, 0.0)
    specs = resolve_cameras(camera, prefix=prefix, zoom=zoom, lookat=lookat)
    frames: list[np.ndarray] = []
    for tiles in _render_frames(result, traj, specs, width, height):
        letterboxed = [with_letterbox(t, (width, height)) for t in tiles]
        if len(letterboxed) == 1:
            frames.append(letterboxed[0])
        else:
            frames.append(hconcat(letterboxed))
    return frames


def _trajectory_of(result: Any) -> Trajectory:
    if isinstance(result, Trajectory):
        raise TypeError("video rendering needs a SimulationResult with a classic model, not a bare Trajectory")
    return Trajectory.from_result(result)


# ----------------------------------------------------------------------
# Layout / compositing
# ----------------------------------------------------------------------


def layout_frames(
    tiles: Sequence[Any],
    *,
    plot: str | Sequence[str] | None = None,
    plot_images: Sequence[Any] | None = None,
    overlay: Overlay = None,
    overlay_values: Mapping[str, Any] | None = None,
    t: float = 0.0,
    frac: float = 0.0,
    resolution: tuple[int, int] = (480, 360),
) -> np.ndarray:
    """Composite one output frame from camera tiles, plot panels and overlay.

    Not usually called directly — :func:`render_frames` and
    :func:`render_video` use it — but it is useful for tests and notebooks.

    * ``tiles`` are letterboxed to ``resolution={'x', 'y'}`` and concatenated
      left-to-right on a grid canvas (2 tiles -> 2×1, 3-4 tiles -> 2×2).
    * ``plot`` selects movement panels: ``{"x","y","z","xyz"}`` or a sequence;
      ``plot_images`` supplies the rendered panel images (one per panel).
    * ``overlay`` is drawn in the bottom-left corner of the video area, after
      the graphs, so the video always carries its freq/amp log.
    """
    width, height = int(resolution[0]), int(resolution[1])
    boxes = [with_letterbox(tile, (width, height)) for tile in tiles]
    if not boxes:
        raise ValueError("at least one camera tile is required")
    scene = boxes[0] if len(boxes) == 1 else hconcat(boxes)
    if plot_images:
        panels = [with_letterbox(img, (width // 2, (2 * height) // len(plot_images))) for img in plot_images]
        graph = vconcat(panels) if len(panels) > 1 else panels[0]
        graph = with_letterbox(graph, (width // 2, scene.shape[0]))
        scene = hconcat([scene, graph])
    if overlay is not None:
        text = _overlay_text(overlay, t, frac, overlay_values or {}, round(frac * 1000))
        if text:
            scene = draw_text(scene, text, (6, 6), anchor="bottom-left", font_size=12)
    return scene


def render_frames(
    result: Any,
    *,
    camera=None,
    cameras=None,
    views: str | Sequence[Any] | None = None,
    prefix: str | None = None,
    plot: str | Sequence[str] | None = None,
    overlay: Overlay | str = "freq/amp",
    resolution: tuple[int, int] = (480, 360),
    max_frames: int | None = None,
    lookat: Sequence[float] | None = None,
    zoom: float = 1.0,
    excitation: Mapping[str, Any] | None = None,
    overlay_extra: str | None = None,
) -> list[np.ndarray]:
    """Return all composited output frames (no encoding).

    Useful for notebooks (``display`` the frames) and for tests.  See
    :func:`render_video` for the arguments.
    """
    traj = _trajectory_of(result)
    width, height = int(resolution[0]), int(resolution[1])
    if views is not None:
        if isinstance(views, str):
            views = cinematic_views(prefix) if views in ALL_SIDES_ALIASES else [views]
        camera = views
    if cameras is not None:
        camera = cameras
    bounds = _scene_bounds(result)
    if lookat is None and bounds is not None:
        lookat = ((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2, 0.0)
    specs = resolve_cameras(camera, prefix=prefix, zoom=zoom, lookat=lookat)

    plot_axes = _normalise_plot(plot)
    camera_res = (width if not plot_axes else width // 2, height)

    if isinstance(overlay, str) and overlay in {"freq/amp", "freq", "default"}:
        exc = dict(excitation or {})
        overlay = default_overlay(
            exc.get("freq_hz", 0.0),
            exc.get("amp_x", 0.0),
            exc.get("amp_z", 0.0),
            vibrate=bool(exc.get("vibrate", True)),
            extra=overlay_extra,
        )

    n_frames = len(traj)
    subset = traj
    if max_frames is not None and 0 < int(max_frames) < n_frames:
        subset = traj.subset(traj.subsample(int(max_frames)))

    frames: list[np.ndarray] = []
    for k, tiles in enumerate(_render_frames(result, subset, specs, camera_res[0], camera_res[1])):
        frac = k / max(1, len(subset) - 1) if len(subset) > 1 else 0.0
        plot_images = _plot_panels(subset, plot_axes, resolution, frac) if plot_axes else None
        frames.append(
            layout_frames(
                tiles,
                plot=plot,
                plot_images=plot_images,
                overlay=overlay,
                overlay_values={"freq_hz": (excitation or {}).get("freq_hz"), "amp": (excitation or {}).get("amp_z")},
                t=float(subset.time[k]),
                frac=frac,
                resolution=resolution,
            )
        )
    return frames


def _normalise_plot(plot: str | Sequence[str] | None) -> list[str]:
    if plot is None or plot == "none":
        return []
    if isinstance(plot, str):
        key = plot.lower()
        if key in {"xyz", "all", "movement"}:
            return ["x", "y", "z"]
        if key not in _PANEL_INDICES:
            raise ValueError(f"plot must be 'x', 'y', 'z', 'xyz', None or a sequence of those (got {plot!r})")
        return [key]
    axes = [str(a).lower() for a in plot]
    bad = [a for a in axes if a not in _PANEL_INDICES]
    if bad:
        raise ValueError(f"unknown plot axes {bad}; use 'x', 'y', 'z'")
    return axes


def _plot_panels(
    traj: Trajectory,
    axes: Sequence[str],
    resolution: tuple[int, int],
    frac: float,
) -> list[np.ndarray]:
    width, height = int(resolution[0]), int(resolution[1])
    panel = (width // 2, height if len(axes) == 1 else max(1, (2 * height) // len(axes)))
    return [trajectory_to_image(traj, size=panel, axis=axis, progress=frac, style="dark") for axis in axes]


# ----------------------------------------------------------------------
# Video
# ----------------------------------------------------------------------


def render_video(
    result: Any,
    path: str | Path,
    *,
    camera=None,
    cameras=None,
    views: str | Sequence[Any] | None = None,
    prefix: str | None = None,
    plot: str | Sequence[str] | None = None,
    overlay: Overlay | str = "freq/amp",
    resolution: tuple[int, int] = (480, 360),
    fps: int = 25,
    max_frames: int | None = None,
    lookat: Sequence[float] | None = None,
    zoom: float = 1.0,
    excitation: Mapping[str, Any] | None = None,
    overlay_extra: str | None = None,
    save_camera_json: bool = True,
) -> Path:
    """Render a recorded run to ``.mp4`` (ffmpeg) or ``.gif`` and return the path.

    Parameters
    ----------
    result : SimulationResult
        Must come from a run with ``record=True``.
    path : str | Path
        ``.mp4``/``.mov``/``.mkv`` stream through ffmpeg; ``.gif`` is written
        with Pillow.  Without ffmpeg a ``RuntimeWarning`` is raised and frames
        are kept in memory (no file).
    camera, cameras, views, prefix, lookat, zoom :
        See the module docstring; ``views="all_sides"`` concatenates the four
        built-in look-down views.
    plot : str | sequence | None
        Movement panels beside the video (``"x"``, ``"y"``, ``"z"``, ``"xyz"``
        or a list).  The panel width matches the video, so output doubles in
        width when graphs are on.
    overlay : str | callable | None
        Default ``"freq/amp"`` draws ``freq`` and ``amp`` bottom-left on every
        frame.
    resolution : (width, height)
        Size of one camera tile (output is ``n_cols * width`` wide).
    fps : int
        Playback rate (frames are sampled tip-to-tip; simulation time per
        encoded frame is ``duration / n_frames``).
    max_frames : int | None
        Thin the trajectory to at most this many frames (keeps the last).
    excitation : mapping | None
        ``{"freq_hz", "amp_x", "amp_z", "vibrate"}`` for the default overlay.
        Defaults to the engine constants (``VIB_FREQ``, ``VIB_AMP_X``,
        ``VIB_AMP_Z``).
    save_camera_json : bool
        Save every resolved camera to ``<stem>.camera.json`` next to the video.

    Returns
    -------
    Path
        The written file (``.mp4``) or the path that would have been written
        when ffmpeg is missing.
    """
    if excitation is None:
        from ..constants import VIB_AMP_X, VIB_AMP_Z, VIB_FREQ

        excitation = {"freq_hz": VIB_FREQ, "amp_x": VIB_AMP_X, "amp_z": VIB_AMP_Z, "vibrate": True}

    # Resolve cameras once here so we can persist the exact view used.
    traj = _trajectory_of(result)
    cam_input = views if views is not None else (cameras if cameras is not None else camera)
    if isinstance(cam_input, str) and cam_input in ALL_SIDES_ALIASES:
        cam_input = cinematic_views(prefix)
    bounds = _scene_bounds(result)
    cam_lookat = (
        lookat
        if lookat is not None
        else (((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2, 0.0) if bounds is not None else None)
    )

    out = Path(path)
    if save_camera_json:
        cam_json = out.with_suffix(".camera.json")
        try:
            specs = resolve_cameras(cam_input, prefix=prefix, zoom=zoom, lookat=cam_lookat)
            cam_json.parent.mkdir(parents=True, exist_ok=True)
            cam_json.write_text(json.dumps({"cameras": [s.as_dict() for s in specs]}, indent=2))
        except ValueError:
            # Camera errors surface from render_frames with a clearer message.
            warnings.warn(f"could not resolve cameras for {cam_json}; skipping camera sidecar", stacklevel=2)

    frames = render_frames(
        result,
        camera=camera,
        cameras=cameras,
        views=views,
        prefix=prefix,
        plot=plot,
        overlay=overlay,
        resolution=resolution,
        max_frames=max_frames,
        lookat=cam_lookat,
        zoom=zoom,
        excitation=excitation,
        overlay_extra=overlay_extra,
    )
    if not frames:
        raise ValueError("no frames to render")
    del traj
    h, w = frames[0].shape[:2]
    writer = VideoWriter(out, fps=int(fps), size=(w, h))
    try:
        if writer.encoder == "memory" and out.suffix.lower() != ".gif" and ffmpeg_available() is None:
            warnings.warn(
                f"ffmpeg is unavailable: {len(frames)} frames were rendered but not written to {out}. "
                "Install ffmpeg (or use a .gif path / imageio-ffmpeg).",
                RuntimeWarning,
                stacklevel=2,
            )
        for frame in frames:
            writer.write(frame)
    finally:
        written = writer.close()
    return written if written is not None else out
