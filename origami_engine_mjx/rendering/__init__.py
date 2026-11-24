from __future__ import annotations

"""Rendering: cameras, recorded trajectories, video, movement plots and overlays.

Everything here is **additive**: the classic ``OrigamiEnv.render`` contract is
unchanged.  To render, record a run first::

    result = engine.run(creases, duration=2.0, record=True)
    render_video(result, "fold.mp4")                          # one camera
    render_video(result, "four.mp4", views="all_sides")       # 4 views in one
    render_video(result, "graphs.mp4", plot="xyz")            # graphs beside
    render_video(result, "manual.mp4", camera=(40, -20, 1.3))  # manual POV

Every frame carries the current ``freq``/``amp`` in its bottom-left corner.

Submodules
----------
cameras
    Preset names (``top``, ``side_left``, ``top_side_left`` …) and manual
    azimuth/elevation/distance coordinates.
trajectory
    :class:`Trajectory` — per-frame ``qpos``/``qvel``/centre-of-mass with
    convenience rates and subsampling.
plots
    Matplotlib movement plots (one panel per axis) and generation-history plots.
compose
    NumPy/Pillow frame helpers (concatenation, letterboxing, text overlay).
video
    ffmpeg-backed :class:`VideoWriter` (MP4/h264) with a GIF fallback.
"""

from .cameras import (
    ALL_SIDES_ALIASES,
    CAMERA_PRESETS,
    available_cameras,
    camera_presets,
    cinematic_views,
    resolve_camera,
    resolve_cameras,
)
from .compose import (
    caption,
    draw_text,
    grid,
    hconcat,
    overlay_text,
    resize_nearest,
    split_channels,
    vconcat,
    with_letterbox,
)
from .plots import (
    axes_figure,
    history_to_image,
    movement_figure,
    plot_axes,
    plot_history,
    plot_trajectory,
    trajectory_to_image,
)
from .render import (
    CameraSpec,
    default_overlay,
    layout_frames,
    load_camera,
    render_frames,
    render_video,
    save_camera,
    trajectory_frames,
)
from .trajectory import Trajectory
from .video import VideoWriter, ffmpeg_available, write_gif

__all__ = [
    "ALL_SIDES_ALIASES",
    "CAMERA_PRESETS",
    "CameraSpec",
    "Trajectory",
    "VideoWriter",
    "available_cameras",
    "axes_figure",
    "camera_presets",
    "caption",
    "cinematic_views",
    "default_overlay",
    "draw_text",
    "ffmpeg_available",
    "grid",
    "hconcat",
    "history_to_image",
    "layout_frames",
    "load_camera",
    "movement_figure",
    "overlay_text",
    "plot_axes",
    "plot_history",
    "plot_trajectory",
    "render_frames",
    "render_video",
    "resize_nearest",
    "resolve_camera",
    "resolve_cameras",
    "save_camera",
    "split_channels",
    "trajectory_frames",
    "trajectory_to_image",
    "vconcat",
    "with_letterbox",
    "write_gif",
]
