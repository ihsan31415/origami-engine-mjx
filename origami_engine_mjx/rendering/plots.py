from __future__ import annotations

"""Movement / trajectory plots beside the simulation (matplotlib only).

These helpers answer "what did the paper actually do?" — per-axis centre-of-mass
movement, hinge actuation, velocity — with one panel per axis.  They do not need
MuJoCo, JAX or a GL context, so they work anywhere ``matplotlib`` does.

All functions accept either a :class:`~origami_engine_mjx.rendering.Trajectory`,
a recorded ``SimulationResult`` (``record=True``) or the raw trajectory dict.
"""

from typing import Any

import numpy as np

from .trajectory import Trajectory

__all__ = [
    "AXIS_NAMES",
    "axes_figure",
    "movement_figure",
    "plot_axes",
    "plot_history",
    "plot_trajectory",
    "trajectory_to_image",
]

AXIS_NAMES = ("x", "y", "z")


def _as_trajectory(obj: Any) -> Trajectory:
    if isinstance(obj, Trajectory):
        return obj
    if isinstance(obj, dict):
        return Trajectory.from_dict(obj)
    return Trajectory.from_result(obj)


def _relative(series: np.ndarray) -> np.ndarray:
    return series - series[0]


def plot_axes(
    trajectory: Any,
    ax=None,
    *,
    axis: str = "x",
    relative: bool = True,
    show_rate: bool = True,
    label: str | None = None,
):
    """Plot one axis of paper centre-of-mass movement over time.

    ``axis`` is ``"x"``, ``"y"`` or ``"z"``.  ``z`` uses the hinge angles up to
    the paper (the sheet itself only translates through the table joints), so
    for ``z`` the curve is the mean absolute hinge rotation in degrees.  Returns
    the matplotlib ``Axes``.
    """
    import matplotlib.pyplot as plt

    traj = _as_trajectory(trajectory)
    ax = plt.gca() if ax is None else ax
    t = traj.time
    axis = axis.lower()
    if axis in {"x", "y"}:
        k = 0 if axis == "x" else 1
        com = traj.center_of_mass(relative=relative)[:, k]
        ax.plot(t, com, color="#1f77b4", lw=1.4, label=f"centroid {axis}")
        if show_rate:
            ax2 = ax.twinx()
            ax2.plot(t, traj.velocity(traj.center_of_mass())[:, k], color="#ff7f0e", lw=0.9, ls="--", alpha=0.8)
            ax2.set_ylabel(f"d{axis}/dt [m/s]", color="#ff7f0e", fontsize=8)
            ax2.tick_params(labelsize=7, colors="#ff7f0e")
    elif axis == "z":
        angles = np.degrees(traj.hinge_angles())
        curve = np.mean(np.abs(angles), axis=1) if angles.size else np.zeros_like(t)
        ax.plot(t, _relative(curve), color="#9467bd", lw=1.4, label="mean |hinge| [deg]")
    else:
        raise ValueError("axis must be 'x', 'y' or 'z'")
    ax.set_xlabel("time [s]", fontsize=8)
    ax.set_ylabel(label or f"{axis} movement", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="best")
    return ax


def axes_figure(
    trajectory: Any,
    *,
    figsize: tuple[float, float] = (7.0, 6.0),
    relative: bool = True,
):
    """Figure with one movement panel per axis (x, y, z), stacked vertically."""
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 1, figsize=figsize, sharex=True, constrained_layout=True)
    for ax, axis in zip(axs, ("x", "y", "z")):
        plot_axes(trajectory, ax=ax, axis=axis, relative=relative)
    axs[-1].set_xlabel("time [s]", fontsize=8)
    return fig


def movement_figure(
    trajectory: Any,
    *,
    axis: str = "x",
    figsize: tuple[float, float] = (4.2, 3.4),
    title: str | None = None,
    relative: bool = True,
):
    """Single-axis movement figure (used beside a video frame)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    plot_axes(trajectory, ax=ax, axis=axis, relative=relative)
    if title:
        ax.set_title(title, fontsize=9)
    return fig


def plot_trajectory(
    trajectory: Any,
    *,
    axis: str = "all",
    path: str | None = None,
    figsize: tuple[float, float] = (7.0, 6.0),
    relative: bool = True,
    dpi: int = 110,
):
    """Plot centre-of-mass movement and optionally save it.

    ``axis="all"`` gives the three-panel figure from :func:`axes_figure`; a
    single axis gives one panel.  ``path`` (PNG/PDF/SVG) saves the figure.
    Returns ``(figure, axes)``.
    """
    fig = (
        axes_figure(trajectory, figsize=figsize, relative=relative)
        if axis == "all"
        else movement_figure(trajectory, axis=axis, figsize=figsize, relative=relative)
    )
    fig.set_dpi(dpi)
    if path is not None:
        fig.savefig(path, dpi=dpi)
    return fig, fig.axes


def trajectory_to_image(
    trajectory: Any,
    *,
    size: tuple[int, int] = (480, 360),
    axis: str = "x",
    relative: bool = True,
    dpi: int = 100,
    progress: float | None = None,
    style: str | None = None,
) -> np.ndarray:
    """Render a movement plot to an RGB ``uint8`` array of ``size=(width, height)``.

    When ``progress`` (in ``[0, 1]``) is given, the curve is drawn up to that
    fraction of the run and the remaining span stays in the axis limits — that
    is what makes per-frame panels in a video line up instead of rescaling on
    every frame.  ``style="dark"`` uses a dark background to match video frames.
    """
    import matplotlib

    backend_was = matplotlib.get_backend()
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    traj = _as_trajectory(trajectory)
    width, height = int(size[0]), int(size[1])
    dark = style == "dark"
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="#101418" if dark else "white")
    ax = fig.add_axes((0.16, 0.18, 0.80, 0.74))
    if dark:
        ax.set_facecolor("#101418")
        for spine in ax.spines.values():
            spine.set_color("#88909a")
        ax.tick_params(colors="#c8d0d8", labelsize=7)
        ax.xaxis.label.set_color("#c8d0d8")
        ax.yaxis.label.set_color("#c8d0d8")
    else:
        ax.tick_params(labelsize=7)

    if axis in {"x", "y"}:
        k = 0 if axis == "x" else 1
        series = traj.center_of_mass(relative=relative)[:, k]
        colour = "#4fa3ff" if dark else "#1f77b4"
    elif axis == "z":
        angles = np.degrees(traj.hinge_angles())
        series = _relative(np.mean(np.abs(angles), axis=1)) if angles.size else np.zeros(len(traj))
        colour = "#c08cff" if dark else "#9467bd"
    else:
        raise ValueError("axis must be 'x', 'y' or 'z'")

    n = len(traj)
    cut = n if progress is None else max(2, round(np.clip(progress, 0.0, 1.0) * (n - 1)) + 1)
    cut = min(cut, n)
    t = traj.time
    ax.plot(t, series, color=colour, lw=1.5)
    ax.plot(t[:cut], series[:cut], color="#ffd166" if dark else "#d62728", lw=1.8)
    if cut < n:
        ax.plot(t[cut - 1], series[cut - 1], "o", color="#ffd166" if dark else "#d62728", ms=4)
    pad = 0.05 * (np.ptp(series) if n > 1 and np.ptp(series) > 0 else 1.0)
    ax.set_xlim(float(t[0]), float(t[-1]) if t[-1] > t[0] else float(t[0]) + 1.0)
    ax.set_ylim(float(series.min()) - pad, float(series.max()) + pad)
    unit = {"x": "x [m]", "y": "y [m]", "z": "mean |hinge|"}[axis]
    ax.set_xlabel("time [s]", fontsize=8)
    ax.set_ylabel(unit, fontsize=8)
    ax.set_title(
        {"x": "Movement X", "y": "Movement Y", "z": "Hinge actuation"}[axis],
        fontsize=9,
        color="#e8eef4" if dark else "black",
    )
    ax.grid(alpha=0.25, color="#3a444f" if dark else "#cccccc")

    fig.canvas.draw()
    canvas: Any = fig.canvas
    buf = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    matplotlib.use(backend_was)
    if buf.shape[:2] != (height, width):
        from .compose import resize_nearest

        buf = resize_nearest(buf, (width, height))
    return buf


def plot_history(
    history: Any,
    *,
    path: str | None = None,
    figsize: tuple[float, float] = (7.0, 4.0),
    dpi: int = 110,
    maximising: bool | None = None,
    title: str | None = None,
):
    """Plot best/mean fitness per generation, with the running global best.

    ``history`` is a :class:`~origami_engine_mjx.integrations.GenerationLogger`
    itself, its ``.history`` list of
    :class:`~origami_engine_mjx.integrations.GenerationSummary` objects, or the
    dict rows returned by
    :func:`~origami_engine_mjx.integrations.load_generation_history`.
    ``maximising`` only labels the y axis ("higher is better"/"lower is
    better") and defaults to the logger's setting when one is passed.
    Returns ``(figure, axes)``.
    """
    import matplotlib.pyplot as plt

    if hasattr(history, "history"):
        maximising = getattr(history, "maximising", True) if maximising is None else maximising
        rows = list(history.history)
        rows = [r.as_dict() if hasattr(r, "as_dict") else r for r in rows]
    else:
        rows = list(history)
        rows = [r.as_dict() if hasattr(r, "as_dict") else dict(r) for r in rows]
    if not rows:
        raise ValueError("history is empty")
    gens = np.array([int(r["generation"]) for r in rows])
    best = np.array([float(r["best_fitness"]) for r in rows])
    mean = np.array([float(r["mean_fitness"]) for r in rows])
    gbest = np.array([float(r["global_best_fitness"]) for r in rows])

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)
    band = ax.fill_between(gens, best, mean, color="#9ecae1", alpha=0.35, label="best..mean")
    del band
    ax.plot(gens, best, color="#1f77b4", marker="o", ms=3, lw=1.4, label="generation best")
    ax.plot(gens, mean, color="#7f7f7f", lw=1.0, ls="--", label="generation mean")
    ax.plot(gens, gbest, color="#d62728", lw=1.6, label="global best")
    label = "higher is better" if (maximising if maximising is not None else True) else "lower is better"
    ax.set_xlabel("generation")
    ax.set_ylabel(f"fitness ({label})")
    ax.set_title(title or f"{len(rows)} generations", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    if path is not None:
        fig.savefig(path, dpi=dpi)
    return fig, ax


def history_to_image(history: Any, *, size: tuple[int, int] = (480, 270), dpi: int = 100) -> np.ndarray:
    """Render :func:`plot_history` to an RGB ``uint8`` array (for notebooks/video intros)."""
    import matplotlib

    backend_was = matplotlib.get_backend()
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, _ax = plot_history(history, figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi)
    fig.canvas.draw()
    canvas: Any = fig.canvas
    buf = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    matplotlib.use(backend_was)
    return buf
