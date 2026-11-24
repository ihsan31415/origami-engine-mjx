from __future__ import annotations

"""Camera presets and name/coordinate resolution (pure NumPy, no MuJoCo).

A camera is described by MuJoCo's free-camera spherical parameters
(``lookat``/``distance``/``azimuth``/``elevation``) so it can be handed to
``mujoco.MjvCamera`` unchanged.  Two ways to get one:

* a **preset name** — ``"top"``, ``"side"``, ``"side_left"``,
  ``"top_side_left"``, ``"iso_top"``, … (see :data:`CAMERA_PRESETS`),
  optionally namespaced with a *prefix* (``prefix="run7"`` accepts
  ``"run7_top"``);
* **manual coordinates** — explicit azimuth/elevation/distance/zoom, either a
  mapping or a tuple ``(azimuth, elevation[, distance[, zoom]])``.

Angles are in **degrees**.  ``azimuth`` rotates around the vertical axis
(0 = looking from +x toward the origin, 90 = from +y), ``elevation`` is the
camera height above the horizon (positive looks down), and ``distance`` is the
camera radius.  ``zoom`` multiplies the preset distance (smaller = closer).

The ``"default"`` preset reproduces MuJoCo's own free camera, so an old
``OrigamiEnv.render`` view and ``camera="default"`` look the same.

This module only resolves *names to numbers*; :class:`CameraSpec` and the
rendering entry points live in :mod:`origami_engine_mjx.rendering.render`.
"""

import types
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

__all__ = [
    "CAMERA_PRESETS",
    "available_cameras",
    "camera_presets",
    "cinematic_views",
    "resolve_camera",
    "resolve_cameras",
]

#: Reference sheet centre used when no bounds/lookat are supplied.
_CENTRE = (0.5, 0.5, 0.0)

#: Default camera distance for a unit-square sheet plus margin.
_DEFAULT_DISTANCE = 1.55

#: Name -> ``(azimuth, elevation, distance, zoom)`` in degrees/metres.
CAMERA_PRESETS: dict[str, tuple[float, float, float, float]] = {
    "default": (90.0, -45.0, 1.45, 1.0),
    "front": (90.0, -15.0, _DEFAULT_DISTANCE, 1.0),
    "back": (270.0, -15.0, _DEFAULT_DISTANCE, 1.0),
    "left": (180.0, -15.0, _DEFAULT_DISTANCE, 1.0),
    "right": (0.0, -15.0, _DEFAULT_DISTANCE, 1.0),
    "side": (0.0, -10.0, _DEFAULT_DISTANCE, 1.0),
    "side_left": (180.0, -10.0, _DEFAULT_DISTANCE, 1.0),
    "side_right": (0.0, -10.0, _DEFAULT_DISTANCE, 1.0),
    "top": (90.0, 85.0, 1.35, 1.0),
    "top_side": (0.0, 40.0, 1.35, 1.0),
    "top_side_left": (180.0, 40.0, 1.35, 1.0),
    "top_side_right": (0.0, 40.0, 1.35, 1.0),
    "top_side_front": (90.0, 40.0, 1.35, 1.0),
    "top_side_back": (270.0, 40.0, 1.35, 1.0),
    "top_left": (180.0, 55.0, 1.35, 1.0),
    "top_right": (0.0, 55.0, 1.35, 1.0),
    "top_front": (90.0, 55.0, 1.35, 1.0),
    "top_back": (270.0, 55.0, 1.35, 1.0),
    "iso": (135.0, -30.0, 1.6, 1.0),
    "iso_left": (225.0, -30.0, 1.6, 1.0),
    "iso_side": (135.0, -15.0, 1.6, 1.0),
    "iso_side_left": (225.0, -15.0, 1.6, 1.0),
    "iso_top": (135.0, 30.0, 1.5, 1.0),
    "iso_top_left": (225.0, 30.0, 1.5, 1.0),
    "iso_top_side": (135.0, 15.0, 1.5, 1.0),
    "iso_top_side_left": (225.0, 15.0, 1.5, 1.0),
}

#: The four views concatenated by ``views="all_sides"``.
_CINEMATIC = ("iso_top", "iso_top_side", "top", "iso_top_left")

#: Values of ``views=`` that expand to :func:`cinematic_views`.
ALL_SIDES_ALIASES = frozenset({"all_sides", "all", "four", "cinematic"})


def _normalise_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def _preset_name(name: str) -> str:
    key = _normalise_name(name)
    if key not in CAMERA_PRESETS:
        near = [k for k in sorted(CAMERA_PRESETS) if key and (key in k or k.startswith(key[:3]))][:6]
        hint = f"; did you mean {near}?" if near else ""
        raise ValueError(f"unknown camera preset {name!r}{hint}. Available: {sorted(CAMERA_PRESETS)}")
    return key


def _lookat_from(bounds: Sequence[float] | None) -> tuple[float, float, float]:
    if bounds is None:
        return _CENTRE
    arr = np.asarray(bounds, dtype=float).reshape(-1)
    if arr.size != 4:
        raise ValueError("bounds must be (xmin, ymin, xmax, ymax)")
    return (float((arr[0] + arr[2]) / 2), float((arr[1] + arr[3]) / 2), 0.0)


def _spec(**kwargs: Any) -> Any:
    """Build a ``CameraSpec`` (imported lazily to avoid a circular import)."""
    from .render import CameraSpec

    return CameraSpec(**kwargs)


def resolve_camera(
    camera: str | Mapping[str, Any] | Sequence[float] | Any | None = None,
    *,
    prefix: str | None = None,
    zoom: float = 1.0,
    lookat: Sequence[float] | None = None,
    bounds: Sequence[float] | None = None,
):
    """Turn a preset name, manual coordinates or a spec into a camera spec.

    Parameters
    ----------
    camera : str | mapping | sequence | CameraSpec | None
        ``None`` / ``"default"`` (MuJoCo's default view), a preset name from
        :data:`CAMERA_PRESETS`, a mapping with any of ``azimuth``,
        ``elevation``, ``distance``, ``zoom`` and ``lookat``, or a tuple
        ``(azimuth, elevation[, distance[, zoom]])``.
    prefix : str | None
        Preset namespace.  ``prefix="run7"`` resolves ``"run7_top"`` (or bare
        ``"run7"``, meaning ``"run7_default"``), so one run can keep its own
        named camera family.
    zoom : float
        Multiplies the preset distance (``< 1`` moves closer).
    lookat : sequence | None
        Explicit look-at point ``(x, y, z)``; overrides ``bounds``.
    bounds : sequence | None
        ``(xmin, ymin, xmax, ymax)`` sheet extent; the camera looks at its centre.

    Examples
    --------
    >>> resolve_camera("top").azimuth
    90.0
    >>> resolve_camera((0.0, -20.0, 1.2)).name
    'manual'
    >>> resolve_camera("cam_top", prefix="cam").name
    'top'
    """
    from .render import CameraSpec

    if zoom <= 0:
        raise ValueError("zoom must be positive")
    centre = tuple(float(v) for v in lookat) if lookat is not None else _lookat_from(bounds)
    if len(centre) != 3:
        raise ValueError("lookat must have three components")

    if isinstance(camera, CameraSpec):
        return camera

    base_name = "default"

    def preset(name: str) -> Any:
        az, el, dist, z = CAMERA_PRESETS[name]
        return _spec(name=name, lookat=centre, distance=dist * z * zoom, azimuth=az % 360.0, elevation=el)

    if camera is None:
        return preset(base_name)

    if isinstance(camera, str):
        wanted = _normalise_name(camera)
        if prefix:
            pre = _normalise_name(prefix)
            if wanted == pre:
                wanted = "default"
            elif wanted.startswith(f"{pre}_"):
                wanted = wanted[len(pre) + 1 :]
        return preset(_preset_name(wanted))

    return _parse_manual(camera, preset(base_name), zoom=zoom)


def _parse_manual(value: Mapping[str, Any] | Sequence[float], base: Any, *, zoom: float) -> Any:
    from .render import CameraSpec

    if isinstance(value, Mapping):
        looks = value.get("lookat", base.lookat)
        lookat = tuple(float(v) for v in looks)
        if len(lookat) != 3:
            raise ValueError("lookat must have three components")
        spec = CameraSpec(
            name=str(value.get("name", "manual")),
            lookat=lookat,
            distance=float(value.get("distance", base.distance)),
            azimuth=float(value.get("azimuth", base.azimuth)) % 360.0,
            elevation=float(value.get("elevation", base.elevation)),
        )
        factor = float(value.get("zoom", 1.0))
    else:
        if isinstance(value, (str, bytes)):
            raise TypeError("camera sequence must be numbers, not a string")
        seq = [float(v) for v in value]
        factor = 1.0
        if len(seq) == 2:
            spec = CameraSpec("manual", base.lookat, base.distance, seq[0] % 360.0, seq[1])
        elif len(seq) == 3:
            spec = CameraSpec("manual", base.lookat, seq[2], seq[0] % 360.0, seq[1])
        elif len(seq) == 4:
            spec = CameraSpec("manual", base.lookat, seq[2], seq[0] % 360.0, seq[1])
            factor = seq[3]
        else:
            raise ValueError("manual camera must be a mapping or a tuple of 2, 3 or 4 numbers")
    if spec.distance <= 0 or factor <= 0 or zoom <= 0:
        raise ValueError("camera distance and zoom must be positive")
    return CameraSpec(spec.name, spec.lookat, spec.distance * factor * zoom, spec.azimuth, spec.elevation)


def resolve_cameras(
    cameras: str | Iterable[str | Mapping[str, Any] | Sequence[float]] | Any | None = None,
    *,
    prefix: str | None = None,
    zoom: float = 1.0,
    lookat: Sequence[float] | None = None,
    bounds: Sequence[float] | None = None,
) -> list[Any]:
    """Resolve one camera or an iterable of cameras into a list of specs.

    A string is one camera (not a sequence of characters), ``None`` gives one
    ``"default"`` camera, and ``"all_sides"`` gives the four
    :func:`cinematic_views`.
    """
    from .render import CameraSpec

    if cameras is None or isinstance(cameras, (str, Mapping, CameraSpec)):
        return [resolve_camera(cameras, prefix=prefix, zoom=zoom, lookat=lookat, bounds=bounds)]
    if isinstance(cameras, types.GeneratorType):
        cameras = list(cameras)
    out = [resolve_camera(cam, prefix=prefix, zoom=zoom, lookat=lookat, bounds=bounds) for cam in cameras]
    if not out:
        raise ValueError("cameras must not be empty")
    return out


def available_cameras(prefix: str | None = None) -> list[str]:
    """Return preset names, optionally namespaced with ``prefix``."""
    if not prefix:
        return sorted(CAMERA_PRESETS)
    return [f"{prefix}_{name}" for name in sorted(CAMERA_PRESETS)]


def camera_presets() -> dict[str, tuple[float, float, float, float]]:
    """Return a copy of :data:`CAMERA_PRESETS`."""
    return dict(CAMERA_PRESETS)


def cinematic_views(prefix: str | None = None) -> list[str]:
    """Return the four views concatenated by ``views="all_sides"``.

    They look down on the sheet from four sides: ``iso_top``, ``iso_top_side``,
    ``top`` and ``iso_top_left``.  With ``prefix`` the names are namespaced as
    :func:`resolve_camera` expects.
    """
    if not prefix:
        return list(_CINEMATIC)
    return [f"{prefix}_{name}" for name in _CINEMATIC]
