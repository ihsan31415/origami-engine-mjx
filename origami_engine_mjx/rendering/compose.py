from __future__ import annotations

"""Small image compositing helpers (NumPy + Pillow, no MuJoCo)."""

from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = [
    "caption",
    "dims",
    "draw_text",
    "grid",
    "hconcat",
    "overlay_text",
    "resize_nearest",
    "split_channels",
    "vconcat",
    "with_letterbox",
]

try:  # pragma: no cover - Pillow comes with matplotlib
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # noqa: BLE001
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    ImageFont = None  # type: ignore[assignment]


#: Pillow 10 moved ``LANCZOS`` under ``Image.Resampling``; accept both.
_LANCZOS: Any = getattr(getattr(Image, "Resampling", None), "LANCZOS", None) if Image is not None else None


def _as_uint8(frame: Any) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    return arr[:, :, :3]


def dims(frame: Any) -> tuple[int, int]:
    """Return ``(width, height)`` of a frame."""
    h, w = np.asarray(frame).shape[:2]
    return int(w), int(h)


def resize_nearest(frame: Any, size: tuple[int, int]) -> np.ndarray:
    """Resize a frame to ``size=(width, height)`` (nearest neighbour, no Pillow)."""
    arr = _as_uint8(frame)
    w, h = int(size[0]), int(size[1])
    if (w, h) == dims(arr):
        return arr.copy()
    ys = np.linspace(0, arr.shape[0] - 1, h).astype(int)
    xs = np.linspace(0, arr.shape[1] - 1, w).astype(int)
    return arr[ys][:, xs]


def hconcat(frames: Sequence[Any], align: str = "top", background: int = 0) -> np.ndarray:
    """Stack frames horizontally, padding shorter ones with ``background``."""
    resized = [_as_uint8(f) for f in frames]
    if not resized:
        raise ValueError("hconcat needs at least one frame")
    h = max(f.shape[0] for f in resized)
    width = sum(f.shape[1] for f in resized)
    out = np.full((h, width, 3), background, dtype=np.uint8)
    x = 0
    for f in resized:
        if align == "center":
            y = (h - f.shape[0]) // 2
        elif align == "bottom":
            y = h - f.shape[0]
        else:
            y = 0
        out[y : y + f.shape[0], x : x + f.shape[1]] = f
        x += f.shape[1]
    return out


def vconcat(frames: Sequence[Any], align: str = "left", background: int = 0) -> np.ndarray:
    """Stack frames vertically, padding narrower ones with ``background``."""
    resized = [_as_uint8(f) for f in frames]
    if not resized:
        raise ValueError("vconcat needs at least one frame")
    w = max(f.shape[1] for f in resized)
    height = sum(f.shape[0] for f in resized)
    out = np.full((height, w, 3), background, dtype=np.uint8)
    y = 0
    for f in resized:
        if align == "center":
            x = (w - f.shape[1]) // 2
        elif align == "right":
            x = w - f.shape[1]
        else:
            x = 0
        out[y : y + f.shape[0], x : x + f.shape[1]] = f
        y += f.shape[0]
    return out


def grid(frames: Sequence[Any], ncols: int = 2, background: int = 0) -> np.ndarray:
    """Arrange frames in a grid with ``ncols`` columns."""
    rows = [hconcat(list(frames[i : i + ncols]), background=background) for i in range(0, len(frames), ncols)]
    return vconcat(rows, background=background)


def with_letterbox(frame: Any, size: tuple[int, int], background: int = 0) -> np.ndarray:
    """Scale ``frame`` to fit inside ``size`` and centre it on a ``background`` canvas."""
    arr = _as_uint8(frame)
    w, h = int(size[0]), int(size[1])
    scale = min(w / arr.shape[1], h / arr.shape[0])
    new_w = max(1, round(arr.shape[1] * scale))
    new_h = max(1, round(arr.shape[0] * scale))
    img = (
        Image.fromarray(arr).resize((new_w, new_h), _LANCZOS)
        if Image is not None
        else _resize_lanczos(arr, (new_w, new_h))
    )
    out = np.full((h, w, 3), background, dtype=np.uint8)
    y0 = (h - new_h) // 2
    x0 = (w - new_w) // 2
    out[y0 : y0 + new_h, x0 : x0 + new_w] = img
    return out


def _resize_lanczos(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Bilinear fallback used only when Pillow is unavailable."""
    w, h = size
    ys = np.linspace(0, arr.shape[0] - 1, h)
    xs = np.linspace(0, arr.shape[1] - 1, w)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, arr.shape[0] - 1)
    x1 = np.minimum(x0 + 1, arr.shape[1] - 1)
    wy = (ys - y0)[:, None, None]
    wx = (xs - x0)[None, :, None]
    top = arr[y0][:, x0] * (1 - wx) + arr[y0][:, x1] * wx
    bottom = arr[y1][:, x0] * (1 - wx) + arr[y1][:, x1] * wx
    return (top * (1 - wy) + bottom * wy).astype(np.uint8)


def split_channels(frame: Any) -> list[np.ndarray]:
    """Return three single-channel frames ``[R, G, B]`` stacked as grey RGB."""
    arr = _as_uint8(frame)
    return [np.repeat(arr[:, :, i : i + 1], 3, axis=2) for i in range(3)]


def _load_font(size: int) -> Any:
    if ImageFont is None:  # pragma: no cover - Pillow is a matplotlib dependency
        return None
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def draw_text(
    frame: Any,
    text: str,
    xy: tuple[int, int] = (4, 4),
    *,
    colour: Sequence[int] = (255, 255, 255),
    font_size: int = 11,
    anchor: str = "top-left",
    background: bool = True,
    pad: int = 0,
) -> np.ndarray:
    """Draw ``text`` on a copy of ``frame`` (``xy`` in pixels).

    ``anchor`` is one of ``top-left``, ``bottom-left``, ``top-right`` or
    ``bottom-right``.  When Pillow is unavailable the frame is returned
    unchanged.  A thin translucent black backing keeps the text readable on
    bright paper.
    """
    out = _as_uint8(frame).copy()
    if Image is None:  # pragma: no cover
        return out
    img = Image.fromarray(out)
    draw = ImageDraw.Draw(img, "RGBA")
    font = _load_font(int(font_size))
    h, w = out.shape[:2]
    probe = draw.textbbox((0, 0), text, font=font)
    tw, th = probe[2] - probe[0], probe[3] - probe[1]
    x, y = int(xy[0]), int(xy[1])
    if anchor.endswith("right"):
        x = w - tw - x
    if anchor.startswith("bottom"):
        y = h - th - y
    if background:
        pad2 = 2 + int(pad)
        draw.rectangle([x - pad2, y - pad2, x + tw + pad2, y + th + pad2], fill=(0, 0, 0, 110))
    draw.text((x, y), text, fill=tuple(int(c) for c in colour), font=font)
    return np.asarray(img)


def overlay_text(
    frame: Any,
    text: str,
    xy: tuple[int, int] = (4, 4),
    **kwargs: Any,
) -> np.ndarray:
    """Alias for :func:`draw_text` that reads naturally in render code."""
    return draw_text(frame, text, xy, **kwargs)


def caption(frame: Any, text: str, *, font_size: int = 11) -> np.ndarray:
    """Draw ``text`` in the bottom-left corner of a copy of ``frame``."""
    return draw_text(frame, text, (4, 4), anchor="bottom-left", font_size=font_size)
