from __future__ import annotations

"""Video encoding: an ``ffmpeg`` pipe with an in-memory fallback.

Encoding is deliberately dependency-light: frames are streamed to an
``ffmpeg`` subprocess when one is on ``PATH`` (MP4/h264 by default), and kept
in memory otherwise so callers can still embed them (notebooks) or write a GIF.
No ``imageio``/``moviepy``/``opencv`` dependency is required.

Install hint: ``pip install imageio-ffmpeg`` (bundles an ffmpeg binary) or use
your system package manager (``apt install ffmpeg``).
"""

import shutil
import subprocess
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["VideoWriter", "ffmpeg_available", "write_gif"]

try:  # pragma: no cover - optional dependency
    from PIL import Image, ImageSequence
except Exception:  # noqa: BLE001 - Pillow ships with matplotlib, but never hard-require it
    Image = None  # type: ignore[assignment]
    ImageSequence = None  # type: ignore[assignment]


def ffmpeg_available() -> str | None:
    """Return the ffmpeg executable path, or ``None`` when it is unavailable.

    Also checks ``imageio_ffmpeg`` when installed, which bundles a binary on
    systems without one.
    """
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:  # pragma: no cover - optional dependency
        import imageio_ffmpeg

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:  # noqa: BLE001
        return None


def _fmt(frame: Any, size: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.shape[:2] != (size[1], size[0]):
        raise ValueError(f"frame has shape {arr.shape[:2]}, expected (height, width) = {(size[1], size[0])}")
    return np.ascontiguousarray(arr)


class VideoWriter:
    """Stream ``(H, W, 3)`` uint8 frames to an ``.mp4``/``.gif`` or keep them in memory.

    Parameters
    ----------
    path :
        Destination file.  ``None`` (or an in-memory writer) keeps frames in
        memory.  ``.mp4``/``.mov``/``.mkv`` stream through ffmpeg; ``.gif`` is
        written after the last frame.
    fps : int
        Frames per second of the encoded video.
    size : (width, height)
        Expected frame size; mismatched frames raise ``ValueError``.

    Notes
    -----
    ``ffmpeg`` is looked up at construction.  When it is missing, frames are
    accumulated in :attr:`frames` and :meth:`close` warns instead of raising,
    so a rendering pipeline still produces something usable (and the caller can
    inspect ``writer.frames``).
    """

    def __init__(self, path: str | Path | None, fps: int, size: tuple[int, int]) -> None:
        if size[0] <= 0 or size[1] <= 0:
            raise ValueError("frame size must be positive")
        self.path = None if path is None else Path(path)
        self.fps = max(1, int(fps))
        self.size = (int(size[0]), int(size[1]))
        self.n_frames = 0
        self.frames: list[np.ndarray] = []
        self.encoder = "memory"
        self._proc: subprocess.Popen | None = None
        if self.path is not None and self.path.suffix.lower() != ".gif":
            exe = ffmpeg_available()
            if exe is None:
                # Keep frames in memory; the caller can still write a GIF.
                warnings.warn(
                    "ffmpeg not found: frames are kept in memory instead of written to "
                    f"{self.path}. Install ffmpeg (or 'pip install imageio-ffmpeg').",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                exe,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{self.size[0]}x{self.size[1]}",
                "-r",
                str(self.fps),
                "-i",
                "-",
                "-an",
                "-vcodec",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-movflags",
                "+faststart",
                str(self.path),
            ]
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            self.encoder = "ffmpeg"
        elif self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.encoder = "gif"
        else:
            self.encoder = "memory"

    # ------------------------------------------------------------------

    def write(self, frame: Any) -> None:
        """Append one frame."""
        arr = _fmt(frame, self.size)
        self.n_frames += 1
        if self._proc is not None and self._proc.stdin is not None:
            self._proc.stdin.write(arr.tobytes())
            return
        self.frames.append(arr)

    def close(self) -> Path | None:
        """Finish encoding and return the written path (or the in-memory path)."""
        if self._proc is not None:
            proc = self._proc
            assert proc.stdin is not None
            proc.stdin.close()
            _, err = proc.communicate()
            self._proc = None
            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {err.decode(errors='replace')[-800:]}")
            return self.path
        if self.path is not None and Image is not None and self.frames:
            duration = max(1, round(1000 / self.fps))
            imgs = [Image.fromarray(f) for f in self.frames]
            imgs[0].save(
                self.path,
                save_all=True,
                append_images=imgs[1:],
                duration=duration,
                loop=0,
                optimize=False,
            )
            self.frames = []
            return self.path
        if self.path is not None and self.frames and Image is None:
            raise RuntimeError("writing a GIF needs Pillow (pip install pillow)")
        return self.path

    def __enter__(self) -> VideoWriter:  # noqa: PYI034 - py3.10 has no typing.Self
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def write_gif(frames: Iterable[Any], path: str | Path, fps: int = 25, size: tuple[int, int] | None = None) -> Path:
    """Write frames to an animated GIF (output width halved if ``size`` is not given)."""
    frame_list = [np.asarray(f) for f in frames]
    if not frame_list:
        raise ValueError("no frames to write")
    if size is None:
        h, w = frame_list[0].shape[:2]
        size = (w, h)
    writer = VideoWriter(path, fps=fps, size=size)
    writer.encoder = "gif"
    for f in frame_list:
        writer.write(f)
    out = writer.close()
    assert out is not None
    return out
