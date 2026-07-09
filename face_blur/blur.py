"""Detect and redact faces in a stream of BGR video frames.

This build of MediaPipe (0.10.x) ships only the Tasks API, so we use
``FaceDetector`` in VIDEO running mode. Each detected face is redacted in place
with one of two styles:

- ``box``   — a solid fill. The pixels are destroyed, so it is **irreversible**;
  this is the default and the safe choice for anonymizing a subject.
- ``mosaic``— blocky pixelation. Natural-looking but only weak de-identification
  (lossy, yet vulnerable to brute-force matching and ML re-identification).
"""

import time
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceDetector,
    FaceDetectorOptions,
    RunningMode,
)

from .model import ensure_model

BLUR_STYLES = ("box", "mosaic")

# Solid-fill colour for the "box" style (BGR); black.
_BOX_COLOR = (0, 0, 0)


def _padded_bounds(frame, x, y, w, h, pad):
    """Pad a detection box by ``pad`` (fraction of its size) and clamp to frame.

    Returns ``(x0, y0, x1, y1)`` slice indices, or ``None`` if the region is empty.
    """
    fh, fw = frame.shape[:2]
    dx, dy = int(w * pad), int(h * pad)
    x0 = max(0, x - dx)
    y0 = max(0, y - dy)
    x1 = min(fw, x + w + dx)
    y1 = min(fh, y + h + dy)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _solid_box(frame, bounds) -> None:
    """Fill the region with a solid colour (irreversible)."""
    x0, y0, x1, y1 = bounds
    frame[y0:y1, x0:x1] = _BOX_COLOR


def _mosaic_region(frame, bounds, blocks) -> None:
    """Pixelate the region into roughly ``blocks``x``blocks`` cells (weak)."""
    x0, y0, x1, y1 = bounds
    roi = frame[y0:y1, x0:x1]
    rh, rw = roi.shape[:2]
    bx = max(1, min(blocks, rw))
    by = max(1, min(blocks, rh))
    small = cv2.resize(roi, (bx, by), interpolation=cv2.INTER_LINEAR)
    frame[y0:y1, x0:x1] = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)


class FaceBlurrer:
    """Redact faces on a stream of BGR video frames.

    Use as a context manager so the underlying detector is closed cleanly::

        with FaceBlurrer(style="box") as blurrer:
            blurrer.blur(frame_bgr)   # modified in place
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        style: str = "box",
        min_confidence: float = 0.5,
        pad: float = 0.15,
        mosaic_blocks: int = 12,
    ) -> None:
        if style not in BLUR_STYLES:
            raise ValueError(f"style must be one of {BLUR_STYLES}, got {style!r}")
        self.style = style
        self.pad = pad
        self.mosaic_blocks = mosaic_blocks
        self._start = time.monotonic()
        self._last_ts = -1

        model_path = ensure_model(model_path)
        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            min_detection_confidence=min_confidence,
        )
        self._detector: Optional[FaceDetector] = FaceDetector.create_from_options(options)

    def _next_timestamp(self) -> int:
        """A strictly increasing millisecond timestamp for VIDEO mode."""
        ts = int((time.monotonic() - self._start) * 1000)
        ts = max(ts, self._last_ts + 1)
        self._last_ts = ts
        return ts

    def blur(self, frame_bgr: np.ndarray, timestamp_ms: Optional[int] = None) -> np.ndarray:
        """Detect and redact every face in ``frame_bgr``, in place.

        ``timestamp_ms`` must increase monotonically across calls; when omitted a
        monotonic clock is used, so callers need no timestamp bookkeeping.
        """
        if self._detector is None:
            raise RuntimeError("FaceBlurrer is closed.")
        ts = self._next_timestamp() if timestamp_ms is None else timestamp_ms

        rgb = frame_bgr[:, :, ::-1]  # BGR -> RGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._detector.detect_for_video(mp_image, ts)

        for detection in result.detections:
            bb = detection.bounding_box
            bounds = _padded_bounds(
                frame_bgr, bb.origin_x, bb.origin_y, bb.width, bb.height, self.pad
            )
            if bounds is None:
                continue
            if self.style == "box":
                _solid_box(frame_bgr, bounds)
            else:
                _mosaic_region(frame_bgr, bounds, self.mosaic_blocks)
        return frame_bgr

    def close(self) -> None:
        if self._detector is not None:
            self._detector.close()
            self._detector = None

    def __enter__(self) -> FaceBlurrer:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
