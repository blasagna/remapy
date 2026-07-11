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

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceDetector,
    FaceDetectorOptions,
    RunningMode,
)

from .model import ensure_model
from .redact import BLUR_STYLES, padded_bounds, redact_region, validate_style


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
        validate_style(style)
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

    def blur(
        self,
        frame_bgr: np.ndarray,
        pose_result: object = None,
        timestamp_ms: Optional[int] = None,
    ) -> np.ndarray:
        """Detect and redact every face in ``frame_bgr``, in place.

        ``pose_result`` is accepted and ignored so this backend is a drop-in for
        :class:`face_blur.pose_blur.PoseFaceBlurrer`, which redacts from an
        already-computed pose result instead of running its own detector.

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
            bounds = padded_bounds(
                frame_bgr, bb.origin_x, bb.origin_y, bb.width, bb.height, self.pad
            )
            if bounds is None:
                continue
            redact_region(frame_bgr, bounds, self.style, self.mosaic_blocks)
        return frame_bgr

    def close(self) -> None:
        if self._detector is not None:
            self._detector.close()
            self._detector = None

    def __enter__(self) -> FaceBlurrer:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
