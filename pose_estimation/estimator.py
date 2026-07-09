"""Thin wrapper around MediaPipe's ``PoseLandmarker`` (Tasks API).

This build of MediaPipe (0.10.x) ships only the Tasks API, so we use
``PoseLandmarker`` in VIDEO running mode and feed it frames from the
``video_capture`` library.
"""

from pathlib import Path
from typing import Optional

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmarkerResult,
    PoseLandmarksConnections,
    RunningMode,
)

from .model import ensure_model

# Skeleton topology as (start, end) landmark-index pairs, for drawing bones.
POSE_CONNECTIONS: list[tuple[int, int]] = [
    (c.start, c.end) for c in PoseLandmarksConnections.POSE_LANDMARKS
]


class PoseEstimator:
    """Detect body pose landmarks on a stream of BGR video frames.

    Use as a context manager so the underlying landmarker is closed cleanly::

        with PoseEstimator() as pose:
            result = pose.detect(frame_bgr, timestamp_ms)
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        num_poses: int = 1,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        model_path = ensure_model(model_path)
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            num_poses=num_poses,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker: Optional[PoseLandmarker] = PoseLandmarker.create_from_options(
            options
        )

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> PoseLandmarkerResult:
        """Run pose detection on one BGR frame.

        ``timestamp_ms`` must increase monotonically across calls (VIDEO mode
        requirement).
        """
        if self._landmarker is None:
            raise RuntimeError("PoseEstimator is closed.")
        rgb = frame_bgr[:, :, ::-1]  # BGR -> RGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        return self._landmarker.detect_for_video(mp_image, timestamp_ms)

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> PoseEstimator:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
