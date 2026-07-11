"""Redact faces with pose keypoints, falling back to the face detector.

Combines the two backends to get the best of each: the pose-keypoint redaction is
reliable wherever a body is tracked (odd angles, profile, small faces), while the
standalone ``FaceDetector`` still covers close-up, face-only framing where the
pose model may not fire. Each frame picks the pose path when a pose is present and
the detector otherwise.
"""

from pathlib import Path

import numpy as np

from .blur import FaceBlurrer
from .pose_blur import PoseFaceBlurrer


class HybridFaceBlurrer:
    """Pose-keypoint redaction with a ``FaceDetector`` fallback.

    Interchangeable with :class:`face_blur.blur.FaceBlurrer` and
    :class:`face_blur.pose_blur.PoseFaceBlurrer` — same
    ``blur(frame, pose_result) / close()`` context-manager surface. When
    ``pose_result`` carries a detected pose the keypoint backend redacts the head;
    otherwise the frame is handed to the detector backend.
    """

    def __init__(
        self,
        style: str = "box",
        model_path: Path | str | None = None,
    ) -> None:
        self._pose = PoseFaceBlurrer(style=style)
        self._detector = FaceBlurrer(model_path=model_path, style=style)

    def blur(self, frame_bgr: np.ndarray, pose_result: object = None) -> np.ndarray:
        """Redact faces in ``frame_bgr`` in place, choosing a backend per frame."""
        if getattr(pose_result, "pose_landmarks", None):
            self._pose.blur(frame_bgr, pose_result)
        else:
            self._detector.blur(frame_bgr)
        return frame_bgr

    def close(self) -> None:
        self._pose.close()
        self._detector.close()

    def __enter__(self) -> "HybridFaceBlurrer":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
