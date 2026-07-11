"""Redact faces using the pose model's face keypoints.

An alternative to :class:`face_blur.blur.FaceBlurrer` for pipelines that already
run :class:`pose_estimation.estimator.PoseEstimator`. Instead of running a
separate ``FaceDetector`` (which can miss faces at odd angles, in profile, or
when small in frame), this backend derives the face region from the pose
landmarks that were *already* detected for the skeleton — so wherever a body is
tracked, its head is redacted, with no second model to run or download.

MediaPipe Pose emits 33 landmarks; the first 11 cover the face:

    0  nose
    1-3  left eye (inner, center, outer)
    4-6  right eye (inner, center, outer)
    7  left ear      8  right ear
    9  mouth (left)  10 mouth (right)

These span roughly eyes-to-mouth vertically and ear-to-ear horizontally. The box
is padded outward — and further above — to cover the forehead, hair and chin that
the keypoints do not mark.
"""

from typing import Optional

import numpy as np

from .redact import padded_bounds, redact_region, validate_style

# Pose landmark indices that lie on the face (see module docstring).
FACE_LANDMARKS = tuple(range(0, 11))


class PoseFaceBlurrer:
    """Redact faces from an already-computed pose result.

    Interchangeable with :class:`face_blur.blur.FaceBlurrer`: it exposes the same
    ``blur(frame, pose_result)`` / ``close()`` context-manager surface, but reads
    the face region from ``pose_result`` rather than detecting faces itself::

        with PoseFaceBlurrer(style="box") as blurrer:
            result = pose.detect(frame, ts)
            blurrer.blur(frame, result)   # frame modified in place

    Parameters
    ----------
    style:
        ``"box"`` (solid, irreversible) or ``"mosaic"`` (weak pixelation).
    pad:
        Fraction of the keypoint box to expand on each side.
    top_pad:
        Larger expansion above the box, to cover the forehead/hair the keypoints
        miss. Defaults to a generous value.
    mosaic_blocks:
        Cell count for the mosaic style.
    min_visibility:
        Ignore face keypoints below this visibility. If too few remain, all face
        keypoints are used as a fallback so a tracked head is still redacted.
    """

    def __init__(
        self,
        style: str = "box",
        pad: float = 0.3,
        top_pad: float = 0.9,
        mosaic_blocks: int = 12,
        min_visibility: float = 0.5,
    ) -> None:
        validate_style(style)
        self.style = style
        self.pad = pad
        self.top_pad = top_pad
        self.mosaic_blocks = mosaic_blocks
        self.min_visibility = min_visibility
        self._closed = False

    def _face_bounds(self, frame_bgr: np.ndarray, pose_landmarks):
        """Pixel-space ``(x0, y0, x1, y1)`` head box for one pose, or ``None``."""
        h, w = frame_bgr.shape[:2]
        face = [pose_landmarks[i] for i in FACE_LANDMARKS if i < len(pose_landmarks)]
        visible = [lm for lm in face if getattr(lm, "visibility", 1.0) >= self.min_visibility]
        # Fall back to all face keypoints if visibility filtering leaves too few.
        pts = visible if len(visible) >= 2 else face
        if len(pts) < 2:
            return None

        xs = [lm.x * w for lm in pts]
        ys = [lm.y * h for lm in pts]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        box_w = max(1, int(x1 - x0))
        box_h = max(1, int(y1 - y0))
        return padded_bounds(
            frame_bgr, int(x0), int(y0), box_w, box_h, self.pad, top_pad=self.top_pad
        )

    def blur(self, frame_bgr: np.ndarray, pose_result: object = None) -> np.ndarray:
        """Redact the head of every detected pose in ``frame_bgr``, in place.

        ``pose_result`` is a ``PoseLandmarkerResult`` (or ``None``). Frames with no
        detected pose are left untouched.
        """
        if self._closed:
            raise RuntimeError("PoseFaceBlurrer is closed.")
        poses = getattr(pose_result, "pose_landmarks", None) or []
        for pose_landmarks in poses:
            bounds = self._face_bounds(frame_bgr, pose_landmarks)
            if bounds is None:
                continue
            redact_region(frame_bgr, bounds, self.style, self.mosaic_blocks)
        return frame_bgr

    def close(self) -> None:
        """Present for parity with :class:`FaceBlurrer`; nothing to release."""
        self._closed = True

    def __enter__(self) -> "PoseFaceBlurrer":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
