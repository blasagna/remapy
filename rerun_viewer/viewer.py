"""Rerun logging helpers for pose estimation.

Given raw frames and MediaPipe ``PoseLandmarkerResult`` objects, this logs to the
Rerun viewer:

- the video frame (``video/image``) with the 2D skeleton overlaid,
- a 3D skeleton reconstructed from the metric *world* landmarks (``pose3d``),
- scalar time series (line plots) for frame rate and joint angles (``metrics/*``).

Uses the rerun-sdk API (NOT the unrelated ``rerun`` file-watcher package).
"""

from __future__ import annotations

import numpy as np
import rerun as rr

from pose_estimation.angles import joint_angles
from pose_estimation.estimator import POSE_CONNECTIONS

# Timeline names shown on the Rerun time axis.
FRAME_TIMELINE = "frame"
TIME_TIMELINE = "time"

_POINT_COLOR = (0, 200, 255)
_BONE_COLOR = (0, 255, 0)


def _world_xyz(landmarks) -> np.ndarray:
    """World landmarks -> (N, 3) array, remapped so the person stands upright.

    MediaPipe world coords are x-right, y-down, z-toward-camera (meters, origin at
    the hips). Negating y/z gives an upright, forward-facing 3D view in Rerun.
    """
    return np.array([[lm.x, -lm.y, -lm.z] for lm in landmarks], dtype=np.float32)


class PoseRerunLogger:
    """Owns the Rerun recording stream and logs per-frame pose data."""

    def __init__(
        self,
        application_id: str = "remapy pose",
        spawn: bool = True,
        save_path: str | None = None,
    ) -> None:
        # When saving to a file we must not also spawn a viewer.
        rr.init(application_id, spawn=spawn and save_path is None)
        if save_path is not None:
            rr.save(save_path)

    def log_frame(
        self,
        frame_count: int,
        elapsed_s: float,
        fps: float,
        frame_bgr: np.ndarray,
        result,
    ) -> None:
        """Log one processed frame to the current Rerun recording."""
        # Place everything at this point on both the frame and wall-clock timelines.
        rr.set_time(FRAME_TIMELINE, sequence=frame_count)
        rr.set_time(TIME_TIMELINE, duration=elapsed_s)

        rr.log("video/image", rr.Image(frame_bgr[:, :, ::-1]))  # BGR -> RGB
        rr.log("metrics/fps", rr.Scalars(fps))

        if not result.pose_landmarks:
            # Clear stale skeletons so they don't linger when tracking drops out.
            rr.log("video/image/skeleton", rr.Clear(recursive=True))
            rr.log("pose3d", rr.Clear(recursive=True))
            return

        self._log_skeleton_2d(frame_bgr.shape, result.pose_landmarks[0])
        self._log_skeleton_3d(result.pose_world_landmarks[0])
        self._log_angles(result.pose_world_landmarks[0])

    def _log_skeleton_2d(self, shape, landmarks) -> None:
        h, w = shape[:2]
        pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)
        strips = [np.stack([pts[s], pts[e]]) for s, e in POSE_CONNECTIONS]
        rr.log("video/image/skeleton", rr.LineStrips2D(strips, colors=_BONE_COLOR))
        rr.log("video/image/keypoints", rr.Points2D(pts, colors=_POINT_COLOR, radii=3.0))

    def _log_skeleton_3d(self, world_landmarks) -> None:
        pts = _world_xyz(world_landmarks)
        strips = [np.stack([pts[s], pts[e]]) for s, e in POSE_CONNECTIONS]
        rr.log("pose3d/skeleton", rr.LineStrips3D(strips, colors=_BONE_COLOR))
        rr.log("pose3d/keypoints", rr.Points3D(pts, colors=_POINT_COLOR, radii=0.02))

    def _log_angles(self, world_landmarks) -> None:
        for name, deg in joint_angles(world_landmarks).items():
            if deg == deg:  # skip NaN
                rr.log(f"metrics/angles/{name}", rr.Scalars(deg))
