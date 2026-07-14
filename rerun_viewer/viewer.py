"""Rerun logging helpers for pose estimation.

Given raw frames and MediaPipe ``PoseLandmarkerResult`` objects, this logs to the
Rerun viewer:

- the video frame (``video/image``) with the 2D skeleton overlaid,
- a 3D skeleton reconstructed from the metric *world* landmarks (``pose3d``),
- scalar time series (line plots), one per joint angle (``metrics/angles/*``);
  frame rate (``metrics/fps``) is also logged but left out of the blueprint's
  plots to avoid crowding the view.

A blueprint (see ``_build_blueprint``) arranges the camera/pose view and a grid of
per-joint line plots either side by side (``layout="split"``, the default, each
taking half the screen) or on separate tabs (``layout="tabs"``), instead of
relying on the viewer's default auto-layout.

Uses the rerun-sdk API (NOT the unrelated ``rerun`` file-watcher package).
"""

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from pose_estimation.angles import JOINT_TRIPLETS, joint_angles
from pose_estimation.estimator import POSE_CONNECTIONS

# Timeline names shown on the Rerun time axis.
FRAME_TIMELINE = "frame"
TIME_TIMELINE = "time"

_POINT_COLOR = (0, 200, 255)
_BONE_COLOR = (0, 255, 0)

# --layout choices for PoseRerunLogger: "split" puts the camera/pose view and the
# line plots side by side (each half the screen); "tabs" puts them in separate tabs.
LAYOUTS = ("split", "tabs")


def _build_blueprint(layout: str) -> rrb.Blueprint:
    camera_view = rrb.Spatial2DView(origin="video/image", name="Camera + pose")
    # One plot per joint so angles aren't overlaid on a shared axis.
    plots = rrb.Grid(
        *(
            rrb.TimeSeriesView(origin=f"metrics/angles/{name}", name=name)
            for name in JOINT_TRIPLETS
        )
    )
    if layout == "tabs":
        root = rrb.Tabs(camera_view, plots, name="View")
    else:
        root = rrb.Horizontal(camera_view, plots, column_shares=[1, 1])
    return rrb.Blueprint(root, collapse_panels=True)


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
        memory_limit: str = "75%",
        jpeg_quality: int = 75,
        layout: str = "split",
    ) -> None:
        self._jpeg_quality = int(jpeg_quality)
        rr.init(application_id)
        # Saving and spawning are mutually exclusive sinks; spawn explicitly so we
        # can pass the viewer's in-memory store limit.
        if save_path is not None:
            rr.save(save_path)
        elif spawn:
            rr.spawn(memory_limit=memory_limit)
        # make_active overrides whatever blueprint the viewer last had active for
        # this application_id (e.g. one the user rearranged by hand in an earlier
        # run) so the chosen --layout always takes effect.
        rr.send_blueprint(_build_blueprint(layout), make_active=True, make_default=True)

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

        # JPEG-encode the frame to keep the viewer's in-memory store small. cv2
        # encodes directly from BGR, so no color conversion is needed here.
        ok, buf = cv2.imencode(
            ".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
        )
        if ok:
            rr.log("video/image", rr.EncodedImage(contents=buf.tobytes(), media_type="image/jpeg"))
        rr.log("metrics/fps", rr.Scalars(fps))

        if not result.pose_landmarks:
            # Clear stale skeletons so they don't linger when tracking drops out.
            rr.log("video/image/skeleton", rr.Clear(recursive=True))
            rr.log("pose3d", rr.Clear(recursive=True))
            return

        self._log_skeleton_2d(frame_bgr.shape, result.pose_landmarks[0])
        # self._log_skeleton_3d(result.pose_world_landmarks[0])
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
