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

from adafruit_feather_sense.motion import GravityFilter
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


# Feather Sense time-series views (added to the blueprint only when the device is
# present). 3-axis streams share one plot (child x/y/z lines); scalar streams get
# their own so differing magnitudes don't crush each other onto a shared axis.
# linear_accel/gravity are derived here from the raw accel stream (see
# log_sensors), not sent by the board.
def _feather_grid() -> rrb.Grid:
    return rrb.Grid(
        rrb.TimeSeriesView(origin="feather/accel", name="accel (m/s²)"),
        rrb.TimeSeriesView(origin="feather/linear_accel", name="linear accel (m/s²)"),
        rrb.TimeSeriesView(origin="feather/gravity", name="gravity (m/s²)"),
        rrb.TimeSeriesView(origin="feather/gyro", name="gyro (rad/s)"),
        rrb.TimeSeriesView(origin="feather/mag", name="mag (µT)"),
        rrb.TimeSeriesView(origin="feather/battery/voltage_v", name="battery (V)"),
        name="Feather Sense",
    )


def _build_blueprint(layout: str, feather: bool = False) -> rrb.Blueprint:
    camera_view = rrb.Spatial2DView(origin="video/image", name="Camera + pose")
    # One plot per joint so angles aren't overlaid on a shared axis.
    plots = rrb.Grid(
        *(
            rrb.TimeSeriesView(origin=f"metrics/angles/{name}", name=name)
            for name in JOINT_TRIPLETS
        )
    )
    if layout == "tabs":
        views = [camera_view, plots]
        if feather:
            views.append(_feather_grid())
        root = rrb.Tabs(*views, name="View")
    else:
        main = rrb.Horizontal(camera_view, plots, column_shares=[1, 1], name="Camera + angles")
        # Keep the camera/angles split; put the sensor plots on a second tab so
        # they don't crowd the video.
        root = rrb.Tabs(main, _feather_grid()) if feather else main
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
        feather: bool = False,
    ) -> None:
        self._jpeg_quality = int(jpeg_quality)
        # Offset mapping the device's monotonic ms clock onto the session time
        # axis, fixed on the first sensor sample so inter-sample timing is kept.
        self._sensor_offset: float | None = None
        # The board streams raw accel only; gravity/linear are reconstructed here.
        self._gravity_filter = GravityFilter()
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
        rr.send_blueprint(
            _build_blueprint(layout, feather=feather), make_active=True, make_default=True
        )

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

    def log_sensors(self, records, elapsed_s: float) -> None:
        """Log a batch of Feather Sense SensorRecords as time series.

        Each record is placed on the shared ``time`` timeline using its own
        device timestamp (mapped onto session time via a one-time offset), so the
        plots keep the sensors' true relative timing and line up with the video.
        3-axis streams log child ``x/y/z`` scalars under ``feather/<name>``;
        scalar streams log under their own leaf; errors go to a text log.

        Each raw ``accel`` sample additionally yields the derived ``gravity`` and
        ``linear_accel`` plots, filtered here rather than on the board.
        """
        for rec in records:
            if self._sensor_offset is None:
                self._sensor_offset = elapsed_s - rec.timestamp_ms / 1000.0
            rr.set_time(TIME_TIMELINE, duration=rec.timestamp_ms / 1000.0 + self._sensor_offset)

            if rec.name == "error":
                source, message = rec.values
                rr.log("feather/error", rr.TextLog(f"{source}: {message}", level="WARN"))
                continue

            for i, value in enumerate(rec.values):
                field = rec.fields[i] if i < len(rec.fields) else f"v{i}"
                rr.log(f"feather/{rec.name}/{field}", rr.Scalars(float(value)))

            if rec.name == "accel":
                self._log_derived_motion(rec)

    def _log_derived_motion(self, accel_rec) -> None:
        """Log gravity / linear_accel derived from one raw accel record.

        Shares the accel record's timeline position (already set by the caller),
        so the derived plots sit exactly on their source sample.
        """
        gravity, linear = self._gravity_filter.update(accel_rec.timestamp_ms, accel_rec.values)
        for name, xyz in (("gravity", gravity), ("linear_accel", linear)):
            for field, value in zip(("x", "y", "z"), xyz):
                rr.log(f"feather/{name}/{field}", rr.Scalars(float(value)))

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
