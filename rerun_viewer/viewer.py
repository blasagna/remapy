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


# Live-metric views (added only when --live-metrics is on). Sway and cadence get their
# own plots rather than a shared axis; `coverage` sits with them on purpose, because a
# sway number is only readable next to the fraction of the window it was measured over.
def _live_grid(mode: str) -> rrb.Grid:
    quality = [
        rrb.TimeSeriesView(origin="live/coverage", name="coverage (0-1)"),
        rrb.TimeSeriesView(origin="live/tracked_s", name="tracked run (s)"),
    ]
    if mode == "crawl":
        views = [
            rrb.TimeSeriesView(origin="live/cadence_cpm", name="cadence (cycles/min)"),
            rrb.TimeSeriesView(origin="live/cycle_period_cv", name="cycle period CV"),
        ]
    else:
        views = [
            rrb.TimeSeriesView(origin="live/sway_rms_m", name="sway RMS (m)"),
            rrb.TimeSeriesView(origin="live/sway_velocity_mps", name="sway velocity (m/s)"),
            rrb.TimeSeriesView(origin="live/trunk_angle_delta_deg", name="trunk Δ from baseline (°)"),
        ]
    return rrb.Grid(*views, *quality, name="Live metrics")


def _build_blueprint(
    layout: str,
    feather: bool = False,
    annotations: bool = False,
    live: str | None = None,
) -> rrb.Blueprint:
    camera_view = rrb.Spatial2DView(origin="video/image", name="Camera + pose")
    # One plot per joint so angles aren't overlaid on a shared axis.
    plots = rrb.Grid(
        *(
            rrb.TimeSeriesView(origin=f"metrics/angles/{name}", name=name)
            for name in JOINT_TRIPLETS
        )
    )
    # Only present when replaying a recording that carries labeled segments.
    annotations_view = rrb.TextLogView(origin="annotations", name="Annotations")
    if layout == "tabs":
        views = [camera_view, plots]
        if live:
            views.append(_live_grid(live))
        if feather:
            views.append(_feather_grid())
        if annotations:
            views.append(annotations_view)
        root = rrb.Tabs(*views, name="View")
    else:
        main = rrb.Horizontal(camera_view, plots, column_shares=[1, 1], name="Camera + angles")
        # Keep the camera/angles split; put the sensor plots on a second tab so
        # they don't crowd the video.
        extra = ([_live_grid(live)] if live else []) + (
            [_feather_grid()] if feather else []
        ) + ([annotations_view] if annotations else [])
        root = rrb.Tabs(main, *extra) if extra else main
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
        annotations: bool = False,
        live: str | None = None,
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
            _build_blueprint(layout, feather=feather, annotations=annotations, live=live),
            make_active=True,
            make_default=True,
        )

    def log_frame(
        self,
        frame_count: int,
        elapsed_s: float,
        fps: float,
        frame_bgr: np.ndarray | None,
        result,
        jpeg_bytes: bytes | None = None,
        image_size: tuple[int, int] | None = None,
    ) -> None:
        """Log one processed frame to the current Rerun recording.

        Live capture passes ``frame_bgr`` and the frame is JPEG-encoded here.
        Replaying a stored recording instead passes the archived ``jpeg_bytes``
        (logged verbatim, no decode/re-encode round trip) plus ``image_size`` as
        ``(height, width)``, which the 2D skeleton needs to scale the normalized
        landmarks; ``frame_bgr`` may then be ``None``.
        """
        # Place everything at this point on both the frame and wall-clock timelines.
        rr.set_time(FRAME_TIMELINE, sequence=frame_count)
        rr.set_time(TIME_TIMELINE, duration=elapsed_s)

        if jpeg_bytes is not None:
            rr.log("video/image", rr.EncodedImage(contents=jpeg_bytes, media_type="image/jpeg"))
        elif frame_bgr is not None:
            # JPEG-encode the frame to keep the viewer's in-memory store small. cv2
            # encodes directly from BGR, so no color conversion is needed here.
            ok, buf = cv2.imencode(
                ".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
            )
            if ok:
                rr.log(
                    "video/image",
                    rr.EncodedImage(contents=buf.tobytes(), media_type="image/jpeg"),
                )
        rr.log("metrics/fps", rr.Scalars(fps))

        if not result.pose_landmarks:
            # Clear stale skeletons so they don't linger when tracking drops out.
            rr.log("video/image/skeleton", rr.Clear(recursive=True))
            rr.log("pose3d", rr.Clear(recursive=True))
            return

        shape = image_size if image_size is not None else frame_bgr.shape
        self._log_skeleton_2d(shape, result.pose_landmarks[0])
        # self._log_skeleton_3d(result.pose_world_landmarks[0])
        self._log_angles(result.pose_world_landmarks[0])

    def log_live_metrics(self, metrics) -> None:
        """Log a :class:`motor_metrics.live.LiveMetrics` readout as time series.

        Called right after ``log_frame``, so it inherits the timeline position already
        set there and each point lands on the frame it describes.

        NaN fields are skipped rather than logged, which is what makes the blanking rule
        visible: while coverage is below the gate the metric plots simply stop advancing,
        instead of drawing a flat line that looks like a steady measurement of a child
        the tracker has actually lost. ``coverage`` itself keeps logging throughout — it
        is the signal explaining the gap.
        """
        quality = {
            "coverage": metrics.live_coverage,
            "tracked_s": metrics.live_tracked_s,
        }
        measured = {
            "sway_rms_m": metrics.live_sway_rms_m,
            "sway_ml_rms_m": metrics.live_sway_ml_rms_m,
            "sway_ap_rms_m": metrics.live_sway_ap_rms_m,
            "sway_velocity_mps": metrics.live_sway_velocity_mps,
            "trunk_angle_delta_deg": metrics.live_trunk_angle_delta_deg,
            "cadence_cpm": metrics.live_cadence_cpm,
            "cycle_period_cv": metrics.live_cycle_period_cv,
        }
        for name, value in (*quality.items(), *measured.items()):
            if value == value:  # skip NaN, as _log_angles does
                rr.log(f"live/{name}", rr.Scalars(float(value)))

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

    def log_annotations(self, annotations, timestamps_ms) -> None:
        """Log labeled time segments as a text log that turns on and off.

        Each segment writes its label under ``annotations/<label>`` at its start
        and an ``rr.Clear`` at its end, so scrubbing shows which trials are
        active. Segments are placed on *both* timelines — ``timestamps_ms`` (the
        recording's per-frame clock, which the annotations share) maps each
        boundary onto a frame index, so the labels are visible whichever timeline
        the viewer is scrubbing.
        """
        ts = np.asarray(timestamps_ms)
        if ts.size == 0:
            return
        t0 = float(ts[0])
        for ann in annotations:
            # "/" would silently nest the label into sub-entities; nothing else
            # in the label vocabulary conflicts with an entity path.
            path = f"annotations/{ann.label.replace('/', '_')}"
            for boundary, entity in ((ann.start_ms, None), (ann.end_ms, rr.Clear(recursive=True))):
                rr.set_time(FRAME_TIMELINE, sequence=int(np.searchsorted(ts, boundary)))
                rr.set_time(TIME_TIMELINE, duration=(float(boundary) - t0) / 1000.0)
                rr.log(path, entity if entity is not None else rr.TextLog(ann.label))

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
