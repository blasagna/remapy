"""Write a session to a self-contained HDF5 file for offline analysis.

Stores only the *minimal raw* signals — the face-blurred video (as per-frame JPEG
blobs) and the pose model's raw landmark outputs — so that everything derivable
(joint angles, fps, pose-present, 2D pixel points) can be recomputed later. See
``recording.reader.Recording`` for the read side.

Layout::

    /                     attrs: format_version, created_at, image_width/height,
                                 num_poses, num_landmarks, model_name,
                                 mediapipe_version, blur_style, faces_blurred,
                                 jpeg_quality, coordinate_note, num_frames
    /meta/landmark_names  (33,)        str
    /meta/pose_connections(35, 2)      int32
    /timestamps_ms        (N,)         int64
    /video/jpeg           (N,)         vlen uint8   (one JPEG per frame)
    /pose/landmarks_norm  (N, 33, 3)   float32      (NaN rows when no pose)
    /pose/landmarks_world (N, 33, 3)   float32
    /pose/visibility      (N, 33)      float32
    /pose/presence        (N, 33)      float32

An optional ``/feather/<stream>`` group per Feather Sense stream (accel, gyro,
mag, battery, error) may also be present when that device was streaming. Each
holds its own ``timestamps_ms`` (device clock) plus an ``(M, K)`` float32
``values`` dataset (or ``source``/``message`` string columns for ``error``),
grown independently since the streams are asynchronous and multi-rate. Written
via :meth:`HDF5Recorder.append_sensor`, read back via
``recording.reader.Recording.feather``.

Consistent with the minimal-raw rule above, only what the board actually samples
is stored: ``gravity``/``linear_accel`` are derived from the raw ``accel`` on
read (see ``adafruit_feather_sense.motion``), so their filter time constant stays
re-tunable after capture.

An optional ``/annotations`` group (labeled time segments) may be added *after*
recording by ``recording.annotations.AnnotationStore``; it is absent here.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import h5py
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.vision import PoseLandmark

from pose_estimation.estimator import POSE_CONNECTIONS

FORMAT_VERSION = 1
NUM_LANDMARKS = 33
_VIDEO_FPS = 30.0  # nominal fps for the optional parallel video; authoritative timing is in timestamps

LANDMARK_NAMES = [PoseLandmark(i).name for i in range(NUM_LANDMARKS)]

_COORD_NOTE = (
    "landmarks_norm: x,y in [0,1] image fractions, z relative depth. "
    "landmarks_world: meters, origin at hip center (x-right, y-down, z-toward-camera). "
    "Missing-pose frames are stored as NaN rows (derive pose_present from non-NaN)."
)


def landmark_rows(result) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One frame's MediaPipe result -> ``(norm, world, visibility, presence)`` rows.

    A frame with no detected pose becomes **full-NaN rows**, and that is the whole
    reason this is a shared function rather than an inline block: the NaN convention
    is load-bearing downstream. ``Recording.pose_present`` and
    ``motor_metrics.quality`` both key off it (a NaN landmark-0 x implies the whole
    row is NaN), so any second implementation that filled zeros instead — or wrote
    NaN coordinates but left visibility at 0.0 — would produce frames that read as
    tracked while carrying nothing. Live buffers (``motor_metrics.live.LiveWindow``)
    call this so they cannot drift from what gets recorded.
    """
    if result.pose_landmarks:
        norm_lm = result.pose_landmarks[0]
        world_lm = result.pose_world_landmarks[0]
        return (
            np.array([[lm.x, lm.y, lm.z] for lm in norm_lm], dtype=np.float32),
            np.array([[lm.x, lm.y, lm.z] for lm in world_lm], dtype=np.float32),
            np.array([lm.visibility for lm in norm_lm], dtype=np.float32),
            np.array([lm.presence for lm in norm_lm], dtype=np.float32),
        )
    return (
        np.full((NUM_LANDMARKS, 3), np.nan, dtype=np.float32),
        np.full((NUM_LANDMARKS, 3), np.nan, dtype=np.float32),
        np.full(NUM_LANDMARKS, np.nan, dtype=np.float32),
        np.full(NUM_LANDMARKS, np.nan, dtype=np.float32),
    )


class HDF5Recorder:
    """Append per-frame signals to an HDF5 file; use as a context manager."""

    def __init__(
        self,
        path: Path | str,
        jpeg_quality: int = 75,
        video_path: Path | str | None = None,
        model_name: str = "pose_landmarker_lite",
        blur_style: str | None = None,
        faces_blurred: bool = False,
    ) -> None:
        self.path = str(path)
        self.jpeg_quality = int(jpeg_quality)
        self.video_path = str(video_path) if video_path is not None else None
        self._model_name = model_name
        self._blur_style = blur_style or "none"
        self._faces_blurred = bool(faces_blurred)

        self._file: Optional[h5py.File] = h5py.File(self.path, "w")
        self._writer: Optional[cv2.VideoWriter] = None
        self._ready = False
        self._n = 0
        # Per-stream Feather Sense sensor datasets, created lazily (see
        # append_sensor). Async + multi-rate, so kept separate from the frames.
        self._feather: dict = {}

    def _init_datasets(self, height: int, width: int) -> None:
        f = self._file
        vlen = h5py.vlen_dtype(np.uint8)
        self._ts = f.create_dataset(
            "timestamps_ms", (0,), maxshape=(None,), dtype="int64", chunks=(256,),
            compression="gzip",
        )
        self._jpeg = f.create_dataset(
            "video/jpeg", (0,), maxshape=(None,), dtype=vlen, chunks=(16,), compression="gzip",
        )
        self._lm_norm = f.create_dataset(
            "pose/landmarks_norm", (0, NUM_LANDMARKS, 3), maxshape=(None, NUM_LANDMARKS, 3),
            dtype="float32", chunks=(64, NUM_LANDMARKS, 3), compression="gzip",
        )
        self._lm_world = f.create_dataset(
            "pose/landmarks_world", (0, NUM_LANDMARKS, 3), maxshape=(None, NUM_LANDMARKS, 3),
            dtype="float32", chunks=(64, NUM_LANDMARKS, 3), compression="gzip",
        )
        self._vis = f.create_dataset(
            "pose/visibility", (0, NUM_LANDMARKS), maxshape=(None, NUM_LANDMARKS),
            dtype="float32", chunks=(64, NUM_LANDMARKS), compression="gzip",
        )
        self._pres = f.create_dataset(
            "pose/presence", (0, NUM_LANDMARKS), maxshape=(None, NUM_LANDMARKS),
            dtype="float32", chunks=(64, NUM_LANDMARKS), compression="gzip",
        )

        f.attrs["format_version"] = FORMAT_VERSION
        f.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
        f.attrs["image_width"] = width
        f.attrs["image_height"] = height
        f.attrs["num_poses"] = 1
        f.attrs["num_landmarks"] = NUM_LANDMARKS
        f.attrs["model_name"] = self._model_name
        f.attrs["mediapipe_version"] = mp.__version__
        f.attrs["blur_style"] = self._blur_style
        f.attrs["faces_blurred"] = self._faces_blurred
        f.attrs["jpeg_quality"] = self.jpeg_quality
        f.attrs["coordinate_note"] = _COORD_NOTE
        f.create_dataset(
            "meta/landmark_names", data=np.array(LANDMARK_NAMES, dtype=h5py.string_dtype())
        )
        f.create_dataset("meta/pose_connections", data=np.array(POSE_CONNECTIONS, dtype="int32"))

        if self.video_path is not None:
            self._writer = cv2.VideoWriter(
                self.video_path, cv2.VideoWriter_fourcc(*"mp4v"), _VIDEO_FPS, (width, height)
            )
            if not self._writer.isOpened():
                raise RuntimeError(f"Could not open parallel video writer: {self.video_path!r}")

        self._ready = True

    def append(self, frame_bgr: np.ndarray, timestamp_ms: int, result) -> None:
        """Append one frame's video + pose signals."""
        if self._file is None:
            raise RuntimeError("HDF5Recorder is closed.")
        if not self._ready:
            h, w = frame_bgr.shape[:2]
            self._init_datasets(h, w)

        i = self._n
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        self._jpeg.resize((i + 1,))
        self._jpeg[i] = np.frombuffer(buf.tobytes(), dtype=np.uint8) if ok else np.zeros(0, np.uint8)

        self._ts.resize((i + 1,))
        self._ts[i] = int(timestamp_ms)

        self._lm_norm.resize((i + 1, NUM_LANDMARKS, 3))
        self._lm_world.resize((i + 1, NUM_LANDMARKS, 3))
        self._vis.resize((i + 1, NUM_LANDMARKS))
        self._pres.resize((i + 1, NUM_LANDMARKS))

        self._lm_norm[i], self._lm_world[i], self._vis[i], self._pres[i] = landmark_rows(result)

        if self._writer is not None:
            self._writer.write(frame_bgr)

        self._n += 1

    def append_sensor(self, name: str, timestamp_ms: int, values, fields=None) -> None:
        """Append one Feather Sense sample to its own ``/feather/<name>`` stream.

        Datasets are created on first sight of each stream (its arity is taken
        from ``values``), so streams that arrive at different rates each grow
        independently. Numeric streams store an ``(M, K)`` float32 ``values``
        array; the ``error`` stream stores ``source`` + ``message`` string
        columns. ``timestamps_ms`` is the device clock (ms since board boot).
        """
        if self._file is None:
            raise RuntimeError("HDF5Recorder is closed.")
        store = self._feather.get(name)
        if store is None:
            store = self._create_feather_stream(name, values, fields)
            self._feather[name] = store

        i = store["n"]
        store["ts"].resize((i + 1,))
        store["ts"][i] = int(timestamp_ms)
        if store["kind"] == "error":
            store["source"].resize((i + 1,))
            store["source"][i] = str(values[0])
            store["message"].resize((i + 1,))
            store["message"][i] = str(values[1])
        else:
            store["values"].resize((i + 1, store["k"]))
            store["values"][i] = np.asarray(values, dtype="float32")
        store["n"] = i + 1

    def _create_feather_stream(self, name: str, values, fields) -> dict:
        grp = self._file.create_group(f"feather/{name}")
        ts = grp.create_dataset(
            "timestamps_ms", (0,), maxshape=(None,), dtype="int64", chunks=(256,),
            compression="gzip",
        )
        if fields is not None:
            grp.attrs["fields"] = list(fields)
        if name == "error":
            strdt = h5py.string_dtype()
            source = grp.create_dataset(
                "source", (0,), maxshape=(None,), dtype=strdt, chunks=(64,), compression="gzip",
            )
            message = grp.create_dataset(
                "message", (0,), maxshape=(None,), dtype=strdt, chunks=(64,), compression="gzip",
            )
            return {"n": 0, "kind": "error", "ts": ts, "source": source, "message": message}
        k = len(values)
        vals = grp.create_dataset(
            "values", (0, k), maxshape=(None, k), dtype="float32", chunks=(256, k),
            compression="gzip",
        )
        return {"n": 0, "kind": "num", "ts": ts, "values": vals, "k": k}

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._file is not None:
            if self._ready:
                self._file.attrs["num_frames"] = self._n
            self._file.close()
            self._file = None

    def __enter__(self) -> HDF5Recorder:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
