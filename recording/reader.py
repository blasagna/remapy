"""Read a recording written by :class:`recording.recorder.HDF5Recorder`.

Loads the stored raw signals into numpy arrays and recomputes the derived
quantities that were intentionally *not* stored (joint angles, fps, pose-present,
decoded frames), reusing :func:`pose_estimation.angles.joint_angles`.
"""

from pathlib import Path
from types import SimpleNamespace

import cv2
import h5py
import numpy as np

from .annotations import DELETED_DS, END_DS, LABEL_DS, START_DS, Annotation, _decode


class Recording:
    """Open a recording ``.h5`` read-only; use as a context manager or call ``close()``."""

    def __init__(self, path: Path | str) -> None:
        self._f = h5py.File(str(path), "r")
        self.timestamps_ms = self._f["timestamps_ms"][:]
        self.landmarks_norm = self._f["pose/landmarks_norm"][:]
        self.landmarks_world = self._f["pose/landmarks_world"][:]
        self.visibility = self._f["pose/visibility"][:]
        self.presence = self._f["pose/presence"][:]
        self.landmark_names = [
            n.decode() if isinstance(n, bytes) else str(n)
            for n in self._f["meta/landmark_names"][:]
        ]
        self.pose_connections = self._f["meta/pose_connections"][:]
        self.metadata = {k: self._f.attrs[k] for k in self._f.attrs}
        # Read-only snapshot of any annotations (see recording.annotations); empty
        # on recordings that predate the feature. Written via AnnotationStore, not here.
        self.annotations: list[Annotation] = self._load_annotations()
        # Per-stream Feather Sense sensor data if the device was recorded; empty
        # dict otherwise. See recording.recorder.HDF5Recorder.append_sensor.
        self.feather: dict[str, SimpleNamespace] = self._load_feather()

    def _load_annotations(self) -> list[Annotation]:
        if LABEL_DS not in self._f:
            return []
        labels = self._f[LABEL_DS][:]
        starts = self._f[START_DS][:]
        ends = self._f[END_DS][:]
        deleted = self._f[DELETED_DS][:]
        return [
            Annotation(i, _decode(lbl), int(s), int(e))
            for i, (lbl, s, e, d) in enumerate(zip(labels, starts, ends, deleted))
            if not bool(d)
        ]

    def _load_feather(self) -> dict:
        """Load each ``/feather/<stream>`` group into a namespace of arrays.

        Numeric streams expose ``timestamps_ms``, ``values`` (M, K) and
        ``fields``; the ``error`` stream exposes ``timestamps_ms``, ``source``
        and ``message`` string arrays. Empty dict when no device was recorded.

        ``gravity`` and ``linear_accel`` are **not stored** — the board streams
        raw accel only. They are derived here at the default time constant and
        added to the dict (each flagged ``derived=True``), so consumers see the
        same streams as before. Use :meth:`motion` to re-derive with a different
        time constant.
        """
        if "feather" not in self._f:
            return {}
        out: dict[str, SimpleNamespace] = {}
        grp = self._f["feather"]
        for name in grp:
            g = grp[name]
            fields = [
                s.decode() if isinstance(s, bytes) else str(s)
                for s in g.attrs.get("fields", [])
            ]
            ns = SimpleNamespace(timestamps_ms=g["timestamps_ms"][:], fields=fields, derived=False)
            if name == "error":
                ns.source = np.array([_decode(s) for s in g["source"][:]])
                ns.message = np.array([_decode(s) for s in g["message"][:]])
            else:
                ns.values = g["values"][:]
            out[name] = ns
        out.update(self._derive_motion(out))
        return out

    def _derive_motion(self, streams: dict, tau_s: float | None = None) -> dict:
        """Reconstruct gravity / linear_accel from a loaded raw ``accel`` stream.

        Returns ``{}`` when the recording has no accel data (nothing to derive
        from), including recordings made before the device was wired up.
        """
        from adafruit_feather_sense.motion import GRAVITY_TAU_S, derive_motion

        accel = streams.get("accel")
        if accel is None or accel.timestamps_ms.shape[0] == 0:
            return {}
        gravity, linear = derive_motion(
            accel.timestamps_ms,
            accel.values,
            GRAVITY_TAU_S if tau_s is None else tau_s,
        )
        return {
            name: SimpleNamespace(
                timestamps_ms=accel.timestamps_ms,
                values=np.asarray(vals, dtype=np.float32),
                fields=["x", "y", "z"],
                derived=True,
            )
            for name, vals in (("gravity", gravity), ("linear_accel", linear))
        }

    def motion(self, tau_s: float | None = None) -> dict:
        """Re-derive ``gravity`` / ``linear_accel`` with a custom time constant.

        ``self.feather`` already carries both at the default
        ``motion.GRAVITY_TAU_S``; this recomputes them from the same stored raw
        accel for a different ``tau_s`` (larger = steadier gravity, slower to
        follow reorientation). Returns ``{}`` if the recording has no accel.
        """
        return self._derive_motion(self.feather, tau_s)

    def __len__(self) -> int:
        return int(self.timestamps_ms.shape[0])

    @property
    def pose_present(self) -> np.ndarray:
        """Boolean mask of frames that have a detected pose (non-NaN landmarks)."""
        return ~np.isnan(self.landmarks_world[:, 0, 0])

    def fps(self) -> np.ndarray:
        """Instantaneous fps between consecutive frames (length ``N-1``)."""
        dt = np.diff(self.timestamps_ms).astype(np.float64)
        return np.where(dt > 0, 1000.0 / dt, np.nan)

    def frame(self, i: int) -> np.ndarray:
        """Decode and return the i-th video frame as a BGR array."""
        blob = self._f["video/jpeg"][i]
        return cv2.imdecode(np.frombuffer(bytes(blob), np.uint8), cv2.IMREAD_COLOR)

    def frame_jpeg(self, i: int) -> bytes:
        """Return the i-th frame's stored JPEG blob, undecoded.

        For consumers that want the archived bytes verbatim (e.g. re-logging to
        Rerun), avoiding a lossy decode/re-encode round trip.
        """
        return bytes(self._f["video/jpeg"][i])

    def frames(self):
        """Iterate decoded BGR frames."""
        for i in range(len(self)):
            yield self.frame(i)

    def angles(self):
        """Recompute the 8 joint angles per frame as a pandas DataFrame (degrees).

        NaN for frames with no pose. Columns are the ``JOINT_TRIPLETS`` names.
        """
        import pandas as pd

        from pose_estimation.angles import JOINT_TRIPLETS, joint_angles

        present = self.pose_present
        rows = []
        for i in range(len(self)):
            if not present[i]:
                rows.append({name: np.nan for name in JOINT_TRIPLETS})
                continue
            lms = [
                SimpleNamespace(x=float(x), y=float(y), z=float(z))
                for x, y, z in self.landmarks_world[i]
            ]
            rows.append(joint_angles(lms))
        return pd.DataFrame(rows, columns=list(JOINT_TRIPLETS))

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None

    def __enter__(self) -> Recording:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
