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
