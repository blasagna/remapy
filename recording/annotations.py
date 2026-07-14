"""Read/write time-segment annotations on an existing recording ``.h5``.

A recording written by :class:`recording.recorder.HDF5Recorder` stores only raw
per-frame signals. This module adds *labeled time segments* (e.g. "walking",
"occluded") on top of an already-written file, stored as an interval table — one
row per segment — so overlapping/concurrent labels are supported for free.

The annotations live in an optional ``/annotations`` group that is created lazily
on the first :meth:`AnnotationStore.add`. Recordings that predate this feature (no
group) remain valid and simply read back as having no annotations.

Layout (appended to the recorder's format)::

    /annotations/label     (M,)  vlen str   one label per segment
    /annotations/start_ms  (M,)  int64      segment start (recording timeline)
    /annotations/end_ms    (M,)  int64      segment end   (>= start_ms)
    /annotations/deleted   (M,)  bool       tombstone flag (row kept, hidden)

Deletes are tombstones, not row removals, so a row index stays valid for the life
of the file. See :class:`recording.reader.Recording` for a read-only view.
"""

from pathlib import Path
from typing import NamedTuple, Optional

import h5py
import numpy as np

LABEL_DS = "annotations/label"
START_DS = "annotations/start_ms"
END_DS = "annotations/end_ms"
DELETED_DS = "annotations/deleted"

_CHUNK = 64  # annotation counts per session are tiny (dozens), unlike per-frame data


class Annotation(NamedTuple):
    """One labeled time segment; ``index`` is its stable HDF5 row index."""

    index: int
    label: str
    start_ms: int
    end_ms: int


def _decode(label) -> str:
    """h5py vlen strings may read back as ``bytes``; normalize to ``str`` (as reader.py does)."""
    return label.decode() if isinstance(label, bytes) else str(label)


class AnnotationStore:
    """Open an existing recording ``.h5`` for annotation edits; use as a context manager.

    Opens the file in ``"r+"`` (read/write, must already exist). Every mutation is
    flushed immediately, so there is no separate save step.
    """

    def __init__(self, path: Path | str) -> None:
        # "r+" requires the file to exist; it never creates one (raises otherwise).
        self._file: Optional[h5py.File] = h5py.File(str(path), "r+")
        self._ready = LABEL_DS in self._file  # group may be absent on older recordings

    def _init_datasets(self) -> None:
        """Create the resizable /annotations datasets (mirrors HDF5Recorder._init_datasets)."""
        f = self._file
        vlen = h5py.string_dtype()
        f.create_dataset(
            LABEL_DS, (0,), maxshape=(None,), dtype=vlen, chunks=(_CHUNK,), compression="gzip"
        )
        f.create_dataset(
            START_DS, (0,), maxshape=(None,), dtype="int64", chunks=(_CHUNK,), compression="gzip"
        )
        f.create_dataset(
            END_DS, (0,), maxshape=(None,), dtype="int64", chunks=(_CHUNK,), compression="gzip"
        )
        f.create_dataset(
            DELETED_DS, (0,), maxshape=(None,), dtype="bool", chunks=(_CHUNK,), compression="gzip"
        )
        self._ready = True

    def add(self, label: str, start_ms: int, end_ms: int) -> int:
        """Append a segment and return its row index. Requires ``end_ms >= start_ms``."""
        if self._file is None:
            raise RuntimeError("AnnotationStore is closed.")
        if int(end_ms) < int(start_ms):
            raise ValueError(f"end_ms ({end_ms}) must be >= start_ms ({start_ms}).")
        if not self._ready:
            self._init_datasets()

        f = self._file
        i = f[LABEL_DS].shape[0]
        for ds, value in (
            (LABEL_DS, str(label)),
            (START_DS, int(start_ms)),
            (END_DS, int(end_ms)),
            (DELETED_DS, False),
        ):
            f[ds].resize((i + 1,))
            f[ds][i] = value
        f.flush()
        return i

    def list(self) -> list[Annotation]:
        """Return all non-deleted segments as :class:`Annotation` tuples, in row order."""
        if self._file is None:
            raise RuntimeError("AnnotationStore is closed.")
        if not self._ready:
            return []
        f = self._file
        labels = f[LABEL_DS][:]
        starts = f[START_DS][:]
        ends = f[END_DS][:]
        deleted = f[DELETED_DS][:]
        return [
            Annotation(i, _decode(lbl), int(s), int(e))
            for i, (lbl, s, e, d) in enumerate(zip(labels, starts, ends, deleted))
            if not bool(d)
        ]

    def delete(self, index: int) -> None:
        """Tombstone the segment at ``index`` (kept on disk, hidden from :meth:`list`)."""
        if self._file is None:
            raise RuntimeError("AnnotationStore is closed.")
        if not self._ready:
            raise IndexError(f"No annotations exist (index {index} out of range).")
        deleted = self._file[DELETED_DS]
        if index < 0 or index >= deleted.shape[0]:
            raise IndexError(f"Annotation index {index} out of range (0..{deleted.shape[0] - 1}).")
        if bool(deleted[index]):
            raise KeyError(f"Annotation {index} is already deleted.")
        deleted[index] = True
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "AnnotationStore":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
