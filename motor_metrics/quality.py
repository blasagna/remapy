"""Frame quality gating for metrics computed off stored pose landmarks.

``Recording`` stores per-landmark ``visibility`` and ``presence``, but nothing in the
pipeline reads them. For metrics they are load-bearing, because of how MediaPipe fails:
it does not drop an occluded landmark, it **extrapolates** one and marks it low-
visibility. So a frame where Remy's arm is under his body still yields coordinates, and
they are invented. ``pose_present`` does not catch this — it is a whole-row NaN check
(``recorder.py`` writes a full 33x3 NaN row when no pose is found at all) and is
correct as-is; the extrapolated frames sail straight through it.

Hence: every metric gates on the landmarks it actually reads, and every metric reports
its :func:`coverage` next to its numbers. A sway figure computed over 40 % of a trial
is not a sway figure, and the caller must be able to see that without reading the code.

All functions here duck-type ``rec``: they touch only ``pose_present``, ``visibility``
and ``presence``, so tests need no HDF5 file.
"""

from dataclasses import dataclass

import numpy as np
from mediapipe.tasks.python.vision import PoseLandmark as L

# Landmark groups a metric might depend on. Gate on what you read, not on all 33 —
# requiring the ankles for a sitting-sway metric would throw away good trials.
TORSO = (L.LEFT_SHOULDER, L.RIGHT_SHOULDER, L.LEFT_HIP, L.RIGHT_HIP)
ARMS = (L.LEFT_ELBOW, L.RIGHT_ELBOW, L.LEFT_WRIST, L.RIGHT_WRIST)
LEGS = (L.LEFT_KNEE, L.RIGHT_KNEE, L.LEFT_ANKLE, L.RIGHT_ANKLE)
WRISTS = (L.LEFT_WRIST, L.RIGHT_WRIST)
KNEES = (L.LEFT_KNEE, L.RIGHT_KNEE)


@dataclass(frozen=True)
class Gate:
    """Thresholds a landmark must clear to be trusted on a given frame.

    The 0.5 defaults are MediaPipe's own detection/presence default and are a starting
    point, not a validated value — tighten them if the notebook's coverage pass shows
    junk frames surviving. Whatever you pick, keep it fixed across sessions you intend
    to compare.
    """

    min_visibility: float = 0.5
    min_presence: float = 0.5


def landmarks_ok(rec, indices, gate: Gate = Gate()) -> np.ndarray:
    """``(N,)`` bool: frames with a pose where **every** landmark in ``indices`` is trusted.

    NaN visibility/presence (the no-pose rows) compare False and so are excluded
    automatically; ``pose_present`` is ANDed in anyway to keep the intent explicit.
    """
    idx = list(indices)
    vis = np.asarray(rec.visibility)[:, idx]
    pres = np.asarray(rec.presence)[:, idx]
    with np.errstate(invalid="ignore"):  # NaN comparisons are the expected path here
        good = (vis >= gate.min_visibility) & (pres >= gate.min_presence)
    return np.asarray(rec.pose_present) & good.all(axis=1)


def coverage(mask: np.ndarray, start: int, stop: int) -> float:
    """Fraction of frames in ``[start, stop)`` that pass ``mask``.

    An empty span returns ``0.0``, not NaN: coverage exists to be threshold-checked
    (``if coverage < 0.8``), and NaN compares False against every threshold, so an
    empty trial would silently *pass* the check it was meant to fail.
    """
    if stop <= start:
        return 0.0
    window = np.asarray(mask)[start:stop]
    return float(np.count_nonzero(window) / window.size)


def longest_run(mask: np.ndarray, start: int, stop: int) -> tuple[int, int]:
    """Longest contiguous ``True`` run of ``mask`` within ``[start, stop)``.

    Returns absolute frame indices ``(run_start, run_stop)``, exclusive stop, so
    ``run_stop - run_start`` is the length. Returns ``(start, start)`` (length 0) when
    nothing passes.

    This is what a *held* duration means: an 8-second sit that dropped tracking in the
    middle is not 8 seconds of measured sitting, and stitching the two halves together
    would invent the transition between them.
    """
    if stop <= start:
        return (start, start)
    window = np.asarray(mask)[start:stop].astype(bool)
    # Pad with False so runs touching either edge produce a boundary in the diff.
    edges = np.diff(np.concatenate(([False], window, [False])).astype(np.int8))
    run_starts = np.flatnonzero(edges == 1)
    run_stops = np.flatnonzero(edges == -1)
    if run_starts.size == 0:
        return (start, start)
    best = int(np.argmax(run_stops - run_starts))
    return (start + int(run_starts[best]), start + int(run_stops[best]))
