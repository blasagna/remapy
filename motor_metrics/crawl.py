"""Prone belly-crawl (GMFM dim C): cadence, cycle variability, left-right reciprocity.

This is Remy's locomotion today, so it is where change is most likely to show first —
and it is also where what the score sheet asks for and what the camera can honestly
answer come apart. Worth being blunt about that up front.

**What the GMFM item asks for is distance: "creeps forward 1.8 m". This package cannot
measure that.** ``landmarks_world`` is hip-centered — MediaPipe puts the frame's origin
*at* the pelvis — so the pelvis cannot travel in it by construction, and travel across
the floor is simply absent from the metric signal. There is no second camera, no depth
sensor, and (in a camera-only recording) no IMU to recover it from. What remains is
:func:`motor_metrics.signals.com_norm`, the pelvis in *image fractions*, which is only
comparable within a session at a fixed camera and a roughly fixed distance from it. It is
reported here as ``speed_norm_per_s``, in image-widths per second, and it is never called
metres. Converting it honestly needs a scale reference; if a real crawl speed ever
matters, the answer is a second camera or a floor fiducial, not cleverer math on this
data.

**What it can measure is the pattern, and that is the better metric anyway.** Limb motion
*relative to the pelvis* is exactly what the hip-centered frame captures perfectly. So:

- **Cadence** — how many pull cycles per minute, per side.
- **Cycle variability** — how metronomic they are (``cycle_period_cv``). Consistency is a
  maturity axis in its own right.
- **Reciprocity** (``phase_offset``) — whether the arms alternate (0.5, a mature crawl) or
  pull together (0.0, a symmetric "bunny" haul). This is a genuine developmental axis and
  it costs nothing to measure here.
- **Amplitude symmetry** — whether one arm does more of the work.

Cadence and reciprocity are the deliverable. They are also the metrics likelier to move
before the ordinal score does, which is this project's whole thesis — so leading with them
over distance is the right call on the merits, not a consolation for a missing sensor.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import find_peaks, hilbert

from .derive import FS, resample_uniform, smooth
from .hold import path_length
from .quality import TORSO, WRISTS, Gate, coverage, landmarks_ok, longest_run
from .signals import com_norm, mid, trunk_vector
from .transition import symmetry_index

from mediapipe.tasks.python.vision import PoseLandmark as L  # isort: skip

# A crawl needs the torso (for the body axis) and both wrists (for the limb signals), so
# one gate and one good run serve every field -- the phase comparison needs both arms on
# the same frames anyway.
CRAWL_LANDMARKS = tuple(TORSO) + tuple(WRISTS)

# A peak counts as a pull cycle if it stands this far above the surrounding trough, as a
# fraction of the signal's full range.
CYCLE_PROMINENCE_FRAC = 0.20

# ...but a *relative* gate alone is scale-invariant, and that is a trap: normalized
# against its own range, pure landmark jitter on a motionless arm looks exactly like a
# crawl, and a still child reports a confident, entirely fictional cadence (measured: 57
# cycles from pure noise). So a signal must also clear an ABSOLUTE excursion in meters
# before it is treated as movement at all. A real belly-crawl arm pull travels 10-20 cm
# along the body axis and wrist landmark noise is on the order of 1 cm, so 2 cm sits well
# clear of both. It is a convention, not a validated constant -- check it against real
# trials, and keep it fixed across sessions you intend to compare.
MIN_CYCLE_EXCURSION_M = 0.02

_MARKERS = {"wrist": (L.LEFT_WRIST, L.RIGHT_WRIST), "knee": (L.LEFT_KNEE, L.RIGHT_KNEE)}


@dataclass(frozen=True)
class CrawlMetrics:
    """One belly-crawl trial.

    Every field here is limb-relative-to-pelvis and so fully available — except
    ``speed_norm_per_s``, which is in **image fractions per second, not metres**. See the
    module docstring before comparing it across sessions.
    """

    duration_s: float
    tracked_s: float
    coverage: float
    n_frames: int

    cadence_cpm: float  # pooled across sides
    cadence_cpm_left: float
    cadence_cpm_right: float
    n_cycles_left: int
    n_cycles_right: int

    cycle_period_sd_s: float
    cycle_period_cv: float  # sd/mean; dimensionless, so comparable across cadences

    phase_offset: float  # 0.5 = reciprocal (mature), 0.0 = symmetric ("bunny" haul)
    phase_offset_circular_sd: float  # how consistently that pattern holds
    amplitude_symmetry: float  # 0 = both arms working equally; sign gives the side

    speed_norm_per_s: float  # IMAGE WIDTHS per second. Not metres. Within-session only.


def limb_signal(rec, seg, side: str, marker: str = "wrist") -> np.ndarray:
    """The limb's position along the body's long axis, per frame, in meters.

    ``dot(limb - mid_hip, trunk_unit)``: how far the wrist (or knee) is toward the head,
    measured relative to the pelvis and to the body's own axis. In prone there is no
    useful vertical, so the trunk vector — not gravity — is the axis a crawl cycle
    oscillates along. Because the world frame is pelvis-centered, this is exactly the
    frame the measurement wants, and the hip-centering that defeats distance is what
    makes the pattern clean.

    Returns the raw per-frame signal over ``[seg.start, seg.stop)``; NaN where the pose
    or the trunk axis is unusable.
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}.")
    if marker not in _MARKERS:
        raise ValueError(f"marker must be one of {tuple(_MARKERS)}, got {marker!r}.")

    world = np.asarray(rec.landmarks_world, dtype=np.float64)[seg.start : seg.stop]
    if world.shape[0] == 0:
        return np.empty(0)

    index = _MARKERS[marker][0 if side == "left" else 1]
    trunk = trunk_vector(world)
    norms = np.linalg.norm(trunk, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        axis = np.where(norms > 1e-9, trunk / norms, np.nan)
    limb = world[:, index, :] - mid(world, L.LEFT_HIP, L.RIGHT_HIP)
    return np.einsum("ij,ij->i", limb, axis)


def cycles(signal, fs: float = FS, *, min_excursion: float = MIN_CYCLE_EXCURSION_M) -> np.ndarray:
    """Indices of pull-cycle peaks in a limb signal (which is in meters).

    Two gates, and both are needed. The signal must travel at least ``min_excursion``
    overall — otherwise it is a still arm and there are no cycles to find, however
    peaky its jitter looks — and each peak must then clear
    :data:`CYCLE_PROMINENCE_FRAC` of the signal's range. The absolute gate is the one
    that stops noise from being normalized up into a plausible cadence; see
    :data:`MIN_CYCLE_EXCURSION_M`.

    Returns empty for a signal that is unusable, flat, or below the excursion floor.
    """
    v = np.asarray(signal, dtype=np.float64)
    if v.size < 3 or not np.isfinite(v).all():
        return np.empty(0, dtype=int)
    span = float(np.max(v) - np.min(v))
    if span < min_excursion:
        return np.empty(0, dtype=int)
    peaks, _ = find_peaks(v, prominence=CYCLE_PROMINENCE_FRAC * span)
    return peaks


def phase_offset(left, right, fs: float = FS) -> tuple[float, float]:
    """Reciprocity of two limb signals: ``(offset, circular_sd)``.

    ``offset`` is the mean phase difference as a fraction of a cycle: **0.5 is perfectly
    reciprocal** (the arms alternate — a mature crawl) and **0.0 is symmetric** (both
    arms pull together). ``circular_sd`` says how consistently that relationship holds
    across the trial; a low value means a settled pattern rather than an average of
    changing ones.

    Uses the analytic signal's instantaneous phase, and circular statistics for the mean
    — a plain mean of angles would put the average of 359 deg and 1 deg at 180.

    Returns ``(nan, nan)`` for signals that are unusable, too short, or below
    :data:`MIN_CYCLE_EXCURSION_M`: two motionless arms have no phase relationship, and
    letting their jitter produce one would invent a crawl pattern out of nothing.
    """
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.size != b.size or a.size < 4:
        return float("nan"), float("nan")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float("nan"), float("nan")
    if np.ptp(a) < MIN_CYCLE_EXCURSION_M or np.ptp(b) < MIN_CYCLE_EXCURSION_M:
        return float("nan"), float("nan")

    # Hilbert needs zero-mean input, or the DC term dominates the phase.
    phase_a = np.angle(hilbert(a - a.mean()))
    phase_b = np.angle(hilbert(b - b.mean()))
    diff = phase_a - phase_b

    resultant = np.mean(np.exp(1j * diff))
    # |resultant| is <= 1 analytically, but rounding can nudge it just over, and then
    # log() goes positive and the sqrt below takes the root of a negative number.
    magnitude = float(np.clip(np.abs(resultant), 0.0, 1.0))
    mean_diff = float(np.angle(resultant))
    # Fold to [0, pi]: leading by a third of a cycle and lagging by one are the same
    # amount of reciprocity, and which arm is nominally "first" is arbitrary.
    offset = abs(mean_diff) / (2 * np.pi)

    circular_sd = float(np.sqrt(-2 * np.log(magnitude))) if magnitude > 1e-12 else float("inf")
    return offset, circular_sd


def crawl_metrics(rec, seg, *, gate: Gate = Gate()) -> CrawlMetrics:
    """Metrics for one ``crawl`` segment.

    Computed over the longest run where the torso and **both** wrists are trusted — the
    reciprocity comparison needs both arms on the same frames. Returns NaN fields rather
    than raising for trials that are empty, untracked, or too short.
    """
    ts = np.asarray(rec.timestamps_ms)

    ok = landmarks_ok(rec, CRAWL_LANDMARKS, gate)
    cov = coverage(ok, seg.start, seg.stop)
    r0, r1 = longest_run(ok, seg.start, seg.stop)
    tracked_s = _span_seconds(ts, r0, r1)

    run = _Span(r0, r1)
    left = _prepared(rec, run, "left", ts)
    right = _prepared(rec, run, "right", ts)

    peaks_l, peaks_r = cycles(left), cycles(right)
    offset, circ_sd = phase_offset(left, right)

    return CrawlMetrics(
        duration_s=_span_seconds(ts, seg.start, seg.stop),
        tracked_s=tracked_s,
        coverage=cov,
        n_frames=seg.stop - seg.start,
        cadence_cpm=_cadence(peaks_l.size + peaks_r.size, 2 * tracked_s),
        cadence_cpm_left=_cadence(peaks_l.size, tracked_s),
        cadence_cpm_right=_cadence(peaks_r.size, tracked_s),
        n_cycles_left=int(peaks_l.size),
        n_cycles_right=int(peaks_r.size),
        **_period_stats(peaks_l, peaks_r),
        phase_offset=offset,
        phase_offset_circular_sd=circ_sd,
        amplitude_symmetry=_amplitude_symmetry(left, right),
        speed_norm_per_s=_speed_norm(rec, r0, r1, tracked_s),
    )


class _Span:
    """Minimal duck-type of Segment for the internal limb_signal calls."""

    def __init__(self, start: int, stop: int) -> None:
        self.start, self.stop = start, stop


def _prepared(rec, span, side: str, ts: np.ndarray) -> np.ndarray:
    """A limb signal on the uniform grid and through the pinned smoothing chain."""
    raw = limb_signal(rec, span, side)
    if raw.size < 2 or not np.isfinite(raw).all():
        return np.empty(0)
    _, uniform = resample_uniform(ts[span.start : span.stop], raw)
    if uniform.size == 0:
        return np.empty(0)
    out = smooth(uniform)
    return out if np.isfinite(out).all() else np.empty(0)


def _span_seconds(ts: np.ndarray, start: int, stop: int) -> float:
    if stop - start < 2:
        return 0.0
    return float(ts[stop - 1] - ts[start]) / 1000.0


def _cadence(n_cycles: int, seconds: float) -> float:
    if seconds <= 0 or n_cycles == 0:
        return float("nan")
    return float(n_cycles) / seconds * 60.0


def _period_stats(peaks_l: np.ndarray, peaks_r: np.ndarray) -> dict:
    """Cycle-period spread, pooled across sides. CV is the comparable-across-cadence form."""
    periods = np.concatenate(
        [np.diff(peaks_l) / FS, np.diff(peaks_r) / FS]
    ) if (peaks_l.size + peaks_r.size) else np.empty(0)
    if periods.size < 2:
        return dict(cycle_period_sd_s=float("nan"), cycle_period_cv=float("nan"))
    mean = float(periods.mean())
    sd = float(periods.std())
    return dict(
        cycle_period_sd_s=sd,
        cycle_period_cv=sd / mean if mean > 0 else float("nan"),
    )


def _amplitude_symmetry(left: np.ndarray, right: np.ndarray) -> float:
    """Whether one arm travels further than the other, on their excursion ranges."""
    if left.size == 0 or right.size == 0:
        return float("nan")
    return symmetry_index([float(np.ptp(left))], [float(np.ptp(right))])


def _speed_norm(rec, start: int, stop: int, tracked_s: float) -> float:
    """Pelvis travel in IMAGE FRACTIONS per second. See the module docstring."""
    if stop - start < 2 or tracked_s <= 0:
        return float("nan")
    ts = np.asarray(rec.timestamps_ms)[start:stop]
    # Columns 0 and 1 only: column 2 is MediaPipe's relative depth, on a different and
    # much weaker scale, and mixing it in would silently corrupt the distance.
    pelvis = com_norm(np.asarray(rec.landmarks_norm)[start:stop])[:, :2]
    if not np.isfinite(pelvis).all():
        return float("nan")
    _, uniform = resample_uniform(ts, pelvis)
    if uniform.shape[0] == 0:
        return float("nan")
    smoothed = smooth(uniform)
    if not np.isfinite(smoothed).all():
        return float("nan")
    return path_length(smoothed) / tracked_s
