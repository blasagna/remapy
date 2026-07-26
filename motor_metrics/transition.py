"""Sit <-> prone transitions (GMFM dim B): duration, smoothness, submovements, symmetry.

Remy makes these transitions today, which is what makes them worth measuring: the
ordinal item is already scored and will not move for a long time, while *how* he does it
has room to change in both directions. They are also the cleanest thing in this package
to measure — a discrete, repeatable event, and pure kinematics with no scale problem.

**Smoothness is the point.** A maturing movement stops being a chain of corrections and
becomes one motion. That shows up in the *speed profile*: a smooth movement has a single
bell-shaped hump, an effortful one has several. :func:`sparc` (spectral arc length,
Balasubramanian et al.) turns that into one number by measuring the arc length of the
normalized Fourier magnitude spectrum of the speed profile. More negative = less smooth.
It needs no segmentation and no amplitude normalization, which is why it is the standard
choice in motor rehab.

**Why SPARC and not dimensionless jerk.** SPARC responds to *submovements* while being
largely indifferent to small measurement noise, where a jerk-based metric differentiates
three times and is swamped by it. On landmark data at these rates a naive derivative is mostly
noise, so that robustness is what makes any smoothness metric viable here at all.

**But the robustness has a ceiling, and it bounds real use.** Measured against a
synthetic unit-amplitude bell (``SparcTests``), with noise as a fraction of peak speed::

    sigma 0.01  ->  -1.416  (sd 0.003)     sigma 0.10  ->  -2.53  (sd 0.17)
    sigma 0.02  ->  -1.415  (sd 0.006)     sigma 0.20  ->  -3.60  (sd 0.48)
    sigma 0.05  ->  -1.68   (sd 0.24)      sigma 0.40  ->  -5.92  (sd 0.93)

Up to ~2 % of peak speed the score does not move. Past that it both degrades *and*
becomes erratic — at 5 % the spread across noise realizations (sd 0.24) is already large
next to the effect being looked for. So a SPARC score is only worth reading when the
speed profile is clean against its own peak, which in practice means **big, brisk
transitions score reliably and small slow ones may not**. Check it on real trials before
trusting a number; prefer the median of several trials to any single one.

**What it is sensitive to is submovements, not their count.** One clean movement scores
far above the same displacement broken into steps (-1.40 vs -7.54 for two). But *more*
steps do not keep scoring worse — many closely spaced corrections blend back into a
continuous motion and the score recovers (-5.75 at four, -5.04 at five). That is correct
behavior rather than a defect, and it means the number separates *fluid from effortful*;
it does not count corrections. :func:`count_submovements` is what counts them.

**Read SPARC only against itself.** The values here are **not** comparable to published
SPARC figures. The band actually integrated is ``min(SPARC_FC, fs/2)`` — the grid's
Nyquist limit wins whenever it is the lower of the two — and landmark noise fills the top
of whatever that band turns out to be; on top of that the :mod:`motor_metrics.derive`
chain attenuates fast movement by tens of percent. The absolute number is therefore a
property of *this pipeline*, meaningful only as a within-child trend computed through
identical constants. That is exactly why those constants are pinned rather than passed —
and why a change to :data:`motor_metrics.derive.FS` moves every SPARC value ever
computed onto a different scale, since it moves the band.

**What the hip-centered frame can and cannot see.** ``sparc_trunk`` measures trunk
reorientation, which is the principal degree of freedom of a sit<->prone transition, so
it is the primary. ``sparc_tip`` is a cross-check on the trunk-over-pelvis path — and
because the world frame is pinned to the pelvis, it describes *only* that component. A
real transition also drags the pelvis across the floor, and no metric here sees that.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.signal import find_peaks

from .derive import FS, resample_uniform, smooth
from .quality import TORSO, WRISTS, Gate, coverage, landmarks_ok, longest_run
from .signals import WORLD_UP, trunk_from_vertical, trunk_vector

# SPARC parameters, pinned for the same reason as the derive.py constants: two scores
# are comparable only through an identical chain. From the reference implementation.
SPARC_FC = 10.0  # Hz, spectral cutoff
SPARC_AMP_THRESHOLD = 0.05  # adaptive cutoff: ignore spectrum below 5% of the peak
SPARC_PAD_LEVEL = 4  # zero-padding exponent, for frequency resolution

# A velocity peak counts as a submovement if it stands this far above the surrounding
# trough, as a fraction of the profile's peak. A convention, not a validated threshold:
# submovement counting needs *some* prominence rule, and this one is at least explicit.
PEAK_PROMINENCE_FRAC = 0.10

# Onset/offset of the movement, as a fraction of peak angular speed. The usual 5%.
MOVEMENT_THRESHOLD_FRAC = 0.05


@dataclass(frozen=True)
class TransitionMetrics:
    """One sit<->prone transition trial."""

    # `duration_s` / `tracked_s` / `coverage` mean the same thing in every metric
    # dataclass here, so the union table in `report.py` has one column for each rather
    # than a per-exercise hole in whichever the reader happens to scan.
    duration_s: float  # the annotated trial: a human's marks
    tracked_s: float  # longest continuously-trusted run; a data-quality figure
    movement_duration_s: float  # onset -> offset at 5% of peak angular speed
    coverage: float
    n_frames: int

    sparc_trunk: float  # PRIMARY: smoothness of trunk reorientation. Within-child only.
    sparc_tip: float  # cross-check on the trunk-over-pelvis path only
    peak_angular_velocity_dps: float
    n_velocity_peaks: int  # submovements; 1 = one clean motion

    side: Optional[str]  # from the label -- the annotator's call, not measured
    leading_wrist: Optional[str]  # DIAGNOSTIC: sanity-checks the `side` label
    up_source: str


def sparc(speed, fs: float = FS, *, fc: float = SPARC_FC, amp_thresh: float = SPARC_AMP_THRESHOLD):
    """Spectral arc length of a speed profile. More negative = less smooth.

    Follows the reference implementation (Balasubramanian et al., *On the analysis of
    movement smoothness*): normalize the speed profile's Fourier magnitude spectrum,
    take an adaptive cutoff at the last frequency clearing ``amp_thresh`` (capped at
    ``fc``), and measure the arc length of the resulting curve.

    Returns NaN for a profile that is empty, non-finite, or identically zero — a
    stationary trial has no movement whose smoothness could be described.

    See the module docstring before comparing a value to anything but another value from
    this same pipeline.
    """
    v = np.asarray(speed, dtype=np.float64)
    if v.size < 2 or not np.isfinite(v).all():
        return float("nan")
    peak = np.max(np.abs(v))
    if peak == 0:
        return float("nan")

    # There is no information above Nyquist, so a cutoff above it describes nothing. At
    # ``fs=30`` this line is inert (``fc=10`` is well under 15) and the clamp only starts
    # binding at grids below 20 Hz — which is precisely when it matters, and precisely when
    # nobody would think to re-check a constant named for a frequency. Stated here rather
    # than left to fall out of the ``rfftfreq`` axis, so the intent survives a refactor.
    fc = min(fc, fs / 2.0)

    # Real-input transform, so the spectrum is a *half* axis: 0 .. fs/2. A full ``fft``
    # with an ``arange(0, fs, ...)`` axis works out identically whenever ``fc`` sits below
    # Nyquist — the upper half is a mirror image, and it is discarded by the band mask
    # before anything reads it. It stops working the moment ``fc`` does not: the arc would
    # then traverse reflected copies of real content and count that noise twice. Taking the
    # half axis makes the band physically meaningful at any ``fs`` rather than by luck.
    n_fft = int(2 ** (np.ceil(np.log2(v.size)) + SPARC_PAD_LEVEL))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    spectrum = np.abs(np.fft.rfft(v, n_fft))
    spectrum = spectrum / spectrum.max()

    in_band = freqs <= fc
    freqs, spectrum = freqs[in_band], spectrum[in_band]

    # Adaptive cutoff: keep the band between the first and last bin above the threshold.
    above = np.flatnonzero(spectrum >= amp_thresh)
    if above.size < 2:
        return float("nan")
    freqs = freqs[above[0] : above[-1] + 1]
    spectrum = spectrum[above[0] : above[-1] + 1]

    span = freqs[-1] - freqs[0]
    if span <= 0:
        return float("nan")
    return float(
        -np.sum(np.sqrt((np.diff(freqs) / span) ** 2 + np.diff(spectrum) ** 2))
    )


def symmetry_index(left: Sequence[float], right: Sequence[float]) -> float:
    """``2 * (L - R) / (L + R)`` on the medians. 0 = symmetric, sign gives the side.

    A **between-trial** comparison: pass the per-trial values for each side, grouped by
    the label's ``side=``. It is not a per-frame quantity — a single transition happens
    to one side, and asking whether *it* was symmetric is not a question.

    Returns NaN when either side has no trials, or when the medians sum to zero (no
    movement to compare).
    """
    left = [v for v in np.asarray(left, dtype=np.float64).ravel() if np.isfinite(v)]
    right = [v for v in np.asarray(right, dtype=np.float64).ravel() if np.isfinite(v)]
    if not left or not right:
        return float("nan")
    l_med, r_med = float(np.median(left)), float(np.median(right))
    total = l_med + r_med
    if total == 0:
        return float("nan")
    return float(2.0 * (l_med - r_med) / total)


def transition_metrics(
    rec, seg, *, up: np.ndarray = WORLD_UP, gate: Gate = Gate()
) -> TransitionMetrics:
    """Metrics for one ``transition`` segment.

    Computed over the longest continuously-tracked run inside the trial. Returns NaN
    fields rather than raising for trials that are empty, untracked, or shorter than the
    smoothing window.
    """
    ts = np.asarray(rec.timestamps_ms)
    world = np.asarray(rec.landmarks_world)

    ok = landmarks_ok(rec, TORSO, gate)
    cov = coverage(ok, seg.start, seg.stop)
    r0, r1 = longest_run(ok, seg.start, seg.stop)

    angular = _angular_speed(ts[r0:r1], world[r0:r1], up)
    tip_speed = _tip_speed(ts[r0:r1], world[r0:r1])

    return TransitionMetrics(
        duration_s=_span_seconds(ts, seg.start, seg.stop),
        tracked_s=_span_seconds(ts, r0, r1),
        movement_duration_s=_movement_duration(angular),
        coverage=cov,
        n_frames=seg.stop - seg.start,
        sparc_trunk=sparc(angular) if angular.size else float("nan"),
        sparc_tip=sparc(tip_speed) if tip_speed.size else float("nan"),
        peak_angular_velocity_dps=(
            float(np.max(angular)) if angular.size and np.isfinite(angular).all()
            else float("nan")
        ),
        n_velocity_peaks=count_submovements(angular),
        side=seg.parsed.params.get("side"),
        leading_wrist=_leading_wrist(rec, r0, r1, gate),
        up_source="world_y" if np.allclose(up, WORLD_UP) else "custom",
    )


def count_submovements(speed) -> int:
    """Peaks in a speed profile: 1 is one clean motion, more means correction.

    Prominence-gated at :data:`PEAK_PROMINENCE_FRAC` of the peak so landmark jitter does
    not read as a dozen submovements. Returns 0 for an unusable profile.
    """
    v = np.asarray(speed, dtype=np.float64)
    if v.size < 3 or not np.isfinite(v).all():
        return 0
    peak = float(np.max(v))
    if peak <= 0:
        return 0
    peaks, _ = find_peaks(v, prominence=PEAK_PROMINENCE_FRAC * peak)
    return int(peaks.size)


def _span_seconds(ts: np.ndarray, start: int, stop: int) -> float:
    if stop - start < 2:
        return 0.0
    return float(ts[stop - 1] - ts[start]) / 1000.0


def _angular_speed(ts: np.ndarray, world: np.ndarray, up: np.ndarray) -> np.ndarray:
    """``|d(trunk angle)/dt|`` in deg/s, on the uniform grid. Empty if unusable."""
    if world.shape[0] < 2:
        return np.empty(0)
    angles = trunk_from_vertical(world, up=up)
    if not np.isfinite(angles).all():
        return np.empty(0)
    _, uniform = resample_uniform(ts, angles)
    if uniform.size == 0:
        return np.empty(0)
    rate = smooth(uniform, deriv=1)
    return np.abs(rate) if np.isfinite(rate).all() else np.empty(0)


def _tip_speed(ts: np.ndarray, world: np.ndarray) -> np.ndarray:
    """Speed of the trunk-over-pelvis path in m/s. See the module docstring on its scope."""
    if world.shape[0] < 2:
        return np.empty(0)
    tip = trunk_vector(world)
    if not np.isfinite(tip).all():
        return np.empty(0)
    _, uniform = resample_uniform(ts, tip)
    if uniform.shape[0] == 0:
        return np.empty(0)
    v = smooth(uniform, deriv=1)
    if not np.isfinite(v).all():
        return np.empty(0)
    return np.linalg.norm(v, axis=1)


def _movement_duration(angular: np.ndarray) -> float:
    """Time from first to last crossing of 5% of peak angular speed, in seconds.

    The annotated span includes the annotator's reaction time at both ends; this is the
    part where Remy was actually moving.
    """
    if angular.size < 2 or not np.isfinite(angular).all():
        return float("nan")
    peak = float(np.max(angular))
    if peak <= 0:
        return float("nan")
    moving = np.flatnonzero(angular >= MOVEMENT_THRESHOLD_FRAC * peak)
    if moving.size < 2:
        return float("nan")
    return float(moving[-1] - moving[0]) / FS


def _leading_wrist(rec, start: int, stop: int, gate: Gate) -> Optional[str]:
    """Which wrist is further lateral (world x) at movement onset. Diagnostic only.

    A cheap check that a ``side=`` label matches what the video shows. It is not a
    measurement of which side led — that would need a reach-onset detector, and this is
    a single frame's geometry.
    """
    if stop <= start:
        return None
    wrists_ok = landmarks_ok(rec, WRISTS, gate)[start:stop]
    if not wrists_ok.any():
        return None
    first = start + int(np.flatnonzero(wrists_ok)[0])
    world = np.asarray(rec.landmarks_world)
    left_x, right_x = world[first, WRISTS[0], 0], world[first, WRISTS[1], 0]
    if not np.isfinite([left_x, right_x]).all() or left_x == right_x:
        return None
    return "left" if abs(left_x) > abs(right_x) else "right"
