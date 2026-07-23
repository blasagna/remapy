"""Static-posture holds: sitting arms-free (GMFM dim B) and supported standing (dim D).

One function serves both. A supported stand and an unsupported sit are the same
measurement problem — how long, and how steadily — and the label carries the difference.

**What replaces the ordinal score.** The GMFM sitting items form a ladder of duration
thresholds (3 s, 5 s, 20 s, 60 s). That ladder is a continuous variable someone binned,
and re-reading the bin number back out would throw away the very resolution this package
exists to recover. So the headline metric is simply the **uncensored duration**, and
underneath it the **sway** of the trunk over the pelvis — the sub-clinical signal that
should move while the bin number sits still.

**Who decides what a hold is.** The annotator does. ``duration_s`` is the marked trial
length, because the in- and out-points *are* a human's judgment of when sitting began and
ended. Nothing here tries to infer loss-of-posture from a trunk-angle threshold: picking
that threshold would be inventing a clinical criterion and burying it in a constant. The
code's job starts inside the marked trial. ``tracked_s`` is a **data-quality** figure —
the longest stretch of continuously trusted tracking — and is not a claim about sitting.

**Two traps this closes.**

- ``path_length_m`` grows with trial length, so a *worse* 20-second hold beats a *better*
  8-second one on it. ``mean_velocity_mps`` is the duration-free form; ``window_s``
  truncates every trial to a common prefix; ``duration_s`` sits next to both so a
  length-confounded comparison is visible rather than inferred.
- ``hands_low_frac`` is **not** an arms-free detector. Whether a hand bears weight is a
  force question and there is no force sensor here. It is a QC flag for "your
  ``arms=free`` label may be wrong", and arms-free remains the annotator's assertion.
"""

from dataclasses import dataclass

import numpy as np

from .derive import resample_uniform, smooth
from .quality import TORSO, WRISTS, Gate, coverage, landmarks_ok, longest_run
from .signals import WORLD_UP, project_horizontal, trunk_from_vertical, trunk_vector

# Chi-square with 2 dof at p=0.95 — the standard posturography 95% sway ellipse.
_CHI2_95_2DOF = 5.991


@dataclass(frozen=True)
class HoldMetrics:
    """One static-hold trial. Distances in meters, angles in degrees, times in seconds."""

    duration_s: float  # the annotated trial: a human's call on when the hold began/ended
    tracked_s: float  # longest continuously-trusted run; a data-quality figure, not a hold
    coverage: float  # fraction of the trial with trusted torso landmarks
    n_frames: int

    # Sway of the trunk over the pelvis, in the plane perpendicular to `up`.
    path_length_m: float  # confounded with duration -- compare only at equal window_s
    mean_velocity_mps: float  # path_length / tracked_s; the duration-free form
    # 95% confidence ellipse of the 2D sway cloud. Read it next to the ML/AP split
    # below, never alone: a trunk rocking on a single axis traces a line, which
    # encloses no area, so this reads ~0 however hard it is rocking.
    ellipse_area_m2: float
    rms_m: float  # radial RMS excursion from the mean posture
    sway_ml_rms_m: float  # medio-lateral: in the image plane, measured well
    sway_ap_rms_m: float  # antero-posterior: inferred depth, markedly noisier

    trunk_angle_mean_deg: float  # unsigned lean from `up`; 0 = upright
    trunk_angle_sd_deg: float
    trunk_angle_range_deg: float

    hands_low_frac: float  # QC DIAGNOSTIC ONLY -- see the module docstring
    up_source: str  # which vertical was used; "world_y" assumes a level camera


def hold_metrics(
    rec,
    seg,
    *,
    up: np.ndarray = WORLD_UP,
    gate: Gate = Gate(),
    window_s: float | None = None,
) -> HoldMetrics:
    """Metrics for one ``sit_hold`` / ``stand_hold`` segment.

    Sway is computed over the longest continuously-tracked run inside the trial, never
    across a tracking gap — bridging one would invent the movement that happened inside
    it. Pass ``window_s`` to truncate that run to a common prefix so trials of unequal
    length can be compared on ``path_length_m``.

    Returns NaN sway fields (rather than raising) for trials that are empty, untracked,
    or shorter than the smoothing window. A mis-marked two-frame annotation is a normal
    thing to find in a session and must not take down a report with 40 good rows in it.
    """
    ts = np.asarray(rec.timestamps_ms)
    world = np.asarray(rec.landmarks_world)

    ok = landmarks_ok(rec, TORSO, gate)
    cov = coverage(ok, seg.start, seg.stop)
    duration_s = _span_seconds(ts, seg.start, seg.stop)

    r0, r1 = longest_run(ok, seg.start, seg.stop)
    if window_s is not None and r1 > r0:
        r1 = _clip_to_window(ts, r0, r1, window_s)
    tracked_s = _span_seconds(ts, r0, r1)

    sway = _sway(ts[r0:r1], world[r0:r1], up, tracked_s)
    angle = _trunk_angle_stats(world[r0:r1], up)

    return HoldMetrics(
        duration_s=duration_s,
        tracked_s=tracked_s,
        coverage=cov,
        n_frames=seg.stop - seg.start,
        **sway,
        **angle,
        hands_low_frac=_hands_low_frac(rec, r0, r1, up, gate),
        up_source="world_y" if np.allclose(up, WORLD_UP) else "custom",
    )


def sway_ellipse_area(points: np.ndarray) -> float:
    """95% confidence ellipse area of a 2D cloud; ``(N, 2)`` in, m^2 out.

    ``chi2(2, 0.95) * pi * sqrt(l1 * l2)`` over the covariance eigenvalues — the
    standard posturography figure. A perfectly collinear cloud has a zero eigenvalue and
    so zero area, which is correct: it encloses no region.
    """
    p = np.asarray(points, dtype=np.float64)
    if p.shape[0] < 3 or not np.isfinite(p).all():
        return float("nan")
    eigenvalues = np.linalg.eigvalsh(np.cov(p, rowvar=False))
    # Clamp: a degenerate cloud can produce a tiny negative eigenvalue from rounding.
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    return float(_CHI2_95_2DOF * np.pi * np.sqrt(eigenvalues[0] * eigenvalues[1]))


def path_length(points: np.ndarray) -> float:
    """Total distance travelled along a ``(N, K)`` path."""
    p = np.asarray(points, dtype=np.float64)
    if p.shape[0] < 2 or not np.isfinite(p).all():
        return float("nan")
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def _span_seconds(ts: np.ndarray, start: int, stop: int) -> float:
    if stop - start < 2:
        return 0.0
    return float(ts[stop - 1] - ts[start]) / 1000.0


def _clip_to_window(ts: np.ndarray, start: int, stop: int, window_s: float) -> int:
    """Shrink ``[start, stop)`` to the first ``window_s`` seconds of it."""
    cutoff = ts[start] + window_s * 1000.0
    return int(np.searchsorted(ts[start:stop], cutoff, side="right")) + start


def _sway(ts: np.ndarray, world: np.ndarray, up: np.ndarray, tracked_s: float) -> dict:
    """Sway of the trunk-over-pelvis path, on the pinned filter chain."""
    nan = dict(
        path_length_m=float("nan"),
        mean_velocity_mps=float("nan"),
        ellipse_area_m2=float("nan"),
        rms_m=float("nan"),
        sway_ml_rms_m=float("nan"),
        sway_ap_rms_m=float("nan"),
    )
    if world.shape[0] < 2:
        return nan

    horizontal = project_horizontal(trunk_vector(world), up=up)
    _, uniform = resample_uniform(ts, horizontal)
    if uniform.shape[0] == 0:
        return nan
    smoothed = smooth(uniform)  # NaN when the run is shorter than the window
    if not np.isfinite(smoothed).all():
        return nan

    centred = smoothed - smoothed.mean(axis=0)
    length = path_length(smoothed)
    return dict(
        path_length_m=length,
        mean_velocity_mps=length / tracked_s if tracked_s > 0 else float("nan"),
        ellipse_area_m2=sway_ellipse_area(smoothed),
        rms_m=float(np.sqrt((centred**2).sum(axis=1).mean())),
        sway_ml_rms_m=float(np.sqrt((centred[:, 0] ** 2).mean())),
        sway_ap_rms_m=float(np.sqrt((centred[:, 1] ** 2).mean())),
    )


def _trunk_angle_stats(world: np.ndarray, up: np.ndarray) -> dict:
    if world.shape[0] == 0:
        return dict(
            trunk_angle_mean_deg=float("nan"),
            trunk_angle_sd_deg=float("nan"),
            trunk_angle_range_deg=float("nan"),
        )
    angles = trunk_from_vertical(world, up=up)
    finite = angles[np.isfinite(angles)]
    if finite.size == 0:
        return dict(
            trunk_angle_mean_deg=float("nan"),
            trunk_angle_sd_deg=float("nan"),
            trunk_angle_range_deg=float("nan"),
        )
    return dict(
        trunk_angle_mean_deg=float(finite.mean()),
        trunk_angle_sd_deg=float(finite.std()),
        trunk_angle_range_deg=float(finite.max() - finite.min()),
    )


def _hands_low_frac(rec, start: int, stop: int, up: np.ndarray, gate: Gate) -> float:
    """Fraction of frames with either wrist below the pelvis, along ``up``.

    A hint that an ``arms=free`` label may be wrong, nothing more: a hand can rest low
    without bearing weight, and can bear weight without being low. Gated on the wrists
    separately from the torso, so a trial stays measurable when the hands are out of
    frame — this returns NaN there instead of poisoning the whole row.
    """
    if stop <= start:
        return float("nan")
    wrists_ok = landmarks_ok(rec, WRISTS, gate)[start:stop]
    if not wrists_ok.any():
        return float("nan")

    world = np.asarray(rec.landmarks_world)[start:stop]
    up_unit = np.asarray(up, dtype=np.float64)
    up_unit = up_unit / np.linalg.norm(up_unit)
    # Height above the pelvis (the world frame's origin) along `up`.
    heights = np.stack([world[:, i, :] @ up_unit for i in WRISTS], axis=1)
    return float(np.mean((heights[wrists_ok] < 0).any(axis=1)))
