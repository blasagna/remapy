"""Live motor metrics over a trailing window, for feedback during a session.

Everything else in this package is offline by construction: :func:`report.metrics_table`
walks a human's annotations over a closed ``.h5``. This module runs the *same functions*
against a rolling buffer instead, so feedback is available in situations looser than the
data-collection protocol. It computes no metric of its own — it is a buffer, a window,
and a dispatch.

**Cost is not the constraint.** A full recompute over a trailing window, measured against
the 33 ms frame budget that MediaPipe already dominates::

    window        hold_metrics   crawl_metrics
     5 s (150 f)      2.89 ms        2.88 ms
    10 s (300 f)      4.52 ms        3.01 ms
    30 s (900 f)     10.89 ms        3.70 ms

At :data:`RECOMPUTE_EVERY` frames that is well under 1 % of the budget, so nothing here
needs an incremental algorithm and the window is simply recomputed whole.

**The constraint is that three samples at the end of any window are extrapolated.**
:func:`derive.window_length` is 7 at the shipped constants, so the Savitzky-Golay
interior fit needs three samples either side and the last three of *any* window are
fitted from one side only. Measured on a trailing 5 s window against the offline
whole-signal :func:`derive.smooth`, as a function of how far back the value is read::

    lag 0 (edge)    position RMSE 0.00228 m    velocity RMSE 0.0757 m/s
    lag 1 ( 33 ms)                0.00110 m                  0.0474 m/s
    lag 2 ( 67 ms)                0.00105 m                  0.0246 m/s
    lag 3 (100 ms)                0.000000                   0.000000

The test signal's velocity sd was 0.0695 m/s, so **the edge-extrapolated derivative is
essentially 100 % error**, while reading three samples back reproduces the offline value
*exactly* rather than approximating it. Hence :data:`LIVE_LAG`. 100 ms is imperceptible
as feedback and buys a number that is the same measurement the offline table reports.

That lag governs the **instantaneous** readouts. The **aggregates** (sway RMS, mean
velocity, cadence) are unaffected in a different way: they average over the whole window,
so the three edge samples are 3 of ~150 and dilute away. They are computed over the full
window and not trimmed — trimming would only move the edge, not remove it.

**Rolling anchors were measured and rejected.** A rolling window re-phases the resample
grid as it advances, which looks like it should add jitter. It does not measurably:
frame-to-frame variation in a rolling sway RMS is 0.143 % of the value with the default
anchor and 0.124 % with a fixed session epoch, with a *worse* maximum (0.63 % vs 0.76 %).
The variation is dominated by frames entering and leaving the window, not by grid phase.
So :mod:`motor_metrics.derive` keeps its no-per-call-knobs property and this module
passes nothing.

**What is deliberately not here.**

- **No movement detector.** :mod:`motor_metrics.hold` refuses to infer loss-of-posture
  from a trunk-angle threshold, because picking that threshold invents a clinical
  criterion and buries it in a constant. Without an annotator the only honest segment is
  a fixed rolling window, so that is what this uses — and it is why SPARC,
  ``count_submovements`` and ``movement_duration_s`` are absent. They need a movement's
  onset and offset, and nothing here is entitled to decide where those are.
- **No ``duration_s``.** It is the annotator's marks by definition.
- **No ``symmetry_index``.** It is a between-trial comparison by construction.

**Live values must never reach the offline table or the ``.h5``.** Different window, no
human-marked boundaries, and an instantaneous readout that is three samples old: the same
name would be a different measurement, and mixing them corrupts the cross-session trend
this package exists for. That is the derive-on-read rule of :mod:`motor_metrics.report`
pushed one step further, and it is enforced structurally rather than by convention —
**every** :class:`LiveMetrics` field is prefixed ``live_``, so no field of it can collide
with a ``metrics_table`` column and a live row cannot be concatenated into an offline
frame by accident. Keep that prefix on any field added later.
"""

from dataclasses import dataclass, fields, replace
from typing import Optional

import numpy as np

from recording.recorder import NUM_LANDMARKS, landmark_rows

from .crawl import crawl_metrics
from .derive import resample_uniform, smooth
from .hold import hold_metrics
from .quality import TORSO, Gate, coverage, landmarks_ok, longest_run
from .segments import Span
from .signals import WORLD_UP, trunk_from_vertical

#: Samples to read back from the end of a window for an instantaneous value. Three is
#: not a tuning choice: it is ``derive.window_length() // 2``, the half-width of the
#: Savitzky-Golay fit, and at exactly this lag the live value equals the offline one.
#: See the module docstring's measurement table.
LIVE_LAG = 3

#: Recompute the expensive window metrics every N frames (~6 Hz at a 30 Hz camera).
#: Cheap quality figures are refreshed on every frame regardless — they are what tells
#: the operator whether the framing is usable, and they must not lag behind the video.
RECOMPUTE_EVERY = 5

#: Default trailing window per mode, in seconds. A hold needs enough samples for a sway
#: statistic; a crawl needs several pull cycles before ``cycle_period_cv`` means
#: anything, and at a ~1 Hz cadence that is a longer window.
MODE_WINDOW_S = {"hold": 5.0, "crawl": 6.0}
MODES = tuple(MODE_WINDOW_S)

#: Below this fraction of trusted frames the readout is blanked rather than shown. A
#: **display** convention, not a validated threshold, and deliberately not a metrics
#: constant: it decides what an operator is shown in the moment, and changing it cannot
#: change any recorded number. A stale figure left on screen while tracking has failed is
#: worse than no figure, because it reads as a measurement of the child.
MIN_COVERAGE = 0.5

#: Ring-buffer sizing assumption. The buffer is sized in *frames* but the window is taken
#: in *time*, so this only has to be an upper bound on the camera's rate for the window
#: to hold its full duration; being generous costs a few hundred KB.
_CAPACITY_FPS = 60.0


class LiveWindow:
    """A fixed-capacity ring buffer that *is* a ``Recording`` as far as metrics care.

    The :mod:`motor_metrics` functions never open a file — they read attributes:
    ``timestamps_ms``, ``landmarks_world``, ``landmarks_norm``, ``visibility``,
    ``presence``, ``pose_present``, ``annotations``. That is exactly the surface
    ``tests.fakes.fake_recording`` provides, and it is what lets a live buffer feed
    ``hold_metrics``/``crawl_metrics`` unmodified instead of a parallel implementation
    that would drift from the offline one.

    ``push`` routes through :func:`recording.recorder.landmark_rows`, the same function
    ``HDF5Recorder.append`` uses, so a live frame and a recorded frame are converted
    identically — including the full-NaN convention that ``pose_present`` reads.

    ``annotations`` is always empty: there is no annotator live, which is the whole
    reason this module windows instead of segmenting.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be positive, got {capacity}.")
        self._cap = int(capacity)
        self._ts = np.zeros(self._cap, dtype=np.int64)
        self._world = np.zeros((self._cap, NUM_LANDMARKS, 3), dtype=np.float32)
        self._norm = np.zeros((self._cap, NUM_LANDMARKS, 3), dtype=np.float32)
        self._vis = np.zeros((self._cap, NUM_LANDMARKS), dtype=np.float32)
        self._pres = np.zeros((self._cap, NUM_LANDMARKS), dtype=np.float32)
        self._n = 0  # total ever pushed; the write cursor is self._n % cap
        self._cache: dict[str, np.ndarray] = {}
        self._cache_gen = -1

    def push(self, timestamp_ms: int, result) -> None:
        """Append one frame's pose result, evicting the oldest when full."""
        i = self._n % self._cap
        self._norm[i], self._world[i], self._vis[i], self._pres[i] = landmark_rows(result)
        self._ts[i] = int(timestamp_ms)
        self._n += 1

    def __len__(self) -> int:
        return min(self._n, self._cap)

    def _ordered(self, name: str, buf: np.ndarray) -> np.ndarray:
        """Oldest-to-newest view of one buffer, cached until the next push.

        The metrics read several of these attributes more than once per call
        (``landmarks_ok`` alone touches three), and a wrapped ring has to be
        concatenated to be read in order. Caching per push keeps that to one copy per
        buffer per frame. ``Recording.pose_present`` has the same shape of problem
        offline — it re-slices the whole ``(N, 33, 3)`` array on every access, which
        ``annotate/main.py`` caches around by hand.
        """
        if self._cache_gen != self._n:
            self._cache.clear()
            self._cache_gen = self._n
        hit = self._cache.get(name)
        if hit is not None:
            return hit
        if self._n < self._cap:
            out = buf[: self._n]
        else:
            cursor = self._n % self._cap
            out = np.concatenate((buf[cursor:], buf[:cursor]))
        self._cache[name] = out
        return out

    @property
    def timestamps_ms(self) -> np.ndarray:
        return self._ordered("ts", self._ts)

    @property
    def landmarks_world(self) -> np.ndarray:
        return self._ordered("world", self._world)

    @property
    def landmarks_norm(self) -> np.ndarray:
        return self._ordered("norm", self._norm)

    @property
    def visibility(self) -> np.ndarray:
        return self._ordered("vis", self._vis)

    @property
    def presence(self) -> np.ndarray:
        return self._ordered("pres", self._pres)

    @property
    def pose_present(self) -> np.ndarray:
        """Frames carrying a pose, by the same whole-row NaN check ``Recording`` uses.

        ``landmark_rows`` writes a full NaN row when nothing was detected, so a NaN
        landmark-0 x implies the whole row is NaN. This is not a visibility test —
        MediaPipe *extrapolates* occluded landmarks rather than dropping them, and those
        frames pass here while carrying invented coordinates. Gating on what a metric
        actually reads is :mod:`motor_metrics.quality`'s job.
        """
        return ~np.isnan(self._ordered("world", self._world)[:, 0, 0])

    @property
    def annotations(self) -> list:
        return []

    def window_span(self, window_s: float) -> Span:
        """The ``[start, stop)`` covering the last ``window_s`` seconds of the buffer.

        Selected by *time*, not by a frame count, so the window keeps its duration
        whatever rate the camera actually delivers. Mirrors
        :func:`motor_metrics.segments.frame_span`, which does the same searchsorted for
        an annotation's interval — the difference being that this boundary is a clock
        edge and that one is a human's judgment.
        """
        ts = self.timestamps_ms
        if ts.size == 0:
            return Span(0, 0)
        cutoff = ts[-1] - window_s * 1000.0
        return Span(int(np.searchsorted(ts, cutoff, side="left")), int(ts.size))


@dataclass(frozen=True)
class LiveMetrics:
    """One live readout. **Every field is prefixed ``live_``** — see the module docstring.

    The prefix is the structural half of the never-mix rule: it guarantees no field here
    can collide with a ``metrics_table`` column, so these rows cannot be concatenated
    into an offline frame even by a careless caller. It is pinned by a test.

    Fields are the union across modes, the same choice ``report.metrics_table`` makes: a
    hold readout carries NaN in the crawl fields and vice versa, so a consumer has one
    shape to render rather than two.

    Numeric fields are NaN whenever ``live_valid`` is False. That is the blanking rule —
    a stale number on screen during a tracking dropout reads as a measurement of the
    child, which is worse than showing nothing.
    """

    live_mode: str
    live_window_s: float
    live_valid: bool

    # Quality. Refreshed every frame, and the precondition for everything below.
    live_n_frames: int
    live_coverage: float
    live_tracked_s: float

    # Hold: sway of the trunk over the pelvis. `path_length_m` has no live counterpart —
    # it is duration-confounded, and a fixed window makes the velocity form the right one
    # for free. `ellipse_area_m2` is dropped too: it reads ~0 for one-axis rocking, so
    # the ML/AP split beside the radial RMS is the honest presentation.
    live_sway_rms_m: float
    live_sway_ml_rms_m: float  # image plane; measured well
    live_sway_ap_rms_m: float  # inferred depth; markedly noisier
    live_sway_velocity_mps: float

    # Trunk lean. `live_trunk_angle_delta_deg` is the one to display: an absolute lean
    # inherits WORLD_UP's level-camera assumption, which is exactly what a phone propped
    # at a random angle breaks. Referencing the window's own median moves a tilted camera
    # into the baseline instead of into the number.
    live_trunk_angle_deg: float  # instantaneous, LIVE_LAG samples back
    live_trunk_angle_baseline_deg: float  # window median
    live_trunk_angle_delta_deg: float
    live_up_source: str

    # Crawl. Reads no vertical at all (the axis is the body's own trunk vector), which
    # makes it the most camera-robust thing here. `speed_norm_per_s` is excluded: it is
    # image-widths per second and only comparable at fixed framing. `phase_offset` too —
    # Hilbert's edge effects are worst at exactly a short trailing window's edges.
    live_cadence_cpm: float
    live_cadence_cpm_left: float
    live_cadence_cpm_right: float
    live_n_cycles_left: int
    live_n_cycles_right: int
    live_cycle_period_cv: float


def _blank(mode: str, window_s: float, n_frames: int, cov: float, tracked_s: float,
           up_source: str) -> LiveMetrics:
    """A readout with the quality figures filled in and every measurement NaN."""
    return LiveMetrics(
        live_mode=mode,
        live_window_s=window_s,
        live_valid=False,
        live_n_frames=n_frames,
        live_coverage=cov,
        live_tracked_s=tracked_s,
        live_sway_rms_m=float("nan"),
        live_sway_ml_rms_m=float("nan"),
        live_sway_ap_rms_m=float("nan"),
        live_sway_velocity_mps=float("nan"),
        live_trunk_angle_deg=float("nan"),
        live_trunk_angle_baseline_deg=float("nan"),
        live_trunk_angle_delta_deg=float("nan"),
        live_up_source=up_source,
        live_cadence_cpm=float("nan"),
        live_cadence_cpm_left=float("nan"),
        live_cadence_cpm_right=float("nan"),
        live_n_cycles_left=0,
        live_n_cycles_right=0,
        live_cycle_period_cv=float("nan"),
    )


def trunk_angle_now(rec, span, up: np.ndarray = WORLD_UP) -> tuple[float, float]:
    """``(instantaneous, baseline)`` trunk lean in degrees over ``span``.

    The instantaneous value is read :data:`LIVE_LAG` samples back from the end of the
    smoothed window, which is where it equals what the offline chain would report for
    that instant; the baseline is the window's median. Returns ``(nan, nan)`` when the
    window is too short to smooth or the angle is unusable, rather than raising.
    """
    ts = np.asarray(rec.timestamps_ms)[span.start : span.stop]
    world = np.asarray(rec.landmarks_world)[span.start : span.stop]
    if world.shape[0] < 2:
        return float("nan"), float("nan")

    angles = trunk_from_vertical(world, up=up)
    if not np.isfinite(angles).all():
        return float("nan"), float("nan")
    _, uniform = resample_uniform(ts, angles)
    if uniform.size == 0:
        return float("nan"), float("nan")
    smoothed = smooth(uniform)
    if not np.isfinite(smoothed).all() or smoothed.size <= LIVE_LAG:
        return float("nan"), float("nan")
    return float(smoothed[-(LIVE_LAG + 1)]), float(np.median(smoothed))


class LiveMetricsComputer:
    """Owns the rolling buffer and turns pushed frames into a :class:`LiveMetrics`.

    ``push`` is called once per camera frame from the capture loop and always returns a
    readout, so a caller has something to render every frame. The expensive window
    metrics are recomputed every :data:`RECOMPUTE_EVERY` frames and the previous values
    reused in between; the quality figures are recomputed every frame.

    Never raises on a cold, partial, or fully-untracked buffer — it returns a blanked
    readout instead. That is the same discipline the offline metrics keep for a
    mis-marked annotation, and it matters more live: this runs inside the capture loop,
    where an exception costs the session rather than one row of a table.
    """

    def __init__(
        self,
        mode: str = "hold",
        *,
        window_s: Optional[float] = None,
        up: np.ndarray = WORLD_UP,
        gate: Gate = Gate(),
        min_coverage: float = MIN_COVERAGE,
        recompute_every: int = RECOMPUTE_EVERY,
    ) -> None:
        if mode not in MODE_WINDOW_S:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}.")
        self.mode = mode
        self.window_s = MODE_WINDOW_S[mode] if window_s is None else float(window_s)
        self.up = np.asarray(up, dtype=np.float64)
        self.gate = gate
        self.min_coverage = float(min_coverage)
        self._recompute_every = max(1, int(recompute_every))
        # Sized for the window plus the smoothing tail, so a full window is always
        # available rather than one frame short of it.
        capacity = int(np.ceil(self.window_s * _CAPACITY_FPS)) + 2 * LIVE_LAG + 2
        self.window = LiveWindow(capacity)
        self._last: Optional[LiveMetrics] = None
        self._pushes = 0

    def push(self, timestamp_ms: int, result) -> LiveMetrics:
        """Add one frame and return the current readout."""
        self.window.push(timestamp_ms, result)
        self._pushes += 1

        span = self.window.window_span(self.window_s)
        ok = landmarks_ok(self.window, TORSO, self.gate)
        cov = coverage(ok, span.start, span.stop)
        tracked_s = _tracked_seconds(self.window.timestamps_ms, ok, span)
        up_source = "n/a" if self.mode == "crawl" else (
            "world_y" if np.allclose(self.up, WORLD_UP) else "custom"
        )
        blank = _blank(self.mode, self.window_s, span.n_frames, cov, tracked_s, up_source)

        if cov < self.min_coverage:
            # Blank rather than reuse: the last good value describes a moment that has
            # passed, and on screen it is indistinguishable from a current measurement.
            self._last = None
            return blank
        if self._last is not None and self._pushes % self._recompute_every != 0:
            # Reuse the measurements, but keep the freshly computed quality figures —
            # coverage is what tells the operator the readout is still trustworthy.
            return _with_quality(self._last, span.n_frames, cov, tracked_s)

        computed = self._compute(span, blank)
        self._last = computed
        return computed

    def _compute(self, span: Span, blank: LiveMetrics) -> LiveMetrics:
        """Dispatch to the offline metric for this mode. Returns ``blank`` on anything
        unusable, so a short or untracked window degrades instead of raising."""
        if span.n_frames < 2:
            return blank
        try:
            if self.mode == "hold":
                m = hold_metrics(self.window, span, up=self.up, gate=self.gate)
                angle_now, baseline = trunk_angle_now(self.window, span, up=self.up)
                out = _replace(
                    blank,
                    live_sway_rms_m=m.rms_m,
                    live_sway_ml_rms_m=m.sway_ml_rms_m,
                    live_sway_ap_rms_m=m.sway_ap_rms_m,
                    live_sway_velocity_mps=m.mean_velocity_mps,
                    live_trunk_angle_deg=angle_now,
                    live_trunk_angle_baseline_deg=baseline,
                    live_trunk_angle_delta_deg=angle_now - baseline,
                    live_up_source=m.up_source,
                )
            else:
                m = crawl_metrics(self.window, span, gate=self.gate)
                out = _replace(
                    blank,
                    live_cadence_cpm=m.cadence_cpm,
                    live_cadence_cpm_left=m.cadence_cpm_left,
                    live_cadence_cpm_right=m.cadence_cpm_right,
                    live_n_cycles_left=m.n_cycles_left,
                    live_n_cycles_right=m.n_cycles_right,
                    live_cycle_period_cv=m.cycle_period_cv,
                )
        except (ValueError, IndexError, FloatingPointError):
            # The offline metrics are written not to raise on degenerate input, and the
            # tests pin that. This is the belt to that braces: inside a capture loop an
            # unexpected exception costs the whole session, not one row of a table.
            return blank
        return _replace(out, live_valid=True)


def _replace(metrics: LiveMetrics, **changes) -> LiveMetrics:
    return replace(metrics, **changes)


def _with_quality(metrics: LiveMetrics, n_frames: int, cov: float, tracked_s: float) -> LiveMetrics:
    return _replace(
        metrics, live_n_frames=n_frames, live_coverage=cov, live_tracked_s=tracked_s
    )


def _tracked_seconds(ts: np.ndarray, ok: np.ndarray, span: Span) -> float:
    """Longest continuously-trusted stretch inside ``span``, in seconds.

    A data-quality figure, exactly as offline: it says how much of the window can be
    measured without bridging a dropout, and is not a claim that the child was doing
    anything for that long.
    """
    r0, r1 = longest_run(ok, span.start, span.stop)
    if r1 - r0 < 2:
        return 0.0
    return float(ts[r1 - 1] - ts[r0]) / 1000.0


def live_field_names() -> tuple[str, ...]:
    """Field names of :class:`LiveMetrics`. Used by the renderers and by the never-mix test."""
    return tuple(f.name for f in fields(LiveMetrics))
