"""Export the metrics kernel's behaviour as language-neutral JSON goldens.

The Android port reimplements :mod:`motor_metrics`' live path in Kotlin. Careful porting
is not evidence of agreement, so this writes what the Python chain actually produces —
inputs paired with outputs, at every level from ``savgol_filter`` up to a whole
``LiveMetricsComputer.push`` sequence — and the Kotlin tests assert against the same file.

**Why goldens and not just ported assertions.** ``tests/test_motor_metrics.py`` pins closed
forms (sway RMS against ``amplitude/sqrt(2)``, cadence against the driving frequency), and
those port directly and should be ported — they are the stronger tests, because they pin the
*intent*. But they do not catch a Savitzky-Golay edge handled differently, or scipy's
prominence walk implemented plausibly-but-differently, and those are exactly the two places
a reimplementation is most likely to diverge while still looking right. Goldens catch the
disagreement; the closed forms say which side is wrong.

**Regenerate with ``pixi run export-fixtures`` whenever a ``derive.py`` constant changes.**
That is not a formality: every number here is a function of ``FS``/``WINDOW_S``/``POLY``, and
a stale goldens file would pin the Kotlin port to a filter chain the Python side no longer
uses — the cross-language version of the drift ``derive.py``'s module docstring warns about.

Non-finite floats are encoded as the strings ``"nan"``/``"inf"``/``"-inf"``, because bare
``NaN`` is not valid JSON and Python's encoder emits it anyway. NaN is a load-bearing value
throughout this package (it is what every metric returns rather than raising), so it has to
survive the round trip rather than being dropped.
"""

import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from motor_metrics import crawl, derive, hold, live, quality, signals
from motor_metrics.segments import Span
from pose_estimation.angles import JOINT_TRIPLETS, angle_between
from pose_estimation.estimator import POSE_CONNECTIONS
from tests.fakes import body_world, fake_recording, pose_result, pose_result_from_row

#: Written into the Kotlin module's test resources, so `gradlew :metrics:test` finds it on
#: the classpath with no path wiring and no dependency on this repo's Python environment.
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "android/metrics/src/test/resources/goldens.json"

NUM_LANDMARKS = 33


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #

def enc(value):
    """Encode floats/arrays for JSON, with non-finite values as string tokens."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        if math.isnan(f):
            return "nan"
        if math.isinf(f):
            return "inf" if f > 0 else "-inf"
        return f
    if isinstance(value, np.ndarray):
        return enc(value.tolist())
    if isinstance(value, (list, tuple)):
        return [enc(v) for v in value]
    if isinstance(value, dict):
        return {str(k): enc(v) for k, v in value.items()}
    if is_dataclass(value):
        return enc(asdict(value))
    return value


def f32(value):
    """Shortest decimal that round-trips through **float32**, not float64.

    Landmark arrays are float32 on disk and in ``LiveWindow``, and the metrics widen them
    to float64 to compute. Emitting the widened double (``0.029999999329447746``) is exact
    but triples the file; emitting numpy's float32 repr (``0.03``) is *equally* exact once
    the Kotlin side stores it back into a ``Float``, which it does for the same reason.
    Precision is preserved by both sides doing float32 storage / float64 arithmetic — not
    by carrying digits neither implementation actually uses.
    """
    v = np.float32(value)
    if np.isnan(v):
        return "nan"
    if np.isinf(v):
        return "inf" if v > 0 else "-inf"
    return float(str(v))


def pack_points(arr):
    """Sparse-encode an ``(N, 33, 3)`` float32 landmark array.

    ``body_world`` places eight landmarks and leaves twenty-five at the origin, so a dense
    dump is ~75 % zeros repeated a few hundred times. Frames with no pose are a whole-row
    NaN by the ``landmark_rows`` convention, which means they need one index each rather
    than 99 NaN tokens.
    """
    a = np.asarray(arr, dtype=np.float32)
    n = a.shape[0]
    nan_frames = np.flatnonzero(np.isnan(a[:, 0, 0])).tolist()
    filled = np.nan_to_num(a, nan=0.0)
    kept = {}
    for i in range(a.shape[1]):
        if np.any(filled[:, i, :] != 0.0):
            kept[str(i)] = [[f32(v) for v in row] for row in filled[:, i, :]]
    return {"n": n, "nan_frames": nan_frames, "landmarks": kept}


def pack_scores(arr):
    """Sparse-encode an ``(N, 33)`` visibility/presence array against its dominant value."""
    a = np.asarray(arr, dtype=np.float32)
    n = a.shape[0]
    values, counts = np.unique(a[np.isfinite(a)], return_counts=True)
    fill = float(values[int(np.argmax(counts))]) if values.size else 1.0
    overrides = {}
    for i in range(a.shape[1]):
        col = a[:, i]
        if not np.all(col == np.float32(fill)):
            overrides[str(i)] = [f32(v) for v in col]
    return {"n": n, "fill": f32(fill), "overrides": overrides}


# --------------------------------------------------------------------------- #
# Signal builders — shared by several cases so the Kotlin side sees one shape
# --------------------------------------------------------------------------- #

def jittered_ms(n, fps=derive.FS, jitter_ms=4.0, seed=7):
    """A realistic camera timebase: nominally ``fps``, never exactly.

    ``resample_uniform`` exists because of this wobble, so feeding it a perfect grid would
    exercise the one input it was not written for.
    """
    rng = np.random.default_rng(seed)
    step = 1000.0 / fps
    return np.cumsum(np.round(step + rng.uniform(-jitter_ms, jitter_ms, n))).astype(np.int64)


def sine(n, freq_hz, amplitude, fps=derive.FS, phase=0.0, offset=0.0):
    t = np.arange(n) / fps
    return offset + amplitude * np.sin(2 * np.pi * freq_hz * t + phase)


def swaying_body(n=180, fps=derive.FS, amplitude=0.03, freq_hz=0.4, trunk_len=0.35):
    """A seated child swaying medio-laterally: trunk tipping side to side about vertical."""
    x = sine(n, freq_hz, amplitude, fps=fps)
    z = sine(n, freq_hz * 0.7, amplitude * 0.5, fps=fps, phase=1.1)
    trunk = np.stack([x, np.full(n, -trunk_len), z], axis=1)
    return body_world(trunk)


def crawling_body(n=200, fps=derive.FS, trunk_len=0.30, cadence_hz=0.9):
    """A prone belly-crawl: wrists and knees oscillating along the body axis, alternating.

    Arms are given a near-anti-phase relationship and legs a deliberately *asymmetric*
    amplitude — that is Remy's pattern per ``crawl.py``, and it is what makes
    ``leg_amplitude_symmetry`` a non-trivial number rather than a constant zero.
    """
    trunk = np.tile(np.array([0.0, -trunk_len, 0.0]), (n, 1))
    up = np.array([0.0, -1.0, 0.0])

    def limb(amp, phase, lateral):
        along = sine(n, cadence_hz, amp, fps=fps, phase=phase, offset=0.20)
        return np.stack([np.full(n, lateral), -along, np.zeros(n)], axis=1)

    return body_world(
        trunk,
        left_wrist=limb(0.09, 0.0, 0.12),
        right_wrist=limb(0.085, np.pi, -0.12),
        left_knee=limb(0.07, 0.4, 0.08),
        right_knee=limb(0.035, np.pi + 0.4, -0.08),
    ), up


def as_result(world_row, norm_row=None):
    return pose_result_from_row(world_row, norm_row)


# --------------------------------------------------------------------------- #
# Cases
# --------------------------------------------------------------------------- #

def case_constants():
    """The module constants. A Kotlin port that silently retunes one is the failure mode."""
    return {
        "FS": derive.FS,
        "WINDOW_S": derive.WINDOW_S,
        "POLY": derive.POLY,
        "window_length": derive.window_length(),
        "LIVE_LAG": live.LIVE_LAG,
        "RECOMPUTE_EVERY": live.RECOMPUTE_EVERY,
        "MIN_COVERAGE": live.MIN_COVERAGE,
        "MODE_WINDOW_S": live.MODE_WINDOW_S,
        "CYCLE_PROMINENCE_FRAC": crawl.CYCLE_PROMINENCE_FRAC,
        "MIN_CYCLE_EXCURSION_M": crawl.MIN_CYCLE_EXCURSION_M,
        "WORLD_UP": signals.WORLD_UP,
        "gate_min_visibility": quality.Gate().min_visibility,
        "gate_min_presence": quality.Gate().min_presence,
        "TORSO": list(quality.TORSO),
        "ARMS": list(quality.ARMS),
        "LEGS": list(quality.LEGS),
        "WRISTS": list(quality.WRISTS),
        "KNEES": list(quality.KNEES),
        "ARM_LANDMARKS": list(crawl.ARM_LANDMARKS),
        "LEG_LANDMARKS": list(crawl.LEG_LANDMARKS),
        "JOINT_TRIPLETS": {k: list(v) for k, v in JOINT_TRIPLETS.items()},
        # The skeleton edge list. Data, not drawing logic — `recording/recorder.py` persists it
        # into `meta/pose_connections` precisely so a consumer can draw a pose without pulling in
        # MediaPipe for a 35-pair constant. Exported so the Kotlin copy is checked, not eyeballed.
        "POSE_CONNECTIONS": [list(p) for p in POSE_CONNECTIONS],
        "live_field_names": list(live.live_field_names()),
    }


def case_window_length():
    return [
        {"fs": fs, "window_s": w, "poly": p, "expected": derive.window_length(fs, w, p)}
        for fs, w, p in [
            (30.0, 0.25, 2), (30.0, 0.2, 2), (30.0, 0.233, 2),
            (60.0, 0.25, 2), (30.0, 0.05, 2), (30.0, 0.25, 3), (10.0, 0.01, 4),
            # The 15 Hz grid, and the trap beside it. (15.0, 0.25, 2) -> 3, and a
            # 3-sample/order-2 Savitzky-Golay fit is an exact interpolating polynomial:
            # the filter becomes the identity and stops filtering with no error anywhere.
            # Pinned in both languages so the collapse is a *tested* fact rather than
            # something the next person rediscovers the hard way.
            (15.0, 0.35, 2), (15.0, 0.25, 2), (15.0, 0.5, 2),
        ]
    ]


def case_savgol():
    """``smooth`` at deriv 0 and 1, on 1D and 2D input, including the short-input NaN path.

    The **edges** are the point of this case. scipy's default ``mode="interp"`` fits a
    polynomial to the first/last ``window_length`` samples rather than padding, and
    ``LIVE_LAG`` is defined by exactly that behaviour — a port that pads instead will agree
    everywhere except the last three samples, which is where the live readout is taken.
    """
    cases = []
    ramp = np.arange(20, dtype=np.float64) * 0.1
    noisy = sine(64, 1.0, 0.05) + sine(64, 6.0, 0.004, phase=0.3)
    two_col = np.stack([noisy, sine(64, 0.5, 0.02, phase=2.0)], axis=1)

    for name, x, deriv in [
        ("ramp_deriv0", ramp, 0),
        ("ramp_deriv1", ramp, 1),
        ("noisy_deriv0", noisy, 0),
        ("noisy_deriv1", noisy, 1),
        ("two_col_deriv0", two_col, 0),
        ("two_col_deriv1", two_col, 1),
        # Sized off `window_length()` rather than the literal it happens to equal: these two
        # cases exist to straddle the short-input NaN boundary, and a hardcoded 7/6 would
        # quietly stop straddling anything the moment a `derive.py` constant moved.
        ("exactly_window", np.arange(derive.window_length(), dtype=np.float64) ** 2, 0),
        ("shorter_than_window", np.arange(derive.window_length() - 1, dtype=np.float64), 0),
        ("empty", np.empty(0), 0),
    ]:
        cases.append({
            "name": name,
            "input": x,
            "deriv": deriv,
            "expected": derive.smooth(x, deriv=deriv),
        })
    return cases


def case_resample_uniform():
    cases = []
    for name, ts, x in [
        ("uniform_1d", jittered_ms(40, jitter_ms=0.0), sine(40, 0.8, 0.05)),
        ("jittered_1d", jittered_ms(40), sine(40, 0.8, 0.05)),
        ("jittered_2d", jittered_ms(50), np.stack([sine(50, 0.8, 0.05), sine(50, 0.3, 0.02)], axis=1)),
        ("two_samples", np.array([0, 100], dtype=np.int64), np.array([1.0, 2.0])),
        ("single_sample", np.array([0], dtype=np.int64), np.array([1.0])),
        ("sub_frame_span", np.array([0, 5], dtype=np.int64), np.array([1.0, 2.0])),
        ("with_nan", jittered_ms(20), np.where(np.arange(20) == 10, np.nan, sine(20, 0.8, 0.05))),
    ]:
        t_s, out = derive.resample_uniform(ts, x)
        cases.append({"name": name, "t_ms": ts, "x": x, "t_s": t_s, "expected": out})
    return cases


def case_find_peaks():
    """Raw ``scipy.signal.find_peaks(v, prominence=p)``, isolated from ``cycles``.

    Prominence is the subtle part: for each peak scipy walks left and right to the nearest
    strictly-higher sample and takes the *higher* of the two intervening minima as the base.
    Plateaus, monotone runs and repeated values are where a plausible reimplementation
    diverges, so they are all here.
    """
    rng = np.random.default_rng(3)
    cases = []
    for name, v, prom in [
        ("clean_sine", sine(90, 1.0, 1.0), 0.2),
        ("two_scales", sine(120, 0.5, 1.0) + sine(120, 4.0, 0.15), 0.2),
        ("plateau", np.array([0.0, 1, 2, 2, 2, 1, 0, 3, 3, 0]), 0.5),
        ("monotone", np.arange(10, dtype=np.float64), 0.1),
        ("flat", np.zeros(10), 0.0),
        ("edge_peaks", np.array([5.0, 1, 4, 1, 5]), 0.5),
        ("repeated_maxima", np.array([0.0, 2, 0, 2, 0, 2, 0]), 1.0),
        ("noise", rng.normal(0, 1, 120), 0.5),
        ("high_prominence_gate", sine(90, 1.0, 1.0), 5.0),
        ("two_points", np.array([0.0, 1.0]), 0.1),
    ]:
        peaks, _ = find_peaks(v, prominence=prom)
        cases.append({"name": name, "signal": v, "prominence": prom, "expected": peaks})
    return cases


def case_cycles():
    """``crawl.cycles`` — the two gates together, including the pure-jitter regression."""
    rng = np.random.default_rng(11)
    cases = []
    for name, v in [
        ("crawl_like", sine(120, 0.9, 0.08)),
        ("pure_jitter", rng.normal(0, 0.004, 200)),  # must yield zero cycles
        ("just_under_excursion", sine(120, 0.9, 0.0099)),
        ("just_over_excursion", sine(120, 0.9, 0.0101)),
        ("with_nan", np.where(np.arange(60) == 5, np.nan, sine(60, 0.9, 0.08))),
        ("too_short", np.array([0.0, 1.0])),
        ("flat", np.zeros(50)),
    ]:
        cases.append({"name": name, "signal": v, "expected": crawl.cycles(v)})
    return cases


def case_angles():
    cases = []
    for name, a, j, c in [
        ("right_angle", [1.0, 0, 0], [0.0, 0, 0], [0.0, 1, 0]),
        ("straight", [1.0, 0, 0], [0.0, 0, 0], [-1.0, 0, 0]),
        ("coincident", [1.0, 0, 0], [0.0, 0, 0], [1.0, 0, 0]),
        ("degenerate", [0.0, 0, 0], [0.0, 0, 0], [1.0, 0, 0]),
        ("oblique", [0.4, 0.9, -0.2], [0.1, 0.1, 0.1], [-0.3, 0.5, 0.8]),
    ]:
        cases.append({
            "name": name, "a": a, "joint": j, "c": c,
            "expected": angle_between(np.array(a), np.array(j), np.array(c)),
        })
    return cases


def case_signals():
    world = swaying_body(n=60)
    tilted = np.array([0.15, -0.98, 0.05])
    norm = np.zeros_like(world)
    norm[:, 23] = np.stack([sine(60, 0.2, 0.1, offset=0.5), np.full(60, 0.6), np.zeros(60)], axis=1)
    norm[:, 24] = norm[:, 23] - np.array([0.05, 0.0, 0.0])
    return {
        "world": pack_points(world),
        "trunk_vector": signals.trunk_vector(world),
        "trunk_from_vertical_world_up": signals.trunk_from_vertical(world),
        "tilted_up": tilted,
        "trunk_from_vertical_tilted": signals.trunk_from_vertical(world, up=tilted),
        "project_horizontal_world_up": signals.project_horizontal(signals.trunk_vector(world)),
        "project_horizontal_tilted": signals.project_horizontal(signals.trunk_vector(world), up=tilted),
        "rolled_up": [1.0, 0.0, 0.0],
        "project_horizontal_rolled": signals.project_horizontal(
            signals.trunk_vector(world), up=np.array([1.0, 0.0, 0.0])
        ),
        "norm": pack_points(norm),
        "com_norm": signals.com_norm(norm),
    }


def case_quality():
    """A visibility pattern with a dropout, a low-visibility stretch, and a no-pose gap."""
    n = 60
    world = swaying_body(n=n)
    world[20:25] = np.nan  # no pose at all
    vis = np.ones((n, NUM_LANDMARKS), dtype=np.float32)
    vis[40:48, list(quality.TORSO)] = 0.3  # extrapolated: present but not trusted
    pres = np.ones((n, NUM_LANDMARKS), dtype=np.float32)
    pres[52:54, quality.TORSO[0]] = 0.1
    rec = fake_recording(world, visibility=vis, presence=pres)

    ok = quality.landmarks_ok(rec, quality.TORSO)
    spans = [(0, n), (0, 20), (18, 30), (35, 60), (10, 10), (55, 60)]
    return {
        "world": pack_points(world), "visibility": pack_scores(vis), "presence": pack_scores(pres),
        "timestamps_ms": rec.timestamps_ms,
        "pose_present": rec.pose_present,
        "landmarks_ok_torso": ok,
        "landmarks_ok_wrists": quality.landmarks_ok(rec, quality.WRISTS),
        "spans": [
            {
                "start": s, "stop": e,
                "coverage": quality.coverage(ok, s, e),
                "longest_run": list(quality.longest_run(ok, s, e)),
            }
            for s, e in spans
        ],
    }


def _recording_payload(rec):
    return {
        "timestamps_ms": rec.timestamps_ms,
        "world": pack_points(rec.landmarks_world),
        "norm": pack_points(rec.landmarks_norm),
        "visibility": pack_scores(rec.visibility),
        "presence": pack_scores(rec.presence),
    }


def case_hold_metrics():
    """Whole ``HoldMetrics`` dicts, over the trial shapes a real session produces."""
    cases = []

    clean = fake_recording(swaying_body(n=180), timestamps_ms=jittered_ms(180))
    cases.append({"name": "clean_sway", "rec": _recording_payload(clean),
                  "start": 0, "stop": 180, "window_s": None,
                  "expected": hold.hold_metrics(clean, Span(0, 180))})

    cases.append({"name": "clean_sway_windowed", "rec": _recording_payload(clean),
                  "start": 0, "stop": 180, "window_s": 3.0,
                  "expected": hold.hold_metrics(clean, Span(0, 180), window_s=3.0)})

    # A dropout in the middle: `longest_run` must pick a side, never bridge it.
    dropped_world = swaying_body(n=180)
    dropped_world[70:95] = np.nan
    dropped = fake_recording(dropped_world, timestamps_ms=jittered_ms(180))
    cases.append({"name": "mid_trial_dropout", "rec": _recording_payload(dropped),
                  "start": 0, "stop": 180, "window_s": None,
                  "expected": hold.hold_metrics(dropped, Span(0, 180))})

    # Low visibility throughout: coverage floors and every measurement is NaN.
    untrusted = fake_recording(swaying_body(n=90), visibility=0.2,
                               timestamps_ms=jittered_ms(90))
    cases.append({"name": "untrusted", "rec": _recording_payload(untrusted),
                  "start": 0, "stop": 90, "window_s": None,
                  "expected": hold.hold_metrics(untrusted, Span(0, 90))})

    short = fake_recording(swaying_body(n=5), timestamps_ms=jittered_ms(5))
    cases.append({"name": "shorter_than_window", "rec": _recording_payload(short),
                  "start": 0, "stop": 5, "window_s": None,
                  "expected": hold.hold_metrics(short, Span(0, 5))})

    cases.append({"name": "empty_span", "rec": _recording_payload(short),
                  "start": 2, "stop": 2, "window_s": None,
                  "expected": hold.hold_metrics(short, Span(2, 2))})

    # A tilted camera, so `up_source` flips to "custom" and the ML/AP basis rotates.
    tilted = np.array([0.15, -0.98, 0.05])
    cases.append({"name": "tilted_up", "rec": _recording_payload(clean),
                  "start": 0, "stop": 180, "window_s": None, "up": tilted,
                  "expected": hold.hold_metrics(clean, Span(0, 180), up=tilted)})

    # Wrists below the pelvis for part of the trial -> a non-trivial hands_low_frac.
    hands = swaying_body(n=120)
    hands[:, 15] = np.array([0.15, 0.10, 0.0])   # left wrist below origin along -up
    hands[:, 16] = np.array([-0.15, -0.10, 0.0])  # right wrist above
    hands_rec = fake_recording(hands, timestamps_ms=jittered_ms(120))
    cases.append({"name": "hands_low", "rec": _recording_payload(hands_rec),
                  "start": 0, "stop": 120, "window_s": None,
                  "expected": hold.hold_metrics(hands_rec, Span(0, 120))})
    return cases


def case_crawl_metrics():
    cases = []
    world, _ = crawling_body(n=200)
    norm = np.zeros_like(world)
    travel = np.linspace(0.2, 0.8, 200)
    norm[:, 23] = np.stack([travel, np.full(200, 0.7), np.zeros(200)], axis=1)
    norm[:, 24] = norm[:, 23]
    rec = fake_recording(world, norm=norm, timestamps_ms=jittered_ms(200))
    cases.append({"name": "alternating_crawl", "rec": _recording_payload(rec),
                  "start": 0, "stop": 200,
                  "expected": crawl.crawl_metrics(rec, Span(0, 200))})

    # A still child. The regression that matters: no cadence, not a fictional one.
    rng = np.random.default_rng(5)
    still_trunk = np.tile(np.array([0.0, -0.30, 0.0]), (150, 1))
    jitter = lambda: rng.normal(0, 0.004, (150, 3)) + np.array([0.1, -0.2, 0.0])
    still = body_world(still_trunk, left_wrist=jitter(), right_wrist=jitter(),
                       left_knee=jitter(), right_knee=jitter())
    still_rec = fake_recording(still, timestamps_ms=jittered_ms(150))
    cases.append({"name": "still_child", "rec": _recording_payload(still_rec),
                  "start": 0, "stop": 150,
                  "expected": crawl.crawl_metrics(still_rec, Span(0, 150))})

    # Legs trusted, one wrist not: the girdles must gate independently.
    vis = np.ones((200, NUM_LANDMARKS), dtype=np.float32)
    vis[60:140, 15] = 0.1
    split = fake_recording(world, norm=norm, visibility=vis, timestamps_ms=jittered_ms(200))
    cases.append({"name": "arm_occluded_legs_clear", "rec": _recording_payload(split),
                  "start": 0, "stop": 200,
                  "expected": crawl.crawl_metrics(split, Span(0, 200))})

    short = fake_recording(crawling_body(n=4)[0], timestamps_ms=jittered_ms(4))
    cases.append({"name": "too_short", "rec": _recording_payload(short),
                  "start": 0, "stop": 4,
                  "expected": crawl.crawl_metrics(short, Span(0, 4))})
    return cases


def case_limb_signal():
    world, _ = crawling_body(n=80)
    rec = fake_recording(world, timestamps_ms=jittered_ms(80))
    return {
        "rec": _recording_payload(rec),
        "start": 0, "stop": 80,
        "left_wrist": crawl.limb_signal(rec, Span(0, 80), "left", "wrist"),
        "right_wrist": crawl.limb_signal(rec, Span(0, 80), "right", "wrist"),
        "left_knee": crawl.limb_signal(rec, Span(0, 80), "left", "knee"),
        "right_knee": crawl.limb_signal(rec, Span(0, 80), "right", "knee"),
    }


def _push_sequence(mode, world, visibility, timestamps, *, window_s=None, sample_every=1):
    """Drive a real ``LiveMetricsComputer`` frame by frame and record every readout.

    Sampling every frame is deliberate: the interesting behaviour is *stateful* — the
    ``RECOMPUTE_EVERY`` reuse, the blanking when coverage falls, the ring wrapping once the
    buffer fills — and none of it shows in a single snapshot.
    """
    computer = live.LiveMetricsComputer(mode, window_s=window_s)
    readouts = []
    for i, ts in enumerate(timestamps):
        row = world[i]
        vis_row = visibility[i]
        if np.isnan(row).all():
            result = pose_result()  # exactly what MediaPipe emits when it finds nothing
        else:
            result = pose_result_from_row(row, None)
            for k, lm in enumerate(result.pose_landmarks[0]):
                lm.visibility = float(vis_row[k])
                lm.presence = float(vis_row[k])
        m = computer.push(int(ts), result)
        if i % sample_every == 0:
            readouts.append({"frame": i, "metrics": m})
    return readouts


def case_live_push():
    cases = []

    n = 240
    ts = jittered_ms(n)
    world = swaying_body(n=n)
    vis = np.ones((n, NUM_LANDMARKS), dtype=np.float32)
    cases.append({
        "name": "hold_clean", "mode": "hold", "window_s": None,
        "timestamps_ms": ts, "world": pack_points(world), "visibility": pack_scores(vis),
        "readouts": _push_sequence("hold", world, vis, ts),
    })

    # Coverage collapses mid-stream and then recovers. Both halves matter: the collapse
    # pins blanking, and the recovery pins that the readout comes back *recomputed* rather
    # than resurrecting the pre-dropout value. Sized around the 5 s window deliberately —
    # a 50-frame hole only drops coverage to ~0.79, which stays above MIN_COVERAGE and
    # exercises none of this, and the run has to continue a full window past the gap for
    # coverage to climb back to 1.0.
    n_drop = 380
    ts_drop = jittered_ms(n_drop, seed=31)
    dropped = swaying_body(n=n_drop)
    dropped[110:210] = np.nan
    vis_d = np.ones((n_drop, NUM_LANDMARKS), dtype=np.float32)
    cases.append({
        "name": "hold_dropout", "mode": "hold", "window_s": None,
        "timestamps_ms": ts_drop, "world": pack_points(dropped), "visibility": pack_scores(vis_d),
        "readouts": _push_sequence("hold", dropped, vis_d, ts_drop),
    })

    # Low visibility without a NaN row: `pose_present` is True while `landmarks_ok` is
    # False. The trap `quality.py` is written around.
    vis_low = np.ones((n, NUM_LANDMARKS), dtype=np.float32)
    vis_low[100:200, list(quality.TORSO)] = 0.2
    cases.append({
        "name": "hold_extrapolated", "mode": "hold", "window_s": None,
        "timestamps_ms": ts, "world": pack_points(world), "visibility": pack_scores(vis_low),
        "readouts": _push_sequence("hold", world, vis_low, ts),
    })

    crawl_world, _ = crawling_body(n=n)
    vis_c = np.ones((n, NUM_LANDMARKS), dtype=np.float32)
    cases.append({
        "name": "crawl_alternating", "mode": "crawl", "window_s": None,
        "timestamps_ms": ts, "world": pack_points(crawl_world), "visibility": pack_scores(vis_c),
        "readouts": _push_sequence("crawl", crawl_world, vis_c, ts),
    })

    # Long enough to wrap the ring buffer several times over a short window.
    long_n = 500
    ts_long = jittered_ms(long_n, seed=21)
    long_world = swaying_body(n=long_n)
    vis_long = np.ones((long_n, NUM_LANDMARKS), dtype=np.float32)
    cases.append({
        "name": "hold_ring_wrap", "mode": "hold", "window_s": 2.0,
        "timestamps_ms": ts_long, "world": pack_points(long_world), "visibility": pack_scores(vis_long),
        "readouts": _push_sequence("hold", long_world, vis_long, ts_long, window_s=2.0, sample_every=3),
    })
    return cases


def case_live_lag_identity():
    """The ``LIVE_LAG`` measurement, exported so the Kotlin port inherits the same pin.

    ``trunk_angle_now`` read over a trailing window must equal what the offline whole-signal
    chain reports for the same instant — that identity is the entire justification for
    reading three samples back instead of at the edge, and a port that pads the Savitzky-Golay
    edges will break it while every other case still passes.
    """
    n = 300
    ts = jittered_ms(n, seed=13)
    world = swaying_body(n=n)
    rec = fake_recording(world, timestamps_ms=ts)

    angles = signals.trunk_from_vertical(world)
    _, uniform = derive.resample_uniform(ts, angles)
    offline = derive.smooth(uniform)

    window = live.LiveWindow(4096)
    samples = []
    for i in range(n):
        window.push(int(ts[i]), pose_result_from_row(world[i]))
        if i >= 200 and i % 17 == 0:
            span = window.window_span(5.0)
            now, baseline = live.trunk_angle_now(window, span)
            samples.append({"frame": i, "trunk_angle_now": now, "baseline": baseline})
    return {
        "timestamps_ms": ts, "world": pack_points(world),
        "offline_smoothed_angles": offline,
        "samples": samples,
    }


def case_derivative_gain():
    """The documented gain-vs-frequency table. A bias, and it must be the *same* bias."""
    out = []
    for freq in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0):
        n = 600
        x = sine(n, freq, 1.0)
        analytic = 2 * np.pi * freq * np.cos(2 * np.pi * freq * np.arange(n) / derive.FS)
        measured = derive.smooth(x, deriv=1)
        interior = slice(20, n - 20)
        gain = float(np.std(measured[interior]) / np.std(analytic[interior]))
        out.append({"freq_hz": freq, "gain": gain})
    return out


# --------------------------------------------------------------------------- #

def build() -> dict:
    return {
        "_comment": "Generated by tests/fixtures/export.py (pixi run export-fixtures). Do not hand-edit.",
        "constants": case_constants(),
        "window_length": case_window_length(),
        "savgol": case_savgol(),
        "resample_uniform": case_resample_uniform(),
        "find_peaks": case_find_peaks(),
        "cycles": case_cycles(),
        "angle_between": case_angles(),
        "signals": case_signals(),
        "quality": case_quality(),
        "limb_signal": case_limb_signal(),
        "hold_metrics": case_hold_metrics(),
        "crawl_metrics": case_crawl_metrics(),
        "live_push": case_live_push(),
        "live_lag_identity": case_live_lag_identity(),
        "derivative_gain": case_derivative_gain(),
    }


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT,
                        help=f"where to write the goldens (default: {DEFAULT_OUT})")
    args = parser.parse_args(argv)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = enc(build())
    # Compact, not pretty-printed: indentation triples the size of arrays this shape, and
    # nothing reads this by eye — `git diff` on it is noise either way, which is why
    # `.gitattributes` marks it generated.
    text = json.dumps(payload, separators=(",", ":"), allow_nan=False, sort_keys=False)
    args.output.write_text(text + "\n")
    size_kb = args.output.stat().st_size / 1024
    print(f"Wrote {args.output} ({size_kb:.0f} KB, {len(payload) - 1} case groups)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
