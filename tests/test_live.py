"""Tests for ``motor_metrics.live`` — the rolling-window live metrics path.

Follows the discipline of ``test_motor_metrics.py``: pin **closed forms and
invariances**, not recorded outputs. Sway RMS of a sine is checked against
``amplitude / sqrt(2)``, crawl cadence against the driving frequency.

Three pins here earn their keep beyond ordinary coverage:

- **The LIVE_LAG identity.** At a lag of three samples the trailing-window value equals
  the offline whole-signal value *exactly*, and at the edge the velocity is ~100 % error.
  That measurement is the entire argument that a live number is the same measurement as
  an offline one, so both halves are asserted.
- **The never-mix rule.** Every ``LiveMetrics`` field is ``live_``-prefixed so a live row
  cannot be concatenated into an offline table. That is a structural guarantee and it is
  only structural for as long as something checks it.
- **Blanking after a dropout.** A stale number left on screen reads as a measurement of
  the child. Asserting it goes NaN is asserting the honesty property, not the plumbing.
"""

import time
import unittest
from dataclasses import fields, replace

import numpy as np

from motor_metrics.crawl import CrawlMetrics
from motor_metrics.derive import FS, resample_uniform, smooth, window_length
from motor_metrics.hold import HoldMetrics
from motor_metrics.live import (
    LIVE_LAG,
    MIN_COVERAGE,
    MODE_WINDOW_S,
    LiveMetrics,
    LiveMetricsComputer,
    LiveWindow,
    live_field_names,
    trunk_angle_now,
)
from motor_metrics.live_draw import (
    STEADINESS_TOL_DEG,
    _fmt,
    _quality_color,
    _rows,
    draw_live_metrics,
    sit_steadiness,
)
from motor_metrics.report import _LEAD_COLUMNS
from motor_metrics.transition import TransitionMetrics
from tests.fakes import NO_POSE, body_world, pose_result_from_row


def _sitting_world(n, *, amp_ml=0.03, amp_ap=0.02, f_ml=0.5, f_ap=0.3, trunk_len=0.30):
    """A synthetic seated body swaying sinusoidally over the pelvis."""
    t = np.arange(n) / FS
    trunk = np.stack(
        [amp_ml * np.sin(2 * np.pi * f_ml * t), -trunk_len + 0 * t, amp_ap * np.cos(2 * np.pi * f_ap * t)],
        axis=1,
    )
    return body_world(trunk)


def _crawling_world(n, *, cadence_hz=1.0, excursion=0.08,
                    leg_excursion_l=0.0, leg_excursion_r=0.0, leg_phase_frac=0.5):
    """A prone body whose wrists (and optionally knees) oscillate *along* the trunk axis.

    The axis matters: ``limb_signal`` projects the limb onto the trunk vector, so limbs
    swinging perpendicular to it produce no cycles however far they travel. Legs are still
    by default; drive them with ``leg_excursion_*`` to exercise the leg girdle.
    """
    t = np.arange(n) / FS
    trunk = np.stack([0 * t, -0.30 + 0 * t, 0 * t], axis=1)
    phase = 2 * np.pi * cadence_hz * t
    left = np.stack([-0.2 + 0 * t, -0.10 + excursion * np.sin(phase), 0 * t], axis=1)
    right = np.stack([0.2 + 0 * t, -0.10 + excursion * np.sin(phase + np.pi), 0 * t], axis=1)
    leg_phase = phase + 2 * np.pi * leg_phase_frac
    left_knee = np.stack([-0.1 + 0 * t, 0.15 + leg_excursion_l * np.sin(phase), 0 * t], axis=1)
    right_knee = np.stack([0.1 + 0 * t, 0.15 + leg_excursion_r * np.sin(leg_phase), 0 * t], axis=1)
    return body_world(trunk, left_wrist=left, right_wrist=right,
                      left_knee=left_knee, right_knee=right_knee)


def _run(computer, world, *, visibility=1.0, fps=FS, start_ms=0):
    """Push a whole synthetic session through a computer; return the last readout."""
    metrics = None
    for i, row in enumerate(world):
        ts = int(start_ms + i * 1000.0 / fps)
        result = (
            NO_POSE if np.isnan(row).all() else pose_result_from_row(row, visibility=visibility)
        )
        metrics = computer.push(ts, result)
    return metrics


class LiveWindowTests(unittest.TestCase):
    def test_partial_fill_exposes_only_pushed_frames(self):
        w = LiveWindow(10)
        self.assertEqual(len(w), 0)
        self.assertEqual(w.timestamps_ms.size, 0)
        for i in range(4):
            w.push(i * 33, NO_POSE)
        self.assertEqual(len(w), 4)
        self.assertEqual(w.landmarks_world.shape, (4, 33, 3))

    def test_wraparound_keeps_chronological_order_and_evicts_oldest(self):
        w = LiveWindow(10)
        for i in range(25):
            w.push(i * 33, NO_POSE)
        ts = w.timestamps_ms
        self.assertEqual(len(w), 10)
        self.assertTrue(np.all(np.diff(ts) > 0), "ring must read oldest-to-newest")
        self.assertEqual(int(ts[-1]), 24 * 33)
        self.assertEqual(int(ts[0]), 15 * 33)

    def test_pose_present_uses_the_whole_row_nan_convention(self):
        world = _sitting_world(4)
        w = LiveWindow(8)
        w.push(0, pose_result_from_row(world[0]))
        w.push(33, NO_POSE)
        w.push(66, pose_result_from_row(world[2]))
        self.assertEqual(list(w.pose_present), [True, False, True])

    def test_pose_present_is_true_for_extrapolated_landmarks(self):
        """The MediaPipe trap: a low-visibility frame still carries coordinates.

        ``pose_present`` is not a trust signal and must not be "fixed" into one — gating
        on what a metric reads is ``quality.landmarks_ok``'s job. Pinned here because a
        live buffer is where someone would be most tempted to conflate them.
        """
        world = _sitting_world(3)
        w = LiveWindow(8)
        for i, row in enumerate(world):
            w.push(i * 33, pose_result_from_row(row, visibility=0.1))
        self.assertTrue(w.pose_present.all())

        from motor_metrics.quality import TORSO, landmarks_ok

        self.assertFalse(landmarks_ok(w, TORSO).any())

    def test_window_span_is_selected_by_time_not_frame_count(self):
        w = LiveWindow(400)
        for i in range(300):  # 60 fps
            w.push(int(i * 1000 / 60), NO_POSE)
        span = w.window_span(2.0)
        ts = w.timestamps_ms
        self.assertAlmostEqual((ts[span.stop - 1] - ts[span.start]) / 1000.0, 2.0, delta=0.05)
        self.assertGreater(span.n_frames, 100, "60 fps must yield ~120 frames in 2 s")

    def test_annotations_are_always_empty(self):
        self.assertEqual(LiveWindow(4).annotations, [])

    def test_zero_capacity_rejected(self):
        with self.assertRaises(ValueError):
            LiveWindow(0)


class LiveLagTests(unittest.TestCase):
    """The measurement the whole module rests on."""

    @staticmethod
    def _signal(n=900, seed=0):
        rng = np.random.default_rng(seed)
        t = np.arange(n) / FS
        return 0.03 * np.sin(2 * np.pi * 0.5 * t) + rng.normal(0, 0.004, n)

    def test_live_lag_is_the_savgol_half_width(self):
        """LIVE_LAG is derived, not tuned: it is the filter's one-sided fit width."""
        self.assertEqual(LIVE_LAG, window_length() // 2)

    def test_trailing_window_at_live_lag_equals_offline_exactly(self):
        x = self._signal()
        offline = smooth(x)
        offline_v = smooth(x, deriv=1)
        width = int(5 * FS)
        for end in range(width, len(x), 37):
            live = smooth(x[end - width : end])
            live_v = smooth(x[end - width : end], deriv=1)
            self.assertAlmostEqual(
                live[-(LIVE_LAG + 1)], offline[end - 1 - LIVE_LAG], places=12
            )
            self.assertAlmostEqual(
                live_v[-(LIVE_LAG + 1)], offline_v[end - 1 - LIVE_LAG], places=12
            )

    def test_edge_velocity_is_far_worse_than_at_live_lag(self):
        """Guards against anyone "simplifying" LIVE_LAG to 0 to save 100 ms.

        The edge-extrapolated derivative's error is comparable to the signal's own
        velocity spread, i.e. it carries essentially no information.
        """
        x = self._signal()
        offline_v = smooth(x, deriv=1)
        width = int(5 * FS)
        edge_err, lag_err = [], []
        for end in range(width, len(x), 11):
            live_v = smooth(x[end - width : end], deriv=1)
            edge_err.append(live_v[-1] - offline_v[end - 1])
            lag_err.append(live_v[-(LIVE_LAG + 1)] - offline_v[end - 1 - LIVE_LAG])
        edge_rmse = float(np.sqrt(np.mean(np.square(edge_err))))
        velocity_sd = float(np.nanstd(offline_v))
        self.assertGreater(edge_rmse, 0.5 * velocity_sd, "edge error should rival the signal")
        self.assertAlmostEqual(float(np.max(np.abs(lag_err))), 0.0, places=12)

    def test_trunk_angle_now_reads_back_at_the_lag(self):
        from motor_metrics.segments import Span

        world = _sitting_world(200)
        rec = _FakeRec(world)
        now, baseline = trunk_angle_now(rec, Span(0, 200))
        self.assertTrue(np.isfinite(now) and np.isfinite(baseline))

        ts = np.asarray(rec.timestamps_ms)
        from motor_metrics.signals import trunk_from_vertical

        _, uniform = resample_uniform(ts, trunk_from_vertical(world))
        expected = smooth(uniform)[-(LIVE_LAG + 1)]
        self.assertAlmostEqual(now, float(expected), places=12)


class _FakeRec:
    """Minimal duck-typed recording for the direct-function tests."""

    def __init__(self, world, fps=FS):
        self.landmarks_world = np.asarray(world, dtype=np.float32)
        self.landmarks_norm = self.landmarks_world
        n = self.landmarks_world.shape[0]
        self.timestamps_ms = (np.arange(n) * (1000.0 / fps)).astype(np.int64)
        self.visibility = np.ones((n, 33), dtype=np.float32)
        self.presence = np.ones((n, 33), dtype=np.float32)
        self.pose_present = ~np.isnan(self.landmarks_world[:, 0, 0])
        self.annotations = []


class LiveMetricsComputerTests(unittest.TestCase):
    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            LiveMetricsComputer("transition")

    def test_cold_buffer_is_blank_and_does_not_raise(self):
        c = LiveMetricsComputer("hold")
        m = c.push(0, pose_result_from_row(_sitting_world(1)[0]))
        self.assertFalse(m.live_valid)
        self.assertTrue(np.isnan(m.live_sway_rms_m))
        self.assertEqual(m.live_n_frames, 1)

    def test_fully_untracked_session_is_blank(self):
        c = LiveMetricsComputer("hold")
        m = _run(c, np.full((200, 33, 3), np.nan))
        self.assertFalse(m.live_valid)
        self.assertEqual(m.live_coverage, 0.0)
        self.assertEqual(m.live_tracked_s, 0.0)
        self.assertTrue(np.isnan(m.live_sway_rms_m))

    def test_low_visibility_blanks_despite_pose_present(self):
        c = LiveMetricsComputer("hold")
        m = _run(c, _sitting_world(200), visibility=0.1)
        self.assertFalse(m.live_valid)
        self.assertEqual(m.live_coverage, 0.0)

    def test_hold_sway_matches_the_closed_form(self):
        """RMS of a sine of amplitude A is A/sqrt(2), on each axis independently."""
        amp_ml, amp_ap = 0.03, 0.02
        c = LiveMetricsComputer("hold")
        m = _run(c, _sitting_world(400, amp_ml=amp_ml, amp_ap=amp_ap))
        self.assertTrue(m.live_valid)
        self.assertAlmostEqual(m.live_sway_ml_rms_m, amp_ml / np.sqrt(2), places=3)
        self.assertAlmostEqual(m.live_sway_ap_rms_m, amp_ap / np.sqrt(2), places=3)
        self.assertAlmostEqual(
            m.live_sway_rms_m,
            np.sqrt((amp_ml**2 + amp_ap**2) / 2),
            places=3,
        )

    def test_hold_trunk_delta_is_referenced_to_the_window_baseline(self):
        """The tilt-robust readout: absolute lean may be large, the delta stays small."""
        c = LiveMetricsComputer("hold")
        m = _run(c, _sitting_world(400))
        self.assertTrue(m.live_valid)
        self.assertGreater(m.live_trunk_angle_baseline_deg, 1.0)
        self.assertAlmostEqual(
            m.live_trunk_angle_delta_deg,
            m.live_trunk_angle_deg - m.live_trunk_angle_baseline_deg,
            places=9,
        )
        self.assertLess(abs(m.live_trunk_angle_delta_deg), m.live_trunk_angle_baseline_deg)

    def test_crawl_cadence_matches_the_driving_frequency(self):
        c = LiveMetricsComputer("crawl")
        m = _run(c, _crawling_world(400, cadence_hz=1.0))
        self.assertTrue(m.live_valid)
        # A 1 Hz pull cycle is 60 cycles/min; the window boundary can clip one cycle
        # off a side, so allow a cycle's worth of slack on the pooled figure.
        self.assertAlmostEqual(m.live_cadence_cpm, 60.0, delta=10.0)
        self.assertLess(m.live_cycle_period_cv, 0.1, "a metronomic crawl has low CV")

    def test_crawl_reports_no_vertical_reference(self):
        """Crawl reads no `up`, which is what makes it survive a tilted camera."""
        m = _run(LiveMetricsComputer("crawl"), _crawling_world(300))
        self.assertEqual(m.live_up_source, "n/a")

    def test_live_crawl_reports_leg_cadence(self):
        c = LiveMetricsComputer("crawl")
        m = _run(c, _crawling_world(400, cadence_hz=1.0,
                                    leg_excursion_l=0.08, leg_excursion_r=0.08))
        self.assertAlmostEqual(m.live_leg_cadence_cpm, 60.0, delta=10.0)

    def test_live_crawl_flags_a_favored_leg(self):
        """Remy's signal: the left leg drives, the right barely moves. Must show live."""
        c = LiveMetricsComputer("crawl")
        m = _run(c, _crawling_world(400, cadence_hz=1.0,
                                    leg_excursion_l=0.08, leg_excursion_r=0.005))
        self.assertGreater(m.live_leg_amplitude_symmetry, 1.0)  # left favored
        self.assertGreater(m.live_leg_n_cycles_left, 0)
        self.assertEqual(m.live_leg_n_cycles_right, 0)

    def test_live_crawl_has_no_leg_reciprocity_field(self):
        """Leg phase offset stays offline (Hilbert edges), like the arm one."""
        self.assertNotIn("live_leg_phase_offset", live_field_names())

    def test_still_child_does_not_report_a_fictional_cadence(self):
        """MIN_CYCLE_EXCURSION_M, live. Jitter must not normalize up into a crawl."""
        rng = np.random.default_rng(3)
        t = np.arange(300) / FS
        trunk = np.stack([0 * t, -0.30 + 0 * t, 0 * t], axis=1)
        jitter = lambda sign: np.stack(  # noqa: E731
            [sign * 0.2 + rng.normal(0, 0.002, t.size),
             -0.10 + rng.normal(0, 0.002, t.size),
             rng.normal(0, 0.002, t.size)],
            axis=1,
        )
        world = body_world(trunk, left_wrist=jitter(-1), right_wrist=jitter(1))
        m = _run(LiveMetricsComputer("crawl"), world)
        self.assertEqual(m.live_n_cycles_left, 0)
        self.assertEqual(m.live_n_cycles_right, 0)
        self.assertTrue(np.isnan(m.live_cadence_cpm))

    def test_dropout_blanks_rather_than_leaving_a_stale_number(self):
        c = LiveMetricsComputer("hold")
        good = _run(c, _sitting_world(300))
        self.assertTrue(good.live_valid)
        after = _run(c, np.full((300, 33, 3), np.nan), start_ms=10_000)
        self.assertFalse(after.live_valid)
        self.assertTrue(np.isnan(after.live_sway_rms_m), "stale sway must not survive")

    def test_quality_refreshes_between_expensive_recomputes(self):
        """Coverage must track the video even on frames that reuse cached measurements."""
        c = LiveMetricsComputer("hold", recompute_every=1000)
        world = _sitting_world(300)
        for i, row in enumerate(world):
            m = c.push(int(i * 1000 / FS), pose_result_from_row(row))
        self.assertEqual(m.live_coverage, 1.0)
        self.assertGreater(m.live_n_frames, 100)

    def test_coverage_gate_is_the_blanking_threshold(self):
        """Half-tracked windows straddle MIN_COVERAGE; the gate must be the decider."""
        world = _sitting_world(400)
        world[::2] = np.nan  # every other frame untracked -> coverage ~0.5
        c = LiveMetricsComputer("hold", min_coverage=0.9)
        m = _run(c, world)
        self.assertLess(m.live_coverage, 0.9)
        self.assertFalse(m.live_valid)

    def test_window_length_defaults_per_mode(self):
        self.assertEqual(LiveMetricsComputer("hold").window_s, MODE_WINDOW_S["hold"])
        self.assertEqual(LiveMetricsComputer("crawl").window_s, MODE_WINDOW_S["crawl"])
        self.assertEqual(LiveMetricsComputer("hold", window_s=3.5).window_s, 3.5)


class NeverMixTests(unittest.TestCase):
    """Live values must not be concatenable into the offline table. See live.py."""

    def test_every_field_is_live_prefixed(self):
        names = live_field_names()
        self.assertTrue(names)
        for name in names:
            self.assertTrue(name.startswith("live_"), f"{name} breaks the never-mix rule")

    def test_live_fields_are_disjoint_from_offline_table_columns(self):
        offline = set(_LEAD_COLUMNS) | {"warnings"}
        for dc in (HoldMetrics, CrawlMetrics, TransitionMetrics):
            offline |= {f.name for f in fields(dc)}
        self.assertEqual(set(live_field_names()) & offline, set())

    def test_live_metrics_is_not_a_metric_dataclass(self):
        """A defensive pin: LiveMetrics must not accidentally gain offline field names."""
        self.assertNotIn("duration_s", live_field_names())
        self.assertNotIn("coverage", live_field_names())
        self.assertNotIn("rms_m", live_field_names())


class BudgetTests(unittest.TestCase):
    def test_push_stays_well_inside_a_frame(self):
        """Cost is not the constraint, and a regression here would make it one."""
        world = _sitting_world(400)
        for mode, w in (("hold", world), ("crawl", _crawling_world(400))):
            c = LiveMetricsComputer(mode)
            for i in range(200):  # prime the window
                c.push(int(i * 1000 / FS), pose_result_from_row(w[i]))
            start = time.perf_counter()
            for i in range(200, 400):
                c.push(int(i * 1000 / FS), pose_result_from_row(w[i]))
            per_push_ms = (time.perf_counter() - start) / 200 * 1000
            self.assertLess(per_push_ms, 16.0, f"{mode}: {per_push_ms:.2f} ms/push")


class LiveDrawTests(unittest.TestCase):
    def _blank(self, mode="hold"):
        return LiveMetricsComputer(mode).push(0, NO_POSE)

    def test_nan_renders_as_dashes_never_a_number(self):
        self.assertEqual(_fmt(float("nan")), "--")
        self.assertEqual(_fmt(None), "--")
        self.assertEqual(_fmt(0.12345, 3), "0.123")
        self.assertEqual(_fmt(np.int64(4)), "4")

    def test_blank_readout_draws_only_dashes_for_measurements(self):
        rows = _rows(self._blank("hold"))
        self.assertTrue(all("--" in value for _, value in rows), rows)

    def test_draw_does_not_raise_on_a_blank_readout(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        for mode in ("hold", "crawl"):
            draw_live_metrics(frame, self._blank(mode))
        self.assertTrue(frame.any(), "something should have been drawn")

    def test_overlay_text_is_ascii_only(self):
        """cv2's Hershey fonts have no glyphs past ASCII — a `°` draws as garbage.

        Not cosmetic: the character renders as noise rather than being dropped, so a
        readout with a stray Unicode label is unreadable rather than merely plain.
        """
        for mode in ("hold", "crawl"):
            for label, value in _rows(self._blank(mode)):
                self.assertTrue(label.isascii(), f"non-ASCII label {label!r}")
                self.assertTrue(value.isascii(), f"non-ASCII value {value!r}")

    def test_labels_do_not_collide_with_their_values(self):
        """`cycles L/R` is exactly 10 characters and once ran straight into its value."""
        from motor_metrics.live_draw import _LABEL_W

        for mode in ("hold", "crawl"):
            for label, _ in _rows(self._blank(mode)):
                self.assertLess(len(label), _LABEL_W, f"{label!r} needs a wider column")

    def test_draw_tolerates_missing_inputs(self):
        draw_live_metrics(None, self._blank())
        draw_live_metrics(np.zeros((10, 10, 3), np.uint8), None)

    def test_draw_mutates_the_frame_in_place(self):
        """Why the capture loop archives *before* drawing: this edits the caller's frame.

        ``rerun_viewer/main.py`` relies on it — the recorder writes the clean blurred
        frame first, so the HUD is never burned into the ``.h5``, where it would show
        numbers the offline metrics legitimately disagree with.
        """
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        before = frame.copy()
        draw_live_metrics(frame, self._blank())
        self.assertFalse(np.array_equal(frame, before))

    def test_coverage_is_always_shown_even_when_blanked(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        blank = self._blank()
        self.assertEqual(blank.live_coverage, 0.0)
        draw_live_metrics(frame, blank)  # must not raise; coverage explains the dashes
        self.assertLess(blank.live_coverage, MIN_COVERAGE)


class SitSteadinessTests(unittest.TestCase):
    """The sit-hold continuum. Pinned to its closed form and its honesty properties."""

    def _hold(self, delta_deg, *, valid=True):
        """A valid hold readout carrying a given trunk deviation from baseline."""
        blank = LiveMetricsComputer("hold").push(0, NO_POSE)
        return replace(blank, live_valid=valid, live_trunk_angle_delta_deg=delta_deg)

    def test_on_baseline_is_full(self):
        self.assertEqual(sit_steadiness(self._hold(0.0)), 1.0)

    def test_half_tolerance_is_half(self):
        self.assertAlmostEqual(sit_steadiness(self._hold(STEADINESS_TOL_DEG / 2)), 0.5)

    def test_at_or_past_tolerance_is_empty_never_negative(self):
        self.assertEqual(sit_steadiness(self._hold(STEADINESS_TOL_DEG)), 0.0)
        self.assertEqual(sit_steadiness(self._hold(3 * STEADINESS_TOL_DEG)), 0.0)

    def test_direction_of_lean_does_not_matter(self):
        """Deviation either side of the baseline is equally unsteady."""
        d = STEADINESS_TOL_DEG / 3
        self.assertEqual(sit_steadiness(self._hold(d)), sit_steadiness(self._hold(-d)))

    def test_blanked_or_nan_readout_draws_nothing(self):
        """None, not a stale bar — the blanking rule the whole overlay follows."""
        self.assertIsNone(sit_steadiness(self._hold(0.0, valid=False)))
        self.assertIsNone(sit_steadiness(self._hold(float("nan"))))
        self.assertIsNone(sit_steadiness(None))

    def test_only_hold_has_a_steadiness_meter(self):
        """Crawl reads no vertical at all; a lean bar would be meaningless there."""
        crawl = LiveMetricsComputer("crawl").push(0, NO_POSE)
        self.assertIsNone(sit_steadiness(replace(crawl, live_valid=True)))

    def test_color_runs_red_through_yellow_to_green(self):
        self.assertEqual(_quality_color(0.0), (0, 0, 255))    # red
        self.assertEqual(_quality_color(0.5), (0, 255, 255))  # yellow
        self.assertEqual(_quality_color(1.0), (0, 255, 0))    # green

    def test_meter_draws_only_for_a_valid_hold(self):
        """The bar is extra pixels: a valid hold must draw more than a blanked one."""
        good = np.zeros((240, 320, 3), dtype=np.uint8)
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        draw_live_metrics(good, self._hold(0.0))
        draw_live_metrics(blank, LiveMetricsComputer("hold").push(0, NO_POSE))
        self.assertGreater(int(good.astype(bool).sum()), int(blank.astype(bool).sum()))


if __name__ == "__main__":
    unittest.main()
