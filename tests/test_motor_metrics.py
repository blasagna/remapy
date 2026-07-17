"""Tests for the :mod:`motor_metrics` package.

Pure logic runs unmocked against real numpy/scipy; nothing here needs a camera, model,
display, or network. Where a metric has a closed-form answer, the test asserts the
closed form rather than a recorded output — a test that pins an observed number pins
the noise along with it.
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from motor_metrics.labels import (
    DIMENSIONS,
    EXERCISES,
    format_label,
    label_warnings,
    parse_label,
)
from motor_metrics.crawl import crawl_metrics, cycles, limb_signal, phase_offset
from motor_metrics.derive import (
    FS,
    POLY,
    WINDOW_S,
    resample_uniform,
    smooth,
    velocity,
    window_length,
)
from motor_metrics.hold import hold_metrics, path_length, sway_ellipse_area
from motor_metrics.quality import TORSO, Gate, coverage, landmarks_ok, longest_run
from motor_metrics.report import TRIAL_EXERCISES, metrics_table, session_table
from motor_metrics.segments import Segment, frame_span, segments
from motor_metrics.transition import (
    count_submovements,
    sparc,
    symmetry_index,
    transition_metrics,
)
from motor_metrics.signals import (
    WORLD_UP,
    com_norm,
    estimate_up,
    mid,
    project_horizontal,
    trunk_from_vertical,
    trunk_vector,
)
from recording.annotations import Annotation, AnnotationStore
from recording.reader import Recording
from recording.recorder import HDF5Recorder
from tests.fakes import (
    FakeLandmark,
    body_world,
    fake_recording,
    make_landmarks,
    pose_result,
    solid_frame,
)

L_HIP, R_HIP = 23, 24


def _upright(n=1, lean=0.0):
    """`n` frames of a 0.4 m trunk leaning `lean` degrees in the x (lateral) direction."""
    rad = np.radians(lean)
    v = np.array([np.sin(rad), -np.cos(rad), 0.0]) * 0.4
    return body_world(np.tile(v, (n, 1)))


class ParseLabelTests(unittest.TestCase):
    def test_bare_exercise(self):
        p = parse_label("sit_hold")
        self.assertEqual(p.exercise, "sit_hold")
        self.assertEqual(p.params, {})
        self.assertEqual(p.raw, "sit_hold")

    def test_params(self):
        p = parse_label("sit_hold;arms=free;support=none")
        self.assertEqual(p.exercise, "sit_hold")
        self.assertEqual(p.params, {"arms": "free", "support": "none"})

    def test_whitespace_and_case_tolerated(self):
        # The annotator types this at a terminal prompt under time pressure.
        p = parse_label("  SIT_HOLD ; Arms = free ; SUPPORT=none ")
        self.assertEqual(p.exercise, "sit_hold")
        self.assertEqual(p.params, {"arms": "free", "support": "none"})

    def test_value_case_preserved(self):
        # `reason` / `gmfm` are free text where case can carry meaning.
        p = parse_label("exclude;reason=Dog Walked Through Shot")
        self.assertEqual(p.params["reason"], "Dog Walked Through Shot")

    def test_value_may_contain_equals(self):
        p = parse_label("exclude;reason=a=b")
        self.assertEqual(p.params["reason"], "a=b")

    def test_empty_segments_skipped(self):
        p = parse_label("crawl;;style=belly;")
        self.assertEqual(p.params, {"style": "belly"})

    def test_bare_token_dropped_not_raised(self):
        p = parse_label("sit_hold;free")
        self.assertEqual(p.params, {})
        self.assertEqual(p.exercise, "sit_hold")

    def test_duplicate_key_last_wins(self):
        p = parse_label("sit_hold;arms=free;arms=held")
        self.assertEqual(p.params["arms"], "held")

    # -- totality: parse_label must never raise and must never guess -------------- #
    def test_unknown_exercise_is_none(self):
        self.assertIsNone(parse_label("walking"))

    def test_legacy_free_text_is_none_not_raised(self):
        # Recordings predate this vocabulary; they must keep loading.
        for legacy in ("walking", "occluded", "Remy sitting nicely!", "a;b;c"):
            self.assertIsNone(parse_label(legacy))

    def test_empty_and_none_are_none(self):
        self.assertIsNone(parse_label(""))
        self.assertIsNone(parse_label(None))

    def test_dimension_mapping(self):
        self.assertEqual(parse_label("sit_hold").dimension, "B")
        self.assertEqual(parse_label("transition;from=sit;to=prone").dimension, "B")
        self.assertEqual(parse_label("crawl;style=belly").dimension, "C")
        self.assertEqual(parse_label("stand_hold;support=trunk").dimension, "D")

    def test_housekeeping_labels_have_no_dimension(self):
        self.assertIsNone(parse_label("calib;pose=upright").dimension)
        self.assertIsNone(parse_label("exclude;reason=x").dimension)

    def test_every_exercise_parses(self):
        for ex in EXERCISES:
            self.assertEqual(parse_label(ex).exercise, ex)

    def test_dimensions_only_reference_real_exercises(self):
        for ex in DIMENSIONS:
            self.assertIn(ex, EXERCISES)


class FormatLabelTests(unittest.TestCase):
    def test_round_trip_via_kwargs(self):
        s = format_label("sit_hold", arms="free", support="none")
        self.assertEqual(parse_label(s).params, {"arms": "free", "support": "none"})

    def test_round_trip_via_mapping(self):
        # `from` cannot be a Python keyword argument, which is why the mapping form exists.
        s = format_label("transition", {"from": "prone", "to": "sit"})
        self.assertEqual(s, "transition;from=prone;to=sit")
        self.assertEqual(parse_label(s).params, {"from": "prone", "to": "sit"})

    def test_mapping_and_kwargs_merge_kwargs_win(self):
        s = format_label("sit_hold", {"arms": "prop"}, arms="free")
        self.assertEqual(parse_label(s).params["arms"], "free")

    def test_key_order_is_declaration_order_then_extras_sorted(self):
        # Stable output matters: labels get compared by eye across months of sessions.
        s = format_label("sit_hold", {"gmfm": "23", "support": "none", "arms": "free"})
        self.assertEqual(s, "sit_hold;arms=free;support=none;gmfm=23")

    def test_extras_sorted_after_known_keys(self):
        s = format_label("sit_hold", {"zeta": "1", "alpha": "2", "arms": "free"})
        self.assertEqual(s, "sit_hold;arms=free;alpha=2;zeta=1")

    def test_bare_exercise(self):
        self.assertEqual(format_label("calib"), "calib")

    def test_unknown_exercise_raises(self):
        # format_label is a code-side constructor, unlike the annotator-facing parse.
        with self.assertRaises(ValueError):
            format_label("moonwalk", speed="fast")


class LabelWarningsTests(unittest.TestCase):
    def test_clean_label_is_silent(self):
        self.assertEqual(label_warnings("sit_hold;arms=free;support=none"), [])

    def test_free_value_params_never_warn(self):
        self.assertEqual(label_warnings("sit_hold;gmfm=whatever-the-sheet-says"), [])
        self.assertEqual(label_warnings("exclude;reason=dog in shot"), [])

    def test_typo_in_value_is_caught(self):
        # The bug this exists for: `arms=freee` would silently become its own
        # groupby bucket and split a baseline in half.
        warns = label_warnings("sit_hold;arms=freee")
        self.assertEqual(len(warns), 1)
        self.assertIn("freee", warns[0])

    def test_value_check_is_case_insensitive(self):
        self.assertEqual(label_warnings("sit_hold;arms=FREE"), [])

    def test_unknown_param_is_caught(self):
        warns = label_warnings("sit_hold;colour=blue")
        self.assertEqual(len(warns), 1)
        self.assertIn("colour", warns[0])

    def test_bare_token_is_caught(self):
        warns = label_warnings("sit_hold;free")
        self.assertEqual(len(warns), 1)
        self.assertIn("free", warns[0])

    def test_duplicate_key_is_caught(self):
        warns = label_warnings("sit_hold;arms=free;arms=held")
        self.assertTrue(any("duplicate" in w for w in warns))

    def test_unparseable_label_yields_one_warning(self):
        warns = label_warnings("walking")
        self.assertEqual(len(warns), 1)
        self.assertIn("walking", warns[0])

    def test_never_raises(self):
        for junk in ("", None, ";;;", "sit_hold;=x", "sit_hold;x="):
            label_warnings(junk)  # must not raise


class LandmarksOkTests(unittest.TestCase):
    def test_all_good_frames_pass(self):
        rec = fake_recording(n=5, visibility=1.0, presence=1.0)
        self.assertTrue(landmarks_ok(rec, TORSO).all())

    def test_low_visibility_frame_is_excluded(self):
        vis = np.ones((5, 33), dtype=np.float32)
        vis[2, TORSO[0]] = 0.1
        rec = fake_recording(n=5, visibility=vis)
        self.assertEqual(list(landmarks_ok(rec, TORSO)), [True, True, False, True, True])

    def test_low_presence_frame_is_excluded(self):
        pres = np.ones((5, 33), dtype=np.float32)
        pres[3, TORSO[2]] = 0.0
        rec = fake_recording(n=5, presence=pres)
        self.assertEqual(list(landmarks_ok(rec, TORSO)), [True, True, True, False, True])

    def test_gates_only_the_landmarks_asked_for(self):
        # The bug this prevents: requiring all 33 landmarks would discard a perfectly
        # good sitting trial because an ankle was out of frame.
        vis = np.ones((4, 33), dtype=np.float32)
        vis[:, 27] = 0.0  # LEFT_ANKLE — irrelevant to a torso metric
        rec = fake_recording(n=4, visibility=vis)
        self.assertTrue(landmarks_ok(rec, TORSO).all())

    def test_requires_every_requested_landmark(self):
        vis = np.ones((3, 33), dtype=np.float32)
        vis[1, TORSO[-1]] = 0.2  # one of four is enough to fail the frame
        rec = fake_recording(n=3, visibility=vis)
        self.assertEqual(list(landmarks_ok(rec, TORSO)), [True, False, True])

    def test_nan_rows_excluded_without_warning(self):
        # No-pose rows are all-NaN; NaN >= threshold is False, so they drop out.
        world = np.zeros((4, 33, 3), dtype=np.float32)
        world[1] = np.nan
        vis = np.ones((4, 33), dtype=np.float32)
        vis[1] = np.nan
        rec = fake_recording(world, visibility=vis, presence=vis)
        with np.errstate(all="raise"):  # must not emit a NaN-comparison warning
            mask = landmarks_ok(rec, TORSO)
        self.assertEqual(list(mask), [True, False, True, True])

    def test_threshold_is_inclusive(self):
        vis = np.full((2, 33), 0.5, dtype=np.float32)
        rec = fake_recording(n=2, visibility=vis)
        self.assertTrue(landmarks_ok(rec, TORSO, Gate(min_visibility=0.5)).all())

    def test_custom_gate_is_honoured(self):
        vis = np.full((2, 33), 0.6, dtype=np.float32)
        rec = fake_recording(n=2, visibility=vis)
        self.assertTrue(landmarks_ok(rec, TORSO, Gate(min_visibility=0.5)).all())
        self.assertFalse(landmarks_ok(rec, TORSO, Gate(min_visibility=0.9)).any())


class CoverageTests(unittest.TestCase):
    def test_fraction(self):
        mask = np.array([True, True, False, True, False])
        self.assertAlmostEqual(coverage(mask, 0, 5), 0.6)

    def test_respects_span(self):
        mask = np.array([False, True, True, True, False])
        self.assertAlmostEqual(coverage(mask, 1, 4), 1.0)

    def test_all_bad_is_zero(self):
        self.assertEqual(coverage(np.zeros(5, dtype=bool), 0, 5), 0.0)

    def test_empty_span_is_zero_not_nan(self):
        # NaN would compare False against every `coverage < threshold` check and so
        # let an empty trial pass the very guard it should trip.
        mask = np.ones(5, dtype=bool)
        self.assertEqual(coverage(mask, 2, 2), 0.0)
        self.assertEqual(coverage(mask, 4, 1), 0.0)


class LongestRunTests(unittest.TestCase):
    def test_picks_longest_of_several(self):
        mask = np.array([True, False, True, True, True, False, True])
        self.assertEqual(longest_run(mask, 0, 7), (2, 5))

    def test_returns_absolute_indices(self):
        mask = np.array([False, False, True, True, False])
        self.assertEqual(longest_run(mask, 1, 5), (2, 4))

    def test_run_touching_start_edge(self):
        mask = np.array([True, True, True, False])
        self.assertEqual(longest_run(mask, 0, 4), (0, 3))

    def test_run_touching_stop_edge(self):
        mask = np.array([False, True, True, True])
        self.assertEqual(longest_run(mask, 0, 4), (1, 4))

    def test_run_clipped_to_span(self):
        mask = np.ones(10, dtype=bool)
        self.assertEqual(longest_run(mask, 3, 7), (3, 7))

    def test_gap_is_not_stitched(self):
        # A trial that dropped tracking mid-way is not one long hold; bridging the gap
        # would invent the movement that happened inside it.
        mask = np.array([True] * 4 + [False] + [True] * 3)
        self.assertEqual(longest_run(mask, 0, 8), (0, 4))

    def test_no_good_frames_is_zero_length(self):
        self.assertEqual(longest_run(np.zeros(5, dtype=bool), 0, 5), (0, 0))

    def test_empty_span_is_zero_length_at_start(self):
        self.assertEqual(longest_run(np.ones(5, dtype=bool), 3, 3), (3, 3))

    def test_ties_are_stable(self):
        mask = np.array([True, True, False, True, True])
        self.assertEqual(longest_run(mask, 0, 5), (0, 2))


class TrunkVectorTests(unittest.TestCase):
    def test_recovers_the_constructed_trunk(self):
        world = body_world([[0.1, -0.4, 0.05]])
        np.testing.assert_allclose(trunk_vector(world)[0], [0.1, -0.4, 0.05], atol=1e-6)

    def test_invariant_to_whole_body_translation(self):
        # Pins the `- mid_hip` subtraction. Without it trunk_vector would be bare
        # mid-shoulder and would move when the child moves across the floor.
        v = [[0.1, -0.4, 0.05]]
        centered = trunk_vector(body_world(v))
        shifted = trunk_vector(body_world(v, hip_center=[2.0, -3.0, 1.5]))
        np.testing.assert_allclose(centered, shifted, atol=1e-6)

    def test_nan_frames_propagate(self):
        world = _upright(3)
        world[1] = np.nan
        self.assertTrue(np.isnan(trunk_vector(world)[1]).all())

    def test_world_mid_hip_com_proxy_is_identically_zero(self):
        # The correction this whole package is built around. MediaPipe defines the
        # world frame's ORIGIN as the mid-hip, so a "COM sway" metric taken there
        # measures nothing at all -- note the trunk below is swaying hard while the
        # mid-hip COM's sway is exactly zero. trunk_vector is the signal instead.
        n = 60
        sway = 0.05 * np.sin(np.linspace(0, 4 * np.pi, n))
        trunk = np.stack([sway, np.full(n, -0.4), np.zeros(n)], axis=1)
        world = body_world(trunk)

        com = mid(world, L_HIP, R_HIP)
        self.assertAlmostEqual(float(np.std(com)), 0.0, places=9)
        self.assertGreater(float(np.std(trunk_vector(world)[:, 0])), 0.03)


class TrunkFromVerticalTests(unittest.TestCase):
    def test_upright_is_zero(self):
        self.assertAlmostEqual(trunk_from_vertical(_upright())[0], 0.0, places=5)

    def test_known_lean_angles(self):
        for lean in (0.0, 10.0, 30.0, 45.0, 90.0, 120.0):
            got = trunk_from_vertical(_upright(lean=lean))[0]
            self.assertAlmostEqual(got, lean, places=4, msg=f"lean={lean}")

    def test_is_unsigned_cannot_separate_lean_direction(self):
        # Documents the inherited limitation: +30 and -30 are indistinguishable here.
        # project_horizontal is what carries direction.
        self.assertAlmostEqual(
            trunk_from_vertical(_upright(lean=30.0))[0],
            trunk_from_vertical(_upright(lean=-30.0))[0],
            places=5,
        )

    def test_lying_flat_is_ninety(self):
        world = body_world([[0.0, 0.0, 0.4]])  # trunk points at the camera, no vertical part
        self.assertAlmostEqual(trunk_from_vertical(world)[0], 90.0, places=4)

    def test_custom_up_shifts_the_reference(self):
        # A camera pitched by 30 deg: an upright child reads 30 against WORLD_UP,
        # and 0 against the corrected vertical.
        tilted = np.array([np.sin(np.radians(30.0)), -np.cos(np.radians(30.0)), 0.0])
        world = _upright(lean=30.0)
        self.assertAlmostEqual(trunk_from_vertical(world, up=tilted)[0], 0.0, places=4)

    def test_scale_invariant(self):
        short = body_world([[0.1, -0.4, 0.0]])
        tall = body_world([[0.25, -1.0, 0.0]])
        self.assertAlmostEqual(
            trunk_from_vertical(short)[0], trunk_from_vertical(tall)[0], places=4
        )

    def test_nan_frames_are_nan_not_raised(self):
        world = _upright(3)
        world[1] = np.nan
        got = trunk_from_vertical(world)
        self.assertTrue(np.isnan(got[1]))
        self.assertFalse(np.isnan(got[0]))

    def test_returns_one_value_per_frame(self):
        self.assertEqual(trunk_from_vertical(_upright(7)).shape, (7,))


class ProjectHorizontalTests(unittest.TestCase):
    def test_level_camera_gives_ml_x_and_ap_z(self):
        pts = np.array([[1.0, 5.0, 2.0]])  # y (vertical) must drop out entirely
        np.testing.assert_allclose(project_horizontal(pts)[0], [1.0, 2.0], atol=1e-9)

    def test_vertical_component_is_discarded(self):
        a = project_horizontal(np.array([[0.3, 0.0, -0.2]]))
        b = project_horizontal(np.array([[0.3, 99.0, -0.2]]))
        np.testing.assert_allclose(a, b, atol=1e-9)

    def test_basis_is_orthonormal_for_a_tilted_up(self):
        up = np.array([0.2, -1.0, 0.1])
        # Round-tripping the basis vectors recovers the identity iff they are orthonormal.
        got = project_horizontal(np.eye(3), up=up)
        self.assertEqual(got.shape, (3, 2))
        basis = np.array([project_horizontal(np.eye(3), up=up)[:, i] for i in range(2)])
        np.testing.assert_allclose(basis @ basis.T, np.eye(2), atol=1e-9)

    def test_tilted_up_is_perpendicular_to_both_axes(self):
        up = np.array([0.3, -1.0, -0.15])
        np.testing.assert_allclose(project_horizontal(up[None, :], up=up)[0], [0, 0], atol=1e-9)

    def test_rolled_camera_falls_back_without_blowing_up(self):
        # `up` parallel to world-x degenerates the ML axis; the basis must stay defined.
        got = project_horizontal(np.array([[0.0, 1.0, 2.0]]), up=np.array([1.0, 0.0, 0.0]))
        self.assertTrue(np.isfinite(got).all())

    def test_shape(self):
        self.assertEqual(project_horizontal(np.zeros((5, 3))).shape, (5, 2))


class ComNormTests(unittest.TestCase):
    def test_midpoint_of_hips_in_image_fractions(self):
        norm = np.zeros((1, 33, 3))
        norm[0, L_HIP] = [0.6, 0.8, 0.1]
        norm[0, R_HIP] = [0.4, 0.6, -0.1]
        np.testing.assert_allclose(com_norm(norm)[0], [0.5, 0.7, 0.0], atol=1e-9)

    def test_tracks_translation_that_the_world_frame_cannot_see(self):
        # The reason com_norm exists: the pelvis is pinned at the origin in world
        # coords, so image space is the only place its travel is visible at all.
        norm = np.zeros((2, 33, 3))
        norm[0, L_HIP] = norm[0, R_HIP] = [0.2, 0.5, 0.0]
        norm[1, L_HIP] = norm[1, R_HIP] = [0.7, 0.5, 0.0]
        self.assertAlmostEqual(float(np.diff(com_norm(norm)[:, 0])[0]), 0.5, places=6)


class EstimateUpTests(unittest.TestCase):
    def test_recovers_the_median_trunk_direction(self):
        rec = fake_recording(_upright(10, lean=20.0))
        seg = SimpleNamespace(start=0, stop=10)
        got = estimate_up(rec, seg)
        expected = np.array([np.sin(np.radians(20.0)), -np.cos(np.radians(20.0)), 0.0])
        np.testing.assert_allclose(got, expected, atol=1e-5)

    def test_result_is_unit_length(self):
        rec = fake_recording(_upright(5, lean=15.0))
        got = estimate_up(rec, SimpleNamespace(start=0, stop=5))
        self.assertAlmostEqual(float(np.linalg.norm(got)), 1.0, places=6)

    def test_calibrating_against_itself_reads_upright(self):
        rec = fake_recording(_upright(10, lean=25.0))
        seg = SimpleNamespace(start=0, stop=10)
        up = estimate_up(rec, seg)
        self.assertAlmostEqual(trunk_from_vertical(rec.landmarks_world, up=up)[0], 0.0, places=4)

    def test_ignores_no_pose_frames(self):
        world = _upright(6, lean=20.0)
        world[2] = np.nan
        rec = fake_recording(world)
        got = estimate_up(rec, SimpleNamespace(start=0, stop=6))
        self.assertTrue(np.isfinite(got).all())

    def test_segment_with_no_pose_raises(self):
        world = _upright(4)
        world[:] = np.nan
        rec = fake_recording(world)
        with self.assertRaises(ValueError):
            estimate_up(rec, SimpleNamespace(start=0, stop=4))

    def test_respects_segment_bounds(self):
        # Only the second half is the calibration pose; the first half must not leak in.
        world = np.concatenate([_upright(5, lean=0.0), _upright(5, lean=40.0)])
        rec = fake_recording(world)
        got = estimate_up(rec, SimpleNamespace(start=5, stop=10))
        expected = np.array([np.sin(np.radians(40.0)), -np.cos(np.radians(40.0)), 0.0])
        np.testing.assert_allclose(got, expected, atol=1e-5)
        self.assertGreater(float(np.linalg.norm(got - WORLD_UP)), 0.3)


class WindowLengthTests(unittest.TestCase):
    def test_is_odd(self):
        for fs in (10.0, 25.0, 30.0, 60.0, 120.0):
            self.assertEqual(window_length(fs) % 2, 1, msg=f"fs={fs}")

    def test_exceeds_poly_order(self):
        # savgol_filter requires window > polyorder; a tiny fs must not violate it.
        for fs in (1.0, 4.0, 8.0):
            self.assertGreater(window_length(fs, WINDOW_S, POLY), POLY)

    def test_tracks_the_requested_duration(self):
        self.assertEqual(window_length(30.0, 0.25, 2), 7)
        self.assertEqual(window_length(100.0, 0.25, 2), 25)


class ResampleUniformTests(unittest.TestCase):
    def test_uniform_input_is_preserved(self):
        t = np.arange(10) * (1000.0 / FS)
        x = np.arange(10, dtype=float)
        t_s, got = resample_uniform(t, x)
        np.testing.assert_allclose(got, x, atol=1e-6)
        np.testing.assert_allclose(t_s, np.arange(10) / FS, atol=1e-9)

    def test_jittery_timestamps_land_on_an_even_grid(self):
        # The reason this exists: real frames are ~30 Hz but never exactly 30 Hz, and
        # savgol's single `delta` silently assumes they are.
        t = np.array([0, 30, 71, 99, 135, 165, 201, 232])
        x = t / 1000.0  # a ramp of 1.0 units/s, so interpolation is exact
        t_s, got = resample_uniform(t, x)
        np.testing.assert_allclose(got, t_s, atol=1e-6)
        np.testing.assert_allclose(np.diff(t_s), 1.0 / FS, atol=1e-9)

    def test_multichannel(self):
        t = np.arange(10) * (1000.0 / FS)
        x = np.stack([np.arange(10), np.arange(10) * 2.0], axis=1)
        _, got = resample_uniform(t, x)
        self.assertEqual(got.shape, (10, 2))
        np.testing.assert_allclose(got[:, 1], got[:, 0] * 2, atol=1e-6)

    def test_time_starts_at_zero_regardless_of_offset(self):
        t = np.arange(10) * (1000.0 / FS) + 987654.0
        t_s, _ = resample_uniform(t, np.arange(10, dtype=float))
        self.assertAlmostEqual(float(t_s[0]), 0.0)

    def test_nan_is_not_bridged(self):
        # Interpolating across a dropout would fabricate the movement the tracker missed.
        t = np.arange(10) * (1000.0 / FS)
        x = np.arange(10, dtype=float)
        x[5] = np.nan
        _, got = resample_uniform(t, x)
        self.assertTrue(np.isnan(got).any())
        self.assertFalse(np.isnan(got[0]))

    def test_single_sample_is_empty_not_raised(self):
        t_s, got = resample_uniform(np.array([5]), np.array([1.0]))
        self.assertEqual(t_s.size, 0)
        self.assertEqual(got.size, 0)

    def test_empty_input_is_empty(self):
        t_s, got = resample_uniform(np.array([]), np.array([]))
        self.assertEqual(t_s.size, 0)
        self.assertEqual(got.size, 0)

    def test_zero_duration_span_is_empty(self):
        t_s, got = resample_uniform(np.array([7, 7, 7]), np.array([1.0, 2.0, 3.0]))
        self.assertEqual(t_s.size, 0)
        self.assertEqual(got.size, 0)


class SmoothTests(unittest.TestCase):
    def test_polynomial_passes_through_unchanged(self):
        # savgol with poly=2 reproduces a quadratic exactly, noise aside.
        t = np.arange(60) / FS
        x = 3.0 + 2.0 * t + 0.5 * t**2
        np.testing.assert_allclose(smooth(x), x, atol=1e-6)

    def test_reduces_noise(self):
        rng = np.random.default_rng(0)
        t = np.arange(300) / FS
        clean = np.sin(2 * np.pi * 0.5 * t)
        noisy = clean + rng.normal(0, 0.1, clean.shape)
        self.assertLess(
            float(np.std(smooth(noisy) - clean)), float(np.std(noisy - clean))
        )

    def test_first_derivative_of_a_ramp_is_the_slope(self):
        t = np.arange(60) / FS
        x = 2.5 * t
        np.testing.assert_allclose(smooth(x, deriv=1), np.full(60, 2.5), atol=1e-6)

    def test_derivative_is_per_second_not_per_frame(self):
        # Pins `delta=1/fs`: without it the derivative is off by a factor of FS.
        t = np.arange(60) / FS
        np.testing.assert_allclose(smooth(1.0 * t, deriv=1), np.ones(60), atol=1e-6)

    def test_derivative_of_a_sway_rate_sine(self):
        # 0.5 Hz is the band postural sway actually lives in; there the chain is ~99%
        # accurate. Interior only: savgol's polynomial edge fit is weakest at the ends.
        f = 0.5
        t = np.arange(300) / FS
        got = smooth(np.sin(2 * np.pi * f * t), deriv=1)
        expected = 2 * np.pi * f * np.cos(2 * np.pi * f * t)
        np.testing.assert_allclose(got[20:-20], expected[20:-20], atol=0.05)

    def test_derivative_gain_rolls_off_with_frequency(self):
        # Pins the passband documented in the module docstring. This attenuation is a
        # real property of the pinned chain, not a defect: it is the same for every
        # trial, so it cancels within-child. It must not drift silently, because a
        # changed window would move every historical number without touching the data.
        expected = {0.25: 0.997, 0.5: 0.987, 1.0: 0.950, 2.0: 0.809, 3.0: 0.607}
        for f, gain in expected.items():
            t = np.arange(600) / FS
            got = smooth(np.sin(2 * np.pi * f * t), deriv=1)[50:-50]
            true = (2 * np.pi * f * np.cos(2 * np.pi * f * t))[50:-50]
            measured = float(np.max(np.abs(got)) / np.max(np.abs(true)))
            self.assertAlmostEqual(measured, gain, places=2, msg=f"{f} Hz")

    def test_multichannel_smooths_along_axis_zero(self):
        t = np.arange(60) / FS
        x = np.stack([2.0 * t, -3.0 * t], axis=1)
        np.testing.assert_allclose(
            smooth(x, deriv=1), np.tile([2.0, -3.0], (60, 1)), atol=1e-6
        )

    def test_short_segment_is_nan_not_raised(self):
        # A mis-marked 2-frame annotation must not take down a 40-row report table.
        short = np.arange(3, dtype=float)
        got = smooth(short)
        self.assertEqual(got.shape, short.shape)
        self.assertTrue(np.isnan(got).all())

    def test_segment_exactly_at_the_window_length_works(self):
        w = window_length()
        got = smooth(np.zeros(w))
        self.assertFalse(np.isnan(got).any())

    def test_short_multichannel_keeps_shape(self):
        got = smooth(np.zeros((3, 2)))
        self.assertEqual(got.shape, (3, 2))
        self.assertTrue(np.isnan(got).all())


class VelocityTests(unittest.TestCase):
    def test_constant_velocity_path(self):
        t = np.arange(60) * (1000.0 / FS)
        p = np.stack([np.arange(60) / FS * 0.3, np.zeros(60)], axis=1)
        _, v, speed = velocity(t, p)
        np.testing.assert_allclose(v[:, 0], 0.3, atol=1e-6)
        np.testing.assert_allclose(speed, 0.3, atol=1e-6)

    def test_speed_is_the_norm_of_the_components(self):
        t = np.arange(60) * (1000.0 / FS)
        s = np.arange(60) / FS
        p = np.stack([3.0 * s, 4.0 * s], axis=1)  # 3-4-5 triangle
        _, _, speed = velocity(t, p)
        np.testing.assert_allclose(speed, 5.0, atol=1e-6)

    def test_stationary_path_has_zero_speed(self):
        t = np.arange(60) * (1000.0 / FS)
        _, _, speed = velocity(t, np.zeros((60, 3)))
        np.testing.assert_allclose(speed, 0.0, atol=1e-9)

    def test_returns_uniform_time_grid(self):
        t = np.array([0, 31, 68, 99, 134, 170, 199, 232, 265, 301])
        t_s, v, speed = velocity(t, np.zeros((10, 2)))
        np.testing.assert_allclose(np.diff(t_s), 1.0 / FS, atol=1e-9)
        self.assertEqual(v.shape[0], t_s.size)
        self.assertEqual(speed.shape[0], t_s.size)

    def test_short_segment_is_nan_not_raised(self):
        t = np.arange(4) * (1000.0 / FS)
        _, _, speed = velocity(t, np.zeros((4, 2)))
        self.assertTrue(np.isnan(speed).all())

    def test_single_sample_is_empty_not_raised(self):
        t_s, v, speed = velocity(np.array([0]), np.zeros((1, 3)))
        self.assertEqual(t_s.size, 0)
        self.assertEqual(speed.size, 0)
        self.assertEqual(v.shape, (0, 3))


def _annotated(labels, *, n=10, fps=30.0):
    """A fake recording carrying `labels` as (label, start_ms, end_ms) annotations."""
    return fake_recording(
        n=n,
        fps=fps,
        annotations=[
            Annotation(i, label, start, end)
            for i, (label, start, end) in enumerate(labels)
        ],
    )


class FrameSpanTests(unittest.TestCase):
    def setUp(self):
        self.ts = np.arange(10) * 100  # 0, 100, ... 900 ms

    def test_covers_the_interval(self):
        self.assertEqual(frame_span(self.ts, 200, 500), (2, 6))

    def test_interval_is_closed_at_the_end(self):
        # The annotator marks the out-point ON a frame they are looking at, so that
        # frame belongs to the trial.
        start, stop = frame_span(self.ts, 0, 300)
        self.assertEqual((start, stop), (0, 4))

    def test_boundaries_between_frames_round_outward(self):
        self.assertEqual(frame_span(self.ts, 150, 450), (2, 5))

    def test_whole_recording(self):
        self.assertEqual(frame_span(self.ts, 0, 900), (0, 10))

    def test_interval_past_the_end_is_empty(self):
        start, stop = frame_span(self.ts, 5000, 6000)
        self.assertEqual(start, stop)

    def test_interval_before_the_start_is_empty(self):
        start, stop = frame_span(self.ts, -500, -100)
        self.assertEqual(start, stop)

    def test_zero_width_interval_between_frames_is_empty(self):
        start, stop = frame_span(self.ts, 150, 150)
        self.assertEqual(start, stop)

    def test_stop_never_precedes_start(self):
        start, stop = frame_span(self.ts, 500, 200)  # inverted, should not go negative
        self.assertGreaterEqual(stop, start)


class SegmentsTests(unittest.TestCase):
    def test_parses_and_spans(self):
        rec = _annotated([("sit_hold;arms=free", 0, 200)])
        segs = segments(rec)
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0].exercise, "sit_hold")
        self.assertEqual(segs[0].parsed.params, {"arms": "free"})
        self.assertEqual((segs[0].start, segs[0].stop), (0, 7))
        self.assertEqual(segs[0].n_frames, 7)

    def test_filters_by_exercise(self):
        rec = _annotated(
            [
                ("sit_hold;arms=free", 0, 100),
                ("crawl;style=belly", 100, 200),
                ("sit_hold;arms=prop", 200, 300),
            ]
        )
        self.assertEqual(len(segments(rec, "sit_hold")), 2)
        self.assertEqual(len(segments(rec, "crawl")), 1)
        self.assertEqual(len(segments(rec)), 3)

    def test_unparseable_labels_are_dropped(self):
        # Recordings predate this vocabulary; free text is not a trial.
        rec = _annotated([("walking", 0, 100), ("sit_hold", 100, 200)])
        segs = segments(rec)
        self.assertEqual([s.exercise for s in segs], ["sit_hold"])

    def test_calib_is_returned(self):
        # estimate_up needs it, so it is a segment even though it is not a trial.
        rec = _annotated([("calib;pose=upright", 0, 100)])
        self.assertEqual([s.exercise for s in segments(rec)], ["calib"])

    def test_exclude_is_not_returned_as_a_trial(self):
        rec = _annotated([("exclude;reason=dog", 0, 100)])
        self.assertEqual(segments(rec), [])

    def test_overlapping_exclude_drops_the_whole_trial(self):
        # A hold whose middle is untrustworthy is not a shorter valid hold.
        rec = _annotated(
            [("sit_hold", 0, 300), ("exclude;reason=assist", 100, 150)]
        )
        self.assertEqual(segments(rec), [])

    def test_exclude_touching_the_edge_still_drops(self):
        rec = _annotated([("sit_hold", 0, 100), ("exclude;reason=x", 100, 200)])
        self.assertEqual(segments(rec), [])

    def test_non_overlapping_exclude_leaves_the_trial_alone(self):
        rec = _annotated(
            [("sit_hold", 0, 100), ("exclude;reason=x", 200, 300)]
        )
        self.assertEqual([s.exercise for s in segments(rec)], ["sit_hold"])

    def test_split_trials_around_an_exclude_both_survive(self):
        # The documented way to keep the usable parts of an interrupted trial.
        rec = _annotated(
            [
                ("sit_hold", 0, 90),
                ("exclude;reason=assist", 100, 150),
                ("sit_hold", 160, 300),
            ]
        )
        self.assertEqual(len(segments(rec, "sit_hold")), 2)

    def test_exclude_applies_across_exercises(self):
        rec = _annotated(
            [("crawl", 0, 300), ("calib", 0, 50), ("exclude;reason=x", 10, 20)]
        )
        self.assertEqual(segments(rec), [])

    def test_empty_span_segment_is_kept_not_hidden(self):
        # A mis-marked annotation covering no frames should surface as zero coverage,
        # not vanish and leave the annotator wondering where their trial went.
        rec = _annotated([("sit_hold", 99000, 99500)])
        segs = segments(rec)
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0].n_frames, 0)

    def test_no_annotations_is_empty(self):
        self.assertEqual(segments(fake_recording(n=5)), [])

    def test_segment_carries_the_original_annotation(self):
        rec = _annotated([("sit_hold;gmfm=23", 0, 200)])
        seg = segments(rec)[0]
        self.assertIsInstance(seg, Segment)
        self.assertEqual(seg.ann.label, "sit_hold;gmfm=23")
        self.assertEqual(seg.ann.start_ms, 0)
        self.assertEqual(seg.parsed.params["gmfm"], "23")


class SegmentsIntegrationTests(unittest.TestCase):
    """Against a real HDF5 file written by the real recorder + annotation store."""

    def _write(self, path, n=30, step_ms=33):
        with HDF5Recorder(path) as rec:
            for i in range(n):
                result = pose_result(
                    poses_norm=[make_landmarks()], poses_world=[make_landmarks()]
                )
                rec.append(solid_frame(), i * step_ms, result)

    def test_end_to_end_ms_to_frame_mapping(self):
        with TemporaryDirectory() as d:
            path = Path(d) / "rec.h5"
            self._write(path, n=30, step_ms=33)
            # AnnotationStore ("r+") must be opened before Recording ("r") -- h5py
            # raises on the reverse order (see test_annotations.test_rw_then_ro...).
            store = AnnotationStore(path)
            store.add("sit_hold;arms=free;support=none", 99, 330)
            store.add("exclude;reason=dog in shot", 500, 560)
            store.add("crawl;style=belly;dir=away", 495, 600)  # overlaps the exclude
            store.add("walking", 700, 800)  # legacy free text
            rec = Recording(path)
            try:
                segs = segments(rec)
                self.assertEqual([s.exercise for s in segs], ["sit_hold"])
                seg = segs[0]
                self.assertEqual((seg.start, seg.stop), (3, 11))
                # The span really does bracket the requested milliseconds.
                self.assertGreaterEqual(int(rec.timestamps_ms[seg.start]), 99)
                self.assertLessEqual(int(rec.timestamps_ms[seg.stop - 1]), 330)
            finally:
                rec.close()
                store.close()

    def test_recording_without_annotations_yields_no_segments(self):
        with TemporaryDirectory() as d:
            path = Path(d) / "rec.h5"
            self._write(path, n=5)
            with Recording(path) as rec:
                self.assertEqual(segments(rec), [])


class PathLengthTests(unittest.TestCase):
    def test_closed_square_is_its_perimeter(self):
        square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
        self.assertAlmostEqual(path_length(square), 4.0, places=9)

    def test_straight_line(self):
        self.assertAlmostEqual(path_length(np.array([[0.0, 0.0], [3.0, 4.0]])), 5.0)

    def test_stationary_path_is_zero(self):
        self.assertAlmostEqual(path_length(np.zeros((10, 2))), 0.0)

    def test_grows_with_more_laps(self):
        # The confound the metric carries: a longer trial travels further, full stop.
        square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
        two_laps = np.concatenate([square, square[1:]])
        self.assertAlmostEqual(path_length(two_laps), 8.0, places=9)

    def test_single_point_is_nan(self):
        self.assertTrue(np.isnan(path_length(np.zeros((1, 2)))))

    def test_nan_input_is_nan(self):
        self.assertTrue(np.isnan(path_length(np.array([[0.0, 0.0], [np.nan, 1.0]]))))


class SwayEllipseAreaTests(unittest.TestCase):
    def test_isotropic_gaussian_matches_the_closed_form(self):
        rng = np.random.default_rng(1)
        sigma = 0.02
        cloud = rng.normal(0, sigma, (20000, 2))
        expected = 5.991 * np.pi * sigma**2
        self.assertAlmostEqual(sway_ellipse_area(cloud) / expected, 1.0, places=1)

    def test_anisotropic_gaussian_uses_both_axes(self):
        rng = np.random.default_rng(2)
        cloud = np.stack([rng.normal(0, 0.04, 20000), rng.normal(0, 0.01, 20000)], axis=1)
        expected = 5.991 * np.pi * 0.04 * 0.01
        self.assertAlmostEqual(sway_ellipse_area(cloud) / expected, 1.0, places=1)

    def test_collinear_cloud_encloses_nothing(self):
        t = np.linspace(0, 1, 100)
        self.assertAlmostEqual(sway_ellipse_area(np.stack([t, 2 * t], axis=1)), 0.0, places=9)

    def test_invariant_to_translation(self):
        rng = np.random.default_rng(3)
        cloud = rng.normal(0, 0.02, (500, 2))
        self.assertAlmostEqual(
            sway_ellipse_area(cloud), sway_ellipse_area(cloud + [5.0, -3.0]), places=12
        )

    def test_invariant_to_rotation(self):
        rng = np.random.default_rng(4)
        cloud = np.stack([rng.normal(0, 0.04, 2000), rng.normal(0, 0.01, 2000)], axis=1)
        angle = np.radians(37.0)
        rot = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        self.assertAlmostEqual(
            sway_ellipse_area(cloud), sway_ellipse_area(cloud @ rot.T), places=9
        )

    def test_scales_with_variance(self):
        rng = np.random.default_rng(5)
        cloud = rng.normal(0, 0.02, (5000, 2))
        self.assertAlmostEqual(
            sway_ellipse_area(2 * cloud) / sway_ellipse_area(cloud), 4.0, places=6
        )

    def test_too_few_points_is_nan(self):
        self.assertTrue(np.isnan(sway_ellipse_area(np.zeros((2, 2)))))

    def test_nan_input_is_nan(self):
        cloud = np.zeros((10, 2))
        cloud[3, 0] = np.nan
        self.assertTrue(np.isnan(sway_ellipse_area(cloud)))


def _hold_rec(trunk, *, fps=30.0, visibility=1.0, wrist_y=None):
    """A fake recording of a body whose pelvis->shoulder vector follows `trunk`."""
    n = len(trunk)
    wrists = None
    if wrist_y is not None:
        wrists = np.stack(
            [np.full(n, 0.2), np.full(n, wrist_y), np.zeros(n)], axis=1
        )
    world = body_world(trunk, left_wrist=wrists, right_wrist=wrists)
    return fake_recording(world, visibility=visibility, fps=fps)


def _seg(start, stop, label="sit_hold;arms=free"):
    return Segment(
        ann=Annotation(0, label, 0, 0), parsed=parse_label(label), start=start, stop=stop
    )


def _swaying(n=300, amp=0.02, freq=0.5, fps=30.0, axis=0):
    """Trunk vector swaying sinusoidally about upright, `amp` meters at `freq` Hz.

    One axis only, so the sway cloud is collinear -- fine for RMS and path length, but
    it encloses no area, so use `_swaying_ellipse` for anything about the sway ellipse.
    """
    t = np.arange(n) / fps
    trunk = np.zeros((n, 3))
    trunk[:, 1] = -0.4
    trunk[:, axis] = amp * np.sin(2 * np.pi * freq * t)
    return trunk


def _swaying_ellipse(n=600, ml=0.03, ap=0.01, freq=0.5, fps=30.0):
    """Trunk tracing an ML-by-AP ellipse: ML on a sine, AP on a cosine (quarter-phase).

    Over whole cycles the covariance is diag(ml^2/2, ap^2/2), so the 95% ellipse area
    has the closed form ``5.991 * pi * ml * ap / 2``.
    """
    t = np.arange(n) / fps
    trunk = np.zeros((n, 3))
    trunk[:, 1] = -0.4
    trunk[:, 0] = ml * np.sin(2 * np.pi * freq * t)  # ML is world x
    trunk[:, 2] = ap * np.cos(2 * np.pi * freq * t)  # AP is world z
    return trunk


class HoldMetricsTests(unittest.TestCase):
    def test_perfectly_still_hold_has_no_sway(self):
        rec = _hold_rec(np.tile([0.0, -0.4, 0.0], (90, 1)))
        m = hold_metrics(rec, _seg(0, 90))
        self.assertAlmostEqual(m.path_length_m, 0.0, places=6)
        self.assertAlmostEqual(m.rms_m, 0.0, places=6)
        self.assertAlmostEqual(m.ellipse_area_m2, 0.0, places=9)
        self.assertAlmostEqual(m.trunk_angle_mean_deg, 0.0, places=4)

    def test_duration_is_the_annotated_span(self):
        # The hold judgment is the annotator's; the code measures inside their marks.
        rec = _hold_rec(_swaying(n=90))
        m = hold_metrics(rec, _seg(0, 90))
        span = float(rec.timestamps_ms[89] - rec.timestamps_ms[0]) / 1000.0
        self.assertAlmostEqual(m.duration_s, span, places=6)
        self.assertAlmostEqual(m.duration_s, 3.0, places=1)
        self.assertEqual(m.n_frames, 90)

    def test_lateral_sway_lands_on_the_ml_axis_not_ap(self):
        # The split exists because ML is in the image plane and AP is inferred depth.
        rec = _hold_rec(_swaying(axis=0))
        m = hold_metrics(rec, _seg(0, 300))
        self.assertGreater(m.sway_ml_rms_m, 0.01)
        self.assertAlmostEqual(m.sway_ap_rms_m, 0.0, places=6)

    def test_depth_sway_lands_on_the_ap_axis(self):
        rec = _hold_rec(_swaying(axis=2))
        m = hold_metrics(rec, _seg(0, 300))
        self.assertGreater(m.sway_ap_rms_m, 0.01)
        self.assertAlmostEqual(m.sway_ml_rms_m, 0.0, places=6)

    def test_sway_rms_matches_the_sine_closed_form(self):
        # RMS of an amplitude-A sine is A/sqrt(2); the 0.5 Hz chain gain is ~0.99.
        rec = _hold_rec(_swaying(n=600, amp=0.03, freq=0.5))
        m = hold_metrics(rec, _seg(0, 600))
        self.assertAlmostEqual(m.sway_ml_rms_m, 0.03 / np.sqrt(2), places=3)

    def test_more_sway_reads_as_more_sway(self):
        small = hold_metrics(_hold_rec(_swaying(amp=0.01)), _seg(0, 300))
        large = hold_metrics(_hold_rec(_swaying(amp=0.04)), _seg(0, 300))
        self.assertGreater(large.rms_m, small.rms_m)
        self.assertGreater(large.path_length_m, small.path_length_m)

    def test_ellipse_area_of_a_traced_ellipse_matches_the_closed_form(self):
        rec = _hold_rec(_swaying_ellipse(n=600, ml=0.03, ap=0.01))
        m = hold_metrics(rec, _seg(0, 600))
        expected = 5.991 * np.pi * 0.03 * 0.01 / 2
        self.assertAlmostEqual(m.ellipse_area_m2 / expected, 1.0, places=2)

    def test_ellipse_area_grows_with_sway_in_both_axes(self):
        small = hold_metrics(_hold_rec(_swaying_ellipse(ml=0.02, ap=0.01)), _seg(0, 600))
        large = hold_metrics(_hold_rec(_swaying_ellipse(ml=0.04, ap=0.02)), _seg(0, 600))
        self.assertAlmostEqual(large.ellipse_area_m2 / small.ellipse_area_m2, 4.0, places=1)

    def test_one_dimensional_sway_encloses_no_area(self):
        # Not a defect: a trunk rocking on a single axis really does enclose nothing.
        # It is why ellipse area must be read next to the ML/AP RMS split, never alone.
        m = hold_metrics(_hold_rec(_swaying(n=300, amp=0.04)), _seg(0, 300))
        self.assertAlmostEqual(m.ellipse_area_m2, 0.0, places=9)
        self.assertGreater(m.sway_ml_rms_m, 0.01)

    def test_mean_velocity_is_path_length_over_tracked_time(self):
        rec = _hold_rec(_swaying(n=300))
        m = hold_metrics(rec, _seg(0, 300))
        self.assertAlmostEqual(m.mean_velocity_mps, m.path_length_m / m.tracked_s, places=6)

    def test_path_length_is_duration_confounded_but_velocity_is_not(self):
        # The trap: the same sway measured for longer travels further. A comparison on
        # path_length alone would rank a longer, steadier hold as worse.
        short = hold_metrics(_hold_rec(_swaying(n=150)), _seg(0, 150))
        long = hold_metrics(_hold_rec(_swaying(n=600)), _seg(0, 600))
        self.assertGreater(long.path_length_m, short.path_length_m * 3)
        self.assertAlmostEqual(long.mean_velocity_mps, short.mean_velocity_mps, places=2)

    def test_window_s_makes_unequal_trials_comparable(self):
        short = hold_metrics(_hold_rec(_swaying(n=180)), _seg(0, 180), window_s=5.0)
        long = hold_metrics(_hold_rec(_swaying(n=600)), _seg(0, 600), window_s=5.0)
        self.assertAlmostEqual(short.tracked_s, 5.0, places=1)
        self.assertAlmostEqual(long.tracked_s, 5.0, places=1)
        self.assertAlmostEqual(long.path_length_m, short.path_length_m, places=2)

    def test_window_s_longer_than_the_trial_keeps_the_whole_trial(self):
        rec = _hold_rec(_swaying(n=90))
        m = hold_metrics(rec, _seg(0, 90), window_s=60.0)
        span = float(rec.timestamps_ms[89] - rec.timestamps_ms[0]) / 1000.0
        self.assertAlmostEqual(m.tracked_s, span, places=6)

    def test_trunk_angle_stats(self):
        rad = np.radians(20.0)
        trunk = np.tile([np.sin(rad) * 0.4, -np.cos(rad) * 0.4, 0.0], (90, 1))
        m = hold_metrics(_hold_rec(trunk), _seg(0, 90))
        self.assertAlmostEqual(m.trunk_angle_mean_deg, 20.0, places=3)
        self.assertAlmostEqual(m.trunk_angle_sd_deg, 0.0, places=4)
        self.assertAlmostEqual(m.trunk_angle_range_deg, 0.0, places=4)

    def test_trunk_angle_range_tracks_lean_excursion(self):
        m = hold_metrics(_hold_rec(_swaying(n=300, amp=0.05)), _seg(0, 300))
        self.assertGreater(m.trunk_angle_range_deg, 5.0)
        self.assertGreater(m.trunk_angle_sd_deg, 0.0)

    def test_up_source_records_which_vertical_was_used(self):
        rec = _hold_rec(_swaying(n=90))
        self.assertEqual(hold_metrics(rec, _seg(0, 90)).up_source, "world_y")
        tilted = np.array([0.2, -1.0, 0.0])
        self.assertEqual(hold_metrics(rec, _seg(0, 90), up=tilted).up_source, "custom")

    def test_custom_up_changes_the_measured_lean(self):
        rec = _hold_rec(np.tile([0.0, -0.4, 0.0], (90, 1)))
        tilted = np.array([np.sin(np.radians(15.0)), -np.cos(np.radians(15.0)), 0.0])
        m = hold_metrics(rec, _seg(0, 90), up=tilted)
        self.assertAlmostEqual(m.trunk_angle_mean_deg, 15.0, places=3)

    # -- gating and coverage ------------------------------------------------------ #
    def test_coverage_reports_untrusted_frames(self):
        vis = np.ones((100, 33), dtype=np.float32)
        vis[40:60, TORSO[0]] = 0.1
        rec = _hold_rec(_swaying(n=100), visibility=vis)
        m = hold_metrics(rec, _seg(0, 100))
        self.assertAlmostEqual(m.coverage, 0.8, places=6)

    def test_sway_uses_only_the_longest_tracked_run(self):
        # A trial that dropped tracking is not one long hold; bridging the gap would
        # invent the movement inside it.
        vis = np.ones((300, 33), dtype=np.float32)
        vis[100:110, TORSO[0]] = 0.0
        rec = _hold_rec(_swaying(n=300), visibility=vis)
        m = hold_metrics(rec, _seg(0, 300))
        self.assertAlmostEqual(m.duration_s, 299 / 30.0, places=2)  # annotated span
        self.assertAlmostEqual(m.tracked_s, 189 / 30.0, places=2)  # frames 110..299
        self.assertLess(m.tracked_s, m.duration_s)

    def test_fully_untracked_trial_is_nan_not_raised(self):
        rec = _hold_rec(_swaying(n=90), visibility=0.0)
        m = hold_metrics(rec, _seg(0, 90))
        self.assertEqual(m.coverage, 0.0)
        self.assertEqual(m.tracked_s, 0.0)
        self.assertTrue(np.isnan(m.path_length_m))
        self.assertTrue(np.isnan(m.rms_m))
        self.assertTrue(np.isnan(m.trunk_angle_mean_deg))

    def test_no_pose_frames_are_excluded(self):
        world = body_world(_swaying(n=100))
        world[50:60] = np.nan
        vis = np.ones((100, 33), dtype=np.float32)
        vis[50:60] = np.nan
        rec = fake_recording(world, visibility=vis, presence=vis)
        m = hold_metrics(rec, _seg(0, 100))
        self.assertAlmostEqual(m.coverage, 0.9, places=6)
        self.assertTrue(np.isfinite(m.rms_m))

    # -- edges that bite ---------------------------------------------------------- #
    def test_empty_segment_is_nan_not_raised(self):
        m = hold_metrics(_hold_rec(_swaying(n=90)), _seg(5, 5))
        self.assertEqual(m.n_frames, 0)
        self.assertEqual(m.duration_s, 0.0)
        self.assertEqual(m.coverage, 0.0)
        self.assertTrue(np.isnan(m.path_length_m))

    def test_single_frame_segment_is_nan_not_raised(self):
        m = hold_metrics(_hold_rec(_swaying(n=90)), _seg(10, 11))
        self.assertEqual(m.duration_s, 0.0)
        self.assertTrue(np.isnan(m.rms_m))

    def test_segment_shorter_than_the_smoothing_window_is_nan_not_raised(self):
        # ~0.1 s of frames, under the 0.233 s window.
        m = hold_metrics(_hold_rec(_swaying(n=90)), _seg(0, 4))
        self.assertTrue(np.isnan(m.path_length_m))
        self.assertTrue(np.isnan(m.ellipse_area_m2))
        self.assertGreater(m.coverage, 0.0)  # the frames were fine, just too few

    # -- hands_low_frac is a diagnostic, not a metric ------------------------------ #
    def test_hands_low_frac_flags_hands_below_the_pelvis(self):
        # y is DOWN in world coords, so +y is below the pelvis.
        rec = _hold_rec(_swaying(n=90), wrist_y=0.1)
        self.assertAlmostEqual(hold_metrics(rec, _seg(0, 90)).hands_low_frac, 1.0)

    def test_hands_low_frac_is_zero_for_raised_hands(self):
        rec = _hold_rec(_swaying(n=90), wrist_y=-0.2)
        self.assertAlmostEqual(hold_metrics(rec, _seg(0, 90)).hands_low_frac, 0.0)

    def test_hands_low_frac_is_nan_when_wrists_are_untracked(self):
        # Must not poison the rest of the row -- a sitting trial stays measurable with
        # the hands out of frame.
        vis = np.ones((90, 33), dtype=np.float32)
        vis[:, 15] = vis[:, 16] = 0.0
        rec = _hold_rec(_swaying(n=90), visibility=vis, wrist_y=0.1)
        m = hold_metrics(rec, _seg(0, 90))
        self.assertTrue(np.isnan(m.hands_low_frac))
        self.assertTrue(np.isfinite(m.rms_m))


def _bell(n=90, centre=0.5, width=0.12, fps=30.0, amp=1.0):
    """A bell-shaped speed profile: one smooth movement."""
    t = np.arange(n) / fps
    span = t[-1]
    return amp * np.exp(-(((t - centre * span) / (width * span)) ** 2))


class SparcTests(unittest.TestCase):
    # SPARC's absolute value is a property of this pipeline, not of Remy, so these
    # tests pin ORDERING and INVARIANCE only. Asserting a number would pin the noise.

    def test_robust_to_small_measurement_noise(self):
        # SPARC's headline advantage over dimensionless jerk, and the reason it is
        # usable at all on 30 Hz landmark data where a derivative is mostly noise.
        # Measured on a unit bell: sigma=0.02 leaves the score put (sd 0.006).
        rng = np.random.default_rng(0)
        clean = _bell()
        for _ in range(5):
            noisy = clean + rng.normal(0, 0.02, clean.shape)
            self.assertAlmostEqual(sparc(clean), sparc(noisy), delta=0.05)

    def test_noise_robustness_has_a_ceiling(self):
        # The other half of the truth, and the one that constrains real use: past
        # roughly 2% of peak speed the robustness gives out. Measured means on a unit
        # bell -- 0.02 -> -1.41 (sd 0.006), 0.05 -> -1.68 (sd 0.24), 0.10 -> -2.53.
        # So a SPARC score is only worth reading when the speed profile's noise is
        # small against its peak; see the module docstring.
        rng = np.random.default_rng(7)
        clean = _bell()
        quiet = np.mean([sparc(clean + rng.normal(0, 0.02, 90)) for _ in range(20)])
        loud = np.mean([sparc(clean + rng.normal(0, 0.10, 90)) for _ in range(20)])
        self.assertAlmostEqual(quiet, sparc(clean), delta=0.05)
        self.assertLess(loud, sparc(clean) - 0.5)

    def test_one_movement_beats_two_submovements(self):
        # The clinical signal SPARC is actually for: an effortful movement is a chain
        # of corrections, and that shows up as extra humps in the speed profile.
        one = _bell(n=120, centre=0.5, width=0.10)
        two = _bell(n=120, centre=0.3, width=0.07) + _bell(n=120, centre=0.7, width=0.07)
        self.assertGreater(sparc(one), sparc(two))

    def test_one_movement_beats_three_submovements(self):
        one = _bell(n=240, centre=0.5, width=0.10)
        three = sum(_bell(n=240, centre=c, width=0.05) for c in (0.25, 0.5, 0.75))
        self.assertGreater(sparc(one), sparc(three))

    def test_more_noise_is_less_smooth(self):
        rng = np.random.default_rng(1)
        clean = _bell()
        a = clean + rng.normal(0, 0.02, clean.shape)
        b = clean + rng.normal(0, 0.10, clean.shape)
        self.assertGreater(sparc(a), sparc(b))

    def test_is_negative(self):
        self.assertLess(sparc(_bell()), 0.0)

    def test_invariant_to_amplitude_scaling(self):
        # The property that makes it usable without normalizing: a bigger movement of
        # the same shape is equally smooth.
        self.assertAlmostEqual(sparc(_bell(amp=1.0)), sparc(_bell(amp=17.0)), places=9)

    def test_zero_movement_is_nan(self):
        self.assertTrue(np.isnan(sparc(np.zeros(90))))

    def test_nan_input_is_nan(self):
        v = _bell()
        v[10] = np.nan
        self.assertTrue(np.isnan(sparc(v)))

    def test_too_short_is_nan_not_raised(self):
        self.assertTrue(np.isnan(sparc(np.array([1.0]))))

    def test_empty_is_nan_not_raised(self):
        self.assertTrue(np.isnan(sparc(np.array([]))))


class CountSubmovementsTests(unittest.TestCase):
    def test_one_bell_is_one_submovement(self):
        self.assertEqual(count_submovements(_bell()), 1)

    def test_two_bells_are_two_submovements(self):
        two = _bell(n=120, centre=0.3, width=0.06) + _bell(n=120, centre=0.7, width=0.06)
        self.assertEqual(count_submovements(two), 2)

    def test_three_bells_are_three_submovements(self):
        three = sum(_bell(n=240, centre=c, width=0.04) for c in (0.25, 0.5, 0.75))
        self.assertEqual(count_submovements(three), 3)

    def test_jitter_does_not_read_as_submovements(self):
        # Without the prominence gate this would count dozens.
        rng = np.random.default_rng(2)
        self.assertEqual(count_submovements(_bell() + rng.normal(0, 0.01, 90)), 1)

    def test_flat_profile_has_none(self):
        self.assertEqual(count_submovements(np.zeros(90)), 0)

    def test_unusable_input_is_zero_not_raised(self):
        self.assertEqual(count_submovements(np.array([])), 0)
        self.assertEqual(count_submovements(np.array([1.0, np.nan, 2.0])), 0)


class SymmetryIndexTests(unittest.TestCase):
    def test_equal_sides_are_symmetric(self):
        self.assertAlmostEqual(symmetry_index([2.0, 2.0], [2.0, 2.0]), 0.0)

    def test_sign_follows_the_larger_side(self):
        self.assertGreater(symmetry_index([3.0], [1.0]), 0.0)
        self.assertLess(symmetry_index([1.0], [3.0]), 0.0)

    def test_closed_form(self):
        # 2*(3-1)/(3+1) = 1.0
        self.assertAlmostEqual(symmetry_index([3.0], [1.0]), 1.0)

    def test_uses_medians_so_one_outlier_does_not_swing_it(self):
        self.assertAlmostEqual(symmetry_index([2.0, 2.0, 2.0, 99.0], [2.0]), 0.0)

    def test_ignores_nan_trials(self):
        self.assertAlmostEqual(symmetry_index([2.0, np.nan], [2.0]), 0.0)

    def test_missing_side_is_nan(self):
        self.assertTrue(np.isnan(symmetry_index([], [1.0])))
        self.assertTrue(np.isnan(symmetry_index([1.0], [])))
        self.assertTrue(np.isnan(symmetry_index([np.nan], [1.0])))

    def test_zero_sum_is_nan(self):
        self.assertTrue(np.isnan(symmetry_index([0.0], [0.0])))


def _rotating(n=90, start_deg=0.0, end_deg=90.0, fps=30.0, submovements=1):
    """Trunk rotating from `start_deg` to `end_deg` with a smooth (or broken) profile."""
    t = np.linspace(0, 1, n)
    if submovements == 1:
        # Minimum-jerk profile: the canonical single smooth movement.
        progress = 10 * t**3 - 15 * t**4 + 6 * t**5
    else:
        steps = np.linspace(0, 1, submovements + 1)[1:]
        progress = np.mean(
            [np.clip((t - s + 0.12) / 0.12, 0, 1) for s in steps], axis=0
        )
    angle = np.radians(start_deg + (end_deg - start_deg) * progress)
    trunk = np.stack([np.sin(angle), -np.cos(angle), np.zeros(n)], axis=1) * 0.4
    return trunk


def _trans_seg(start, stop, label="transition;from=prone;to=sit;side=left"):
    return Segment(
        ann=Annotation(0, label, 0, 0), parsed=parse_label(label), start=start, stop=stop
    )


class TransitionMetricsTests(unittest.TestCase):
    def test_smooth_transition_reads_as_one_submovement(self):
        rec = fake_recording(body_world(_rotating(n=90)))
        m = transition_metrics(rec, _trans_seg(0, 90))
        self.assertEqual(m.n_velocity_peaks, 1)
        self.assertTrue(np.isfinite(m.sparc_trunk))

    def test_broken_transition_is_less_smooth_than_a_fluid_one(self):
        smooth_rec = fake_recording(body_world(_rotating(n=120, submovements=1)))
        jerky_rec = fake_recording(body_world(_rotating(n=120, submovements=4)))
        fluid = transition_metrics(smooth_rec, _trans_seg(0, 120))
        jerky = transition_metrics(jerky_rec, _trans_seg(0, 120))
        self.assertGreater(fluid.sparc_trunk, jerky.sparc_trunk)
        self.assertGreater(jerky.n_velocity_peaks, fluid.n_velocity_peaks)

    def test_peak_angular_velocity_is_positive_for_a_real_rotation(self):
        rec = fake_recording(body_world(_rotating(n=90, start_deg=0, end_deg=90)))
        m = transition_metrics(rec, _trans_seg(0, 90))
        self.assertGreater(m.peak_angular_velocity_dps, 0.0)

    def test_faster_transition_has_higher_peak_angular_velocity(self):
        slow = transition_metrics(
            fake_recording(body_world(_rotating(n=180))), _trans_seg(0, 180)
        )
        fast = transition_metrics(
            fake_recording(body_world(_rotating(n=45))), _trans_seg(0, 45)
        )
        self.assertGreater(fast.peak_angular_velocity_dps, slow.peak_angular_velocity_dps)

    def test_movement_duration_excludes_the_still_margins(self):
        # The annotated span includes the annotator's reaction time at both ends.
        still = np.tile([0.0, -0.4, 0.0], (45, 1))
        world = body_world(np.concatenate([still, _rotating(n=90), still]))
        m = transition_metrics(fake_recording(world), _trans_seg(0, 180))
        self.assertGreater(m.duration_s, m.movement_duration_s)
        self.assertGreater(m.movement_duration_s, 0.0)

    def test_stationary_trial_has_no_movement_duration(self):
        rec = fake_recording(body_world(np.tile([0.0, -0.4, 0.0], (90, 1))))
        m = transition_metrics(rec, _trans_seg(0, 90))
        self.assertTrue(np.isnan(m.movement_duration_s))
        self.assertTrue(np.isnan(m.sparc_trunk))

    def test_side_comes_from_the_label_not_the_pixels(self):
        rec = fake_recording(body_world(_rotating(n=90)))
        m = transition_metrics(rec, _trans_seg(0, 90))
        self.assertEqual(m.side, "left")
        unlabelled = transition_metrics(
            rec, _trans_seg(0, 90, "transition;from=sit;to=prone")
        )
        self.assertIsNone(unlabelled.side)

    def test_leading_wrist_is_the_more_lateral_one(self):
        n = 90
        left = np.stack([np.full(n, 0.4), np.zeros(n), np.zeros(n)], axis=1)
        right = np.stack([np.full(n, -0.05), np.zeros(n), np.zeros(n)], axis=1)
        world = body_world(_rotating(n=n), left_wrist=left, right_wrist=right)
        m = transition_metrics(fake_recording(world), _trans_seg(0, n))
        self.assertEqual(m.leading_wrist, "left")

    def test_leading_wrist_is_none_when_wrists_are_untracked(self):
        vis = np.ones((90, 33), dtype=np.float32)
        vis[:, 15] = vis[:, 16] = 0.0
        rec = fake_recording(body_world(_rotating(n=90)), visibility=vis)
        m = transition_metrics(rec, _trans_seg(0, 90))
        self.assertIsNone(m.leading_wrist)
        self.assertTrue(np.isfinite(m.sparc_trunk))  # torso metrics unaffected

    def test_coverage_and_gating(self):
        vis = np.ones((100, 33), dtype=np.float32)
        vis[80:100, TORSO[0]] = 0.0
        rec = fake_recording(body_world(_rotating(n=100)), visibility=vis)
        m = transition_metrics(rec, _trans_seg(0, 100))
        self.assertAlmostEqual(m.coverage, 0.8, places=6)

    def test_up_source_is_recorded(self):
        rec = fake_recording(body_world(_rotating(n=90)))
        self.assertEqual(transition_metrics(rec, _trans_seg(0, 90)).up_source, "world_y")
        tilted = np.array([0.1, -1.0, 0.0])
        self.assertEqual(
            transition_metrics(rec, _trans_seg(0, 90), up=tilted).up_source, "custom"
        )

    # -- edges -------------------------------------------------------------------- #
    def test_empty_segment_is_nan_not_raised(self):
        rec = fake_recording(body_world(_rotating(n=90)))
        m = transition_metrics(rec, _trans_seg(5, 5))
        self.assertEqual(m.n_frames, 0)
        self.assertTrue(np.isnan(m.sparc_trunk))
        self.assertEqual(m.n_velocity_peaks, 0)

    def test_short_segment_is_nan_not_raised(self):
        rec = fake_recording(body_world(_rotating(n=90)))
        m = transition_metrics(rec, _trans_seg(0, 4))
        self.assertTrue(np.isnan(m.sparc_trunk))
        self.assertTrue(np.isnan(m.sparc_tip))

    def test_fully_untracked_is_nan_not_raised(self):
        rec = fake_recording(body_world(_rotating(n=90)), visibility=0.0)
        m = transition_metrics(rec, _trans_seg(0, 90))
        self.assertEqual(m.coverage, 0.0)
        self.assertTrue(np.isnan(m.sparc_trunk))


def _crawl_rec(n=300, freq=1.0, phase_frac=0.5, fps=30.0, amp_l=0.10, amp_r=0.10,
               travel=0.0):
    """A belly-crawling body: wrists oscillating along the body axis.

    The trunk points along -x (prone, head-first), so the body's long axis is x and each
    wrist's reach is an oscillation along it. `phase_frac` is the left-right phase
    offset in cycles: 0.5 alternating, 0.0 together. `travel` moves the pelvis across
    the image (in normalized units) to exercise the image-space speed.
    """
    t = np.arange(n) / fps
    trunk = np.tile([-0.4, 0.0, 0.0], (n, 1))  # prone: body axis along -x

    def wrist(amp, phase):
        reach = -0.25 - amp * np.sin(2 * np.pi * freq * t + phase)
        return np.stack([reach, np.zeros(n), np.zeros(n)], axis=1)

    world = body_world(
        trunk,
        left_wrist=wrist(amp_l, 0.0),
        right_wrist=wrist(amp_r, 2 * np.pi * phase_frac),
    )
    norm = np.zeros((n, 33, 3), dtype=np.float32)
    x = 0.2 + travel * (t / t[-1] if t[-1] > 0 else 0)
    norm[:, 23, 0] = norm[:, 24, 0] = x
    norm[:, 23, 1] = norm[:, 24, 1] = 0.5
    return fake_recording(world, norm=norm, fps=fps)


def _crawl_seg(start, stop, label="crawl;style=belly;dir=away"):
    return Segment(
        ann=Annotation(0, label, 0, 0), parsed=parse_label(label), start=start, stop=stop
    )


class LimbSignalTests(unittest.TestCase):
    def test_projects_the_wrist_onto_the_body_axis(self):
        rec = _crawl_rec(n=300, freq=1.0, amp_l=0.1)
        sig = limb_signal(rec, _crawl_seg(0, 300), "left")
        self.assertEqual(sig.shape, (300,))
        # The wrist oscillates +/-0.1 m about 0.25 m along the axis toward the head.
        self.assertAlmostEqual(float(np.ptp(sig)), 0.2, places=2)
        self.assertAlmostEqual(float(np.mean(sig)), 0.25, places=2)

    def test_is_measured_relative_to_the_pelvis_so_travel_cannot_leak_in(self):
        # The property that makes cadence immune to the missing-translation problem.
        rec = _crawl_rec(n=120)
        shifted = fake_recording(
            rec.landmarks_world + np.array([3.0, -2.0, 1.0], dtype=np.float32)
        )
        np.testing.assert_allclose(
            limb_signal(rec, _crawl_seg(0, 120), "left"),
            limb_signal(shifted, _crawl_seg(0, 120), "left"),
            atol=1e-4,
        )

    def test_left_and_right_are_distinct(self):
        rec = _crawl_rec(n=300, amp_l=0.10, amp_r=0.02)
        left = limb_signal(rec, _crawl_seg(0, 300), "left")
        right = limb_signal(rec, _crawl_seg(0, 300), "right")
        self.assertGreater(np.ptp(left), np.ptp(right) * 3)

    def test_knee_marker(self):
        rec = _crawl_rec(n=60)
        self.assertEqual(limb_signal(rec, _crawl_seg(0, 60), "left", "knee").shape, (60,))

    def test_bad_side_or_marker_raises(self):
        rec = _crawl_rec(n=60)
        with self.assertRaises(ValueError):
            limb_signal(rec, _crawl_seg(0, 60), "middle")
        with self.assertRaises(ValueError):
            limb_signal(rec, _crawl_seg(0, 60), "left", "elbow")

    def test_empty_segment_is_empty(self):
        self.assertEqual(limb_signal(_crawl_rec(n=60), _crawl_seg(5, 5), "left").size, 0)


class CyclesTests(unittest.TestCase):
    def test_recovers_a_known_cycle_count(self):
        t = np.arange(300) / FS
        for freq in (0.5, 1.0, 2.0):
            sig = np.sin(2 * np.pi * freq * t)
            expected = int(freq * (299 / FS))  # whole peaks in the window
            self.assertAlmostEqual(cycles(sig).size, expected, delta=1, msg=f"{freq} Hz")

    def test_jitter_on_a_still_arm_is_not_a_crawl(self):
        # The bug the absolute excursion floor exists for. The prominence gate alone is
        # relative to the signal's own range, so it normalizes pure noise up into a
        # textbook crawl: this read 57 confident cycles before MIN_CYCLE_EXCURSION_M.
        rng = np.random.default_rng(0)
        self.assertEqual(cycles(rng.normal(0, 1e-4, 300)).size, 0)
        self.assertEqual(cycles(rng.normal(0, 0.002, 300)).size, 0)

    def test_real_sized_pull_survives_the_excursion_floor(self):
        # The floor must not swallow a genuine crawl -- a pull travels ~10-20 cm.
        t = np.arange(300) / FS
        self.assertGreater(cycles(0.06 * np.sin(2 * np.pi * 1.0 * t)).size, 5)

    def test_excursion_floor_is_in_metres_not_relative(self):
        t = np.arange(300) / FS
        wave = np.sin(2 * np.pi * 1.0 * t)
        self.assertEqual(cycles(0.005 * wave).size, 0)  # 1 cm peak-to-peak: not a pull
        self.assertGreater(cycles(0.05 * wave).size, 5)  # 10 cm: a pull

    def test_flat_signal_has_no_cycles(self):
        self.assertEqual(cycles(np.zeros(300)).size, 0)

    def test_unusable_input_is_empty_not_raised(self):
        self.assertEqual(cycles(np.array([])).size, 0)
        self.assertEqual(cycles(np.array([1.0, np.nan, 2.0])).size, 0)


class PhaseOffsetTests(unittest.TestCase):
    def setUp(self):
        self.t = np.arange(600) / FS

    def _sig(self, phase):
        return np.sin(2 * np.pi * 1.0 * self.t + phase)

    def test_antiphase_is_reciprocal(self):
        offset, sd = phase_offset(self._sig(0), self._sig(np.pi))
        self.assertAlmostEqual(offset, 0.5, places=2)
        self.assertLess(sd, 0.1)

    def test_inphase_is_symmetric(self):
        offset, sd = phase_offset(self._sig(0), self._sig(0))
        self.assertAlmostEqual(offset, 0.0, places=2)
        self.assertLess(sd, 0.1)

    def test_quarter_cycle_offset(self):
        offset, _ = phase_offset(self._sig(0), self._sig(np.pi / 2))
        self.assertAlmostEqual(offset, 0.25, places=2)

    def test_offset_is_direction_agnostic(self):
        # Which arm is nominally "first" is arbitrary; leading and lagging by a third of
        # a cycle are the same amount of reciprocity.
        lead, _ = phase_offset(self._sig(0), self._sig(2 * np.pi / 3))
        lag, _ = phase_offset(self._sig(0), self._sig(-2 * np.pi / 3))
        self.assertAlmostEqual(lead, lag, places=2)

    def test_circular_sd_flags_an_unsettled_pattern(self):
        # Two arms at different frequencies have no stable phase relationship at all.
        drifting = np.sin(2 * np.pi * 1.4 * self.t)
        _, steady_sd = phase_offset(self._sig(0), self._sig(np.pi))
        _, drift_sd = phase_offset(self._sig(0), drifting)
        self.assertGreater(drift_sd, steady_sd * 3)

    def test_circular_mean_does_not_average_across_the_wrap(self):
        # A plain mean of angles would put the average of ~0 and ~2pi at pi.
        offset, _ = phase_offset(self._sig(0), self._sig(2 * np.pi - 0.01))
        self.assertAlmostEqual(offset, 0.0, places=2)

    def test_unusable_input_is_nan_not_raised(self):
        self.assertTrue(np.isnan(phase_offset(np.array([]), np.array([]))[0]))
        self.assertTrue(np.isnan(phase_offset(np.zeros(300), np.zeros(300))[0]))
        self.assertTrue(np.isnan(phase_offset(self._sig(0), np.zeros(600))[0]))
        self.assertTrue(np.isnan(phase_offset(self._sig(0), self._sig(0)[:10])[0]))

    def test_still_arms_have_no_phase_relationship(self):
        # Jitter has a phase, but it is not a crawl pattern; inventing one would be
        # worse than reporting nothing.
        rng = np.random.default_rng(3)
        a = rng.normal(0, 0.001, 600)
        b = rng.normal(0, 0.001, 600)
        self.assertTrue(np.isnan(phase_offset(a, b)[0]))

    def test_no_warning_on_a_perfectly_locked_pair(self):
        # |resultant| can round just above 1.0, and log() then makes the sqrt complain.
        with np.errstate(all="raise"):
            offset, sd = phase_offset(self._sig(0), self._sig(0))
        self.assertTrue(np.isfinite(offset))
        self.assertTrue(np.isfinite(sd))


class CrawlMetricsTests(unittest.TestCase):
    def test_cadence_matches_the_constructed_frequency(self):
        rec = _crawl_rec(n=600, freq=1.0)  # 1 Hz = 60 pulls/min
        m = crawl_metrics(rec, _crawl_seg(0, 600))
        self.assertAlmostEqual(m.cadence_cpm_left, 60.0, delta=4)
        self.assertAlmostEqual(m.cadence_cpm_right, 60.0, delta=4)
        self.assertAlmostEqual(m.cadence_cpm, 60.0, delta=4)

    def test_slower_crawl_reads_as_lower_cadence(self):
        slow = crawl_metrics(_crawl_rec(n=600, freq=0.5), _crawl_seg(0, 600))
        fast = crawl_metrics(_crawl_rec(n=600, freq=1.5), _crawl_seg(0, 600))
        self.assertLess(slow.cadence_cpm, fast.cadence_cpm)
        self.assertAlmostEqual(slow.cadence_cpm, 30.0, delta=4)
        self.assertAlmostEqual(fast.cadence_cpm, 90.0, delta=6)

    def test_reciprocal_crawl_reads_as_reciprocal(self):
        # The developmental axis: alternating arms is the mature pattern.
        m = crawl_metrics(_crawl_rec(n=600, phase_frac=0.5), _crawl_seg(0, 600))
        self.assertAlmostEqual(m.phase_offset, 0.5, places=1)
        self.assertLess(m.phase_offset_circular_sd, 0.2)

    def test_symmetric_bunny_haul_reads_as_symmetric(self):
        m = crawl_metrics(_crawl_rec(n=600, phase_frac=0.0), _crawl_seg(0, 600))
        self.assertAlmostEqual(m.phase_offset, 0.0, places=1)

    def test_cycle_counts(self):
        m = crawl_metrics(_crawl_rec(n=600, freq=1.0), _crawl_seg(0, 600))
        self.assertAlmostEqual(m.n_cycles_left, 20, delta=2)
        self.assertAlmostEqual(m.n_cycles_right, 20, delta=2)

    def test_metronomic_crawl_has_low_period_variability(self):
        m = crawl_metrics(_crawl_rec(n=600, freq=1.0), _crawl_seg(0, 600))
        self.assertLess(m.cycle_period_cv, 0.1)
        self.assertLess(m.cycle_period_sd_s, 0.1)

    def test_amplitude_symmetry_flags_a_lopsided_pull(self):
        even = crawl_metrics(_crawl_rec(n=600, amp_l=0.1, amp_r=0.1), _crawl_seg(0, 600))
        lopsided = crawl_metrics(
            _crawl_rec(n=600, amp_l=0.12, amp_r=0.04), _crawl_seg(0, 600)
        )
        self.assertAlmostEqual(even.amplitude_symmetry, 0.0, places=1)
        self.assertGreater(lopsided.amplitude_symmetry, 0.5)

    def test_speed_is_reported_in_image_fractions_not_metres(self):
        # 0.5 image widths over the trial's ~20 s.
        rec = _crawl_rec(n=600, travel=0.5)
        m = crawl_metrics(rec, _crawl_seg(0, 600))
        self.assertAlmostEqual(m.speed_norm_per_s, 0.5 / m.tracked_s, places=2)

    def test_stationary_pelvis_has_no_image_speed(self):
        m = crawl_metrics(_crawl_rec(n=300, travel=0.0), _crawl_seg(0, 300))
        self.assertAlmostEqual(m.speed_norm_per_s, 0.0, places=4)

    def test_cadence_is_unaffected_by_how_far_he_travels(self):
        # Cadence lives in the pelvis-relative frame; image travel cannot touch it.
        still = crawl_metrics(_crawl_rec(n=600, travel=0.0), _crawl_seg(0, 600))
        moving = crawl_metrics(_crawl_rec(n=600, travel=0.6), _crawl_seg(0, 600))
        self.assertAlmostEqual(still.cadence_cpm, moving.cadence_cpm, places=6)

    # -- gating and edges --------------------------------------------------------- #
    def test_coverage_and_gating_on_both_wrists(self):
        vis = np.ones((300, 33), dtype=np.float32)
        vis[240:300, 15] = 0.0  # left wrist lost -- a crawl metric needs both
        rec = _crawl_rec(n=300)
        rec.visibility = vis
        m = crawl_metrics(rec, _crawl_seg(0, 300))
        self.assertAlmostEqual(m.coverage, 0.8, places=6)

    def test_empty_segment_is_nan_not_raised(self):
        m = crawl_metrics(_crawl_rec(n=300), _crawl_seg(5, 5))
        self.assertEqual(m.n_frames, 0)
        self.assertTrue(np.isnan(m.cadence_cpm))
        self.assertTrue(np.isnan(m.phase_offset))

    def test_short_segment_is_nan_not_raised(self):
        m = crawl_metrics(_crawl_rec(n=300), _crawl_seg(0, 4))
        self.assertTrue(np.isnan(m.cadence_cpm))
        self.assertTrue(np.isnan(m.speed_norm_per_s))

    def test_fully_untracked_is_nan_not_raised(self):
        rec = _crawl_rec(n=300)
        rec.visibility = np.zeros((300, 33), dtype=np.float32)
        m = crawl_metrics(rec, _crawl_seg(0, 300))
        self.assertEqual(m.coverage, 0.0)
        self.assertTrue(np.isnan(m.cadence_cpm))

    def test_still_child_is_not_a_crawl(self):
        rec = _crawl_rec(n=300, amp_l=0.0, amp_r=0.0)
        m = crawl_metrics(rec, _crawl_seg(0, 300))
        self.assertEqual(m.n_cycles_left, 0)
        self.assertTrue(np.isnan(m.cadence_cpm))


class MetricsTableTests(unittest.TestCase):
    def _rec(self, labels, n=300):
        rec = _hold_rec(_swaying(n=n))
        rec.annotations = [
            Annotation(i, label, start, end)
            for i, (label, start, end) in enumerate(labels)
        ]
        return rec

    def test_one_row_per_trial(self):
        rec = self._rec(
            [
                ("sit_hold;arms=free", 0, 2000),
                ("sit_hold;arms=prop", 3000, 5000),
                ("stand_hold;support=trunk", 6000, 8000),
            ]
        )
        table = metrics_table(rec)
        self.assertEqual(len(table), 3)
        self.assertEqual(list(table["exercise"]), ["sit_hold", "sit_hold", "stand_hold"])

    def test_calib_and_exclude_produce_no_rows(self):
        rec = self._rec([("calib;pose=upright", 0, 1000), ("sit_hold", 2000, 4000)])
        table = metrics_table(rec)
        self.assertEqual(list(table["exercise"]), ["sit_hold"])

    def test_gmfm_dimension_is_carried(self):
        rec = self._rec([("sit_hold", 0, 2000), ("stand_hold;support=trunk", 3000, 5000)])
        table = metrics_table(rec)
        self.assertEqual(list(table["dimension"]), ["B", "D"])

    def test_label_params_are_prefixed_to_avoid_colliding_with_metrics(self):
        rec = self._rec([("sit_hold;arms=free;support=none;gmfm=23", 0, 2000)])
        table = metrics_table(rec)
        self.assertEqual(table.loc[0, "p_arms"], "free")
        self.assertEqual(table.loc[0, "p_support"], "none")
        self.assertEqual(table.loc[0, "p_gmfm"], "23")

    def test_lead_columns_come_first(self):
        # A reader should meet `coverage` before any number it qualifies.
        rec = self._rec([("sit_hold;arms=free", 0, 2000)])
        cols = list(metrics_table(rec).columns)
        self.assertEqual(cols[:4], ["session", "exercise", "dimension", "label"])
        self.assertIn("coverage", cols[:10])
        self.assertEqual(cols[-1], "warnings")

    def test_label_typos_surface_in_the_warnings_column(self):
        rec = self._rec([("sit_hold;arms=freee", 0, 2000)])
        table = metrics_table(rec)
        self.assertIn("freee", table.loc[0, "warnings"])

    def test_clean_labels_warn_about_nothing(self):
        rec = self._rec([("sit_hold;arms=free;support=none", 0, 2000)])
        self.assertEqual(metrics_table(rec).loc[0, "warnings"], "")

    def test_columns_are_the_union_across_exercise_types(self):
        # Four separate tables could not be concatenated for a trend.
        rec = self._rec([("sit_hold;arms=free", 0, 2000), ("crawl;style=belly", 3000, 6000)])
        table = metrics_table(rec)
        self.assertIn("ellipse_area_m2", table.columns)
        self.assertIn("cadence_cpm", table.columns)
        self.assertTrue(np.isnan(table.loc[0, "cadence_cpm"]))  # the sitting row
        self.assertTrue(np.isnan(table.loc[1, "ellipse_area_m2"]))  # the crawl row

    def test_shared_columns_are_populated_for_every_exercise(self):
        # These are the lead columns a reader scans first, so a per-exercise hole in one
        # of them is a wart: transitions used to call duration_s `annotated_duration_s`
        # and left a NaN in the column for every transition row.
        rec = self._rec(
            [
                ("sit_hold;arms=free", 0, 2000),
                ("transition;from=sit;to=prone", 3000, 5000),
                ("stand_hold;support=trunk", 6000, 8000),
                ("crawl;style=belly", 8500, 9900),
            ]
        )
        table = metrics_table(rec)
        self.assertEqual(len(table), 4)
        for column in ("duration_s", "tracked_s", "coverage", "n_frames"):
            self.assertFalse(
                table[column].isna().any(), msg=f"{column} has a per-exercise hole"
            )

    def test_window_s_works_on_a_mixed_session(self):
        # Regression: metrics_table forwarded **kwargs blindly to every metric, so
        # `--window-s` reached transition_metrics() -- which takes no window_s -- and
        # raised TypeError on any session that contained a transition or a crawl.
        rec = self._rec(
            [
                ("sit_hold;arms=free", 0, 5000),
                ("transition;from=sit;to=prone", 5500, 7000),
                ("crawl;style=belly", 7500, 9900),
            ]
        )
        table = metrics_table(rec, window_s=2.0)
        self.assertEqual(len(table), 3)
        # Only the hold is truncated: windowing a discrete transition would cut the
        # movement in half, and crawl metrics are already duration-free.
        holds = table[table["exercise"] == "sit_hold"]
        self.assertAlmostEqual(holds.iloc[0]["tracked_s"], 2.0, places=1)
        others = table[table["exercise"] != "sit_hold"]
        self.assertTrue((others["tracked_s"] > 1.0).all())

    def test_window_s_reaches_session_table_too(self):
        rec = self._rec([("sit_hold;arms=free", 0, 5000)])
        self.assertAlmostEqual(
            metrics_table(rec, window_s=2.0).iloc[0]["tracked_s"], 2.0, places=1
        )

    def test_up_defaults_to_world_y_and_is_recorded(self):
        rec = self._rec([("sit_hold;arms=free", 0, 2000)])
        self.assertEqual(metrics_table(rec).loc[0, "up_source"], "world_y")

    def test_custom_up_is_honoured_and_recorded(self):
        rec = self._rec([("sit_hold;arms=free", 0, 2000)])
        table = metrics_table(rec, up=np.array([0.2, -1.0, 0.0]))
        self.assertEqual(table.loc[0, "up_source"], "custom")

    def test_gate_is_forwarded(self):
        rec = self._rec([("sit_hold;arms=free", 0, 2000)])
        table = metrics_table(rec, gate=Gate(min_visibility=1.1))  # nothing can pass
        self.assertEqual(table.loc[0, "coverage"], 0.0)

    def test_trial_exercises_is_a_strict_subset_of_the_vocabulary(self):
        # Housekeeping labels never produce a row, so the CLI must not offer them as a
        # --exercise filter: that is just a way to ask for an empty table.
        self.assertTrue(set(TRIAL_EXERCISES) < set(EXERCISES))
        self.assertNotIn("calib", TRIAL_EXERCISES)
        self.assertNotIn("exclude", TRIAL_EXERCISES)

    def test_no_trials_gives_an_empty_frame_not_an_error(self):
        self.assertTrue(metrics_table(self._rec([("walking", 0, 2000)])).empty)
        self.assertTrue(metrics_table(self._rec([])).empty)


class ReportIntegrationTests(unittest.TestCase):
    """The whole path: real recorder -> real annotations -> real reader -> table."""

    def _write(self, path, n=200):
        trunk = _swaying(n=n, amp=0.02)
        world = body_world(trunk)
        with HDF5Recorder(path) as rec:
            for i in range(n):
                norm = [FakeLandmark(x=0.5, y=0.5, visibility=1.0) for _ in range(33)]
                world_lms = [
                    FakeLandmark(x=float(x), y=float(y), z=float(z))
                    for x, y, z in world[i]
                ]
                rec.append(solid_frame(), i * 33, pose_result([norm], [world_lms]))

    def test_end_to_end_metrics_from_a_real_file(self):
        with TemporaryDirectory() as d:
            path = Path(d) / "session.h5"
            self._write(path)
            store = AnnotationStore(path)  # "r+" before "r" -- h5py locking
            store.add("sit_hold;arms=free;support=none", 0, 3000)
            store.add("exclude;reason=dog in shot", 4000, 4500)
            store.add("crawl;style=belly", 4200, 5000)  # overlaps the exclude -> dropped
            rec = Recording(path)
            try:
                table = metrics_table(rec, session="session")
                self.assertEqual(len(table), 1)
                row = table.iloc[0]
                self.assertEqual(row["exercise"], "sit_hold")
                self.assertEqual(row["session"], "session")
                self.assertEqual(row["dimension"], "B")
                self.assertAlmostEqual(row["coverage"], 1.0)
                self.assertAlmostEqual(row["duration_s"], 3.0, delta=0.1)
                self.assertGreater(row["sway_ml_rms_m"], 0.005)
                self.assertTrue(np.isfinite(row["ellipse_area_m2"]))
            finally:
                rec.close()
                store.close()

    def test_session_table_concatenates_for_a_trend(self):
        with TemporaryDirectory() as d:
            paths = []
            for name in ("day1", "day2"):
                path = Path(d) / f"{name}.h5"
                self._write(path)
                with AnnotationStore(path) as store:
                    store.add("sit_hold;arms=free", 0, 3000)
                paths.append(path)
            table = session_table(paths)
            self.assertEqual(len(table), 2)
            self.assertEqual(sorted(table["session"]), ["day1", "day2"])

    def test_session_table_with_no_paths_is_empty(self):
        self.assertTrue(session_table([]).empty)


if __name__ == "__main__":
    unittest.main()
