"""Tests for the annotate GUI's pure drawing/query logic.

The CLI loop itself is not tested (no window, no keyboard), but everything it draws
with is: :func:`pose_estimation.draw.draw_skeleton` is a reusable library function, and
``_active_labels``/``_render`` are pure enough to exercise headlessly.

The NaN cases are the point of most of this: ``annotate`` scrubs freely across frames
that ``HDF5Recorder`` wrote as full-NaN rows, which the previous drawing code
(``pose_estimation.main.draw_pose``) would have crashed on via ``int(nan)``.
"""

import unittest

import numpy as np

from annotate.main import _active_labels, _render, _spans_playhead, _truncate
from pose_estimation.draw import (
    BONE_COLOR,
    BONE_COLOR_DIM,
    JOINT_COLOR,
    draw_skeleton,
    shade_box,
)
from recording.annotations import Annotation
from tests.fakes import fake_recording

NUM_LANDMARKS = 33


def _frame(h=120, w=160):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _line_landmarks(n=NUM_LANDMARKS):
    """Landmarks all at the frame center, so any drawn mark lands predictably."""
    pts = np.full((n, 3), 0.5, dtype=np.float64)
    return pts


class DrawSkeletonTests(unittest.TestCase):
    def test_all_nan_row_is_a_noop(self):
        """A frame with no pose must not raise and must not draw."""
        frame = _frame()
        pts = np.full((NUM_LANDMARKS, 3), np.nan)
        draw_skeleton(frame, pts, [(0, 1), (1, 2)])
        self.assertEqual(frame.sum(), 0)

    def test_partial_nan_skips_only_affected_bones(self):
        pts = _line_landmarks()
        pts[2] = np.nan  # landmark 2 is untracked
        frame = _frame()
        draw_skeleton(frame, pts, [(0, 1), (1, 2)])
        # (0,1) still drew something; (1,2) was skipped rather than crashing.
        self.assertGreater(frame.sum(), 0)

    def test_maps_normalized_coords_to_pixels(self):
        """x=0.5,y=0.5 must mark the frame center, using the frame's own size."""
        frame = _frame(h=120, w=160)
        pts = np.zeros((2, 3))
        pts[0] = (0.5, 0.5, 0.0)
        pts[1] = (0.5, 0.5, 0.0)
        draw_skeleton(frame, pts, [])
        self.assertTrue((frame[60, 80] == JOINT_COLOR).all())

    def test_visibility_dims_bones_and_joints(self):
        """Low-visibility landmarks draw dimmed -- the extrapolation the metrics gate on."""
        pts = np.zeros((2, 3))
        pts[0] = (0.25, 0.5, 0.0)
        pts[1] = (0.75, 0.5, 0.0)

        bright = _frame()
        draw_skeleton(bright, pts, [(0, 1)], visibility=np.array([1.0, 1.0]))
        dim = _frame()
        draw_skeleton(dim, pts, [(0, 1)], visibility=np.array([0.1, 0.1]))

        self.assertTrue((bright[60, 80] == BONE_COLOR).all())
        self.assertTrue((dim[60, 80] == BONE_COLOR_DIM).all())

    def test_bone_uses_the_weaker_endpoint(self):
        pts = np.zeros((2, 3))
        pts[0] = (0.25, 0.5, 0.0)
        pts[1] = (0.75, 0.5, 0.0)
        frame = _frame()
        draw_skeleton(frame, pts, [(0, 1)], visibility=np.array([1.0, 0.1]))
        self.assertTrue((frame[60, 80] == BONE_COLOR_DIM).all())

    def test_out_of_range_connection_indices_are_ignored(self):
        frame = _frame()
        draw_skeleton(frame, _line_landmarks(n=4), [(0, 99), (-1, 2)])
        self.assertIsNone(None)  # reaching here without raising is the assertion

    def test_missing_visibility_draws_everything_bright(self):
        pts = np.zeros((2, 3))
        pts[0] = (0.25, 0.5, 0.0)
        pts[1] = (0.75, 0.5, 0.0)
        frame = _frame()
        draw_skeleton(frame, pts, [(0, 1)])
        self.assertTrue((frame[60, 80] == BONE_COLOR).all())


class ShadeBoxTests(unittest.TestCase):
    """The translucent panel behind the overlay text (legibility over busy footage)."""

    def test_blends_toward_dark_without_erasing(self):
        frame = np.full((40, 40, 3), 200, np.uint8)
        shade_box(frame, (5, 5), (35, 35), alpha=0.5)
        inside = int(frame[20, 20, 0])
        self.assertLess(inside, 200)  # darkened
        self.assertGreater(inside, 0)  # translucent, not a solid fill
        self.assertEqual(int(frame[0, 0, 0]), 200)  # outside the box untouched

    def test_clamps_to_frame_and_never_raises(self):
        frame = np.full((20, 20, 3), 128, np.uint8)
        shade_box(frame, (-50, -50), (999, 999))  # fully off-bounds corners
        self.assertTrue((frame < 128).all())  # whole frame shaded, no crash

    def test_empty_or_inverted_box_is_a_noop(self):
        frame = np.full((20, 20, 3), 128, np.uint8)
        shade_box(frame, (10, 10), (10, 10))  # zero area
        shade_box(frame, (15, 15), (5, 5))  # inverted
        self.assertTrue((frame == 128).all())


class ActiveLabelTests(unittest.TestCase):
    def setUp(self):
        self.anns = [
            Annotation(index=0, label="sit_hold;arms=free", start_ms=0, end_ms=1000),
            Annotation(index=1, label="calib;pose=upright", start_ms=500, end_ms=1500),
            Annotation(index=2, label="crawl;style=belly", start_ms=3000, end_ms=4000),
        ]

    def test_inside_a_single_span(self):
        active = _active_labels(self.anns, 200)
        self.assertEqual([a.index for a in active], [0])

    def test_overlapping_spans_all_returned(self):
        active = _active_labels(self.anns, 700)
        self.assertEqual([a.index for a in active], [0, 1])

    def test_boundaries_are_inclusive(self):
        self.assertTrue(_spans_playhead(self.anns[0], 0))
        self.assertTrue(_spans_playhead(self.anns[0], 1000))
        self.assertFalse(_spans_playhead(self.anns[0], 1001))

    def test_gap_between_spans_is_empty(self):
        self.assertEqual(_active_labels(self.anns, 2000), [])

    def test_empty_annotation_list(self):
        self.assertEqual(_active_labels([], 100), [])

    def test_truncate_keeps_short_labels(self):
        self.assertEqual(_truncate("sit_hold", 30), "sit_hold")
        self.assertEqual(len(_truncate("x" * 50, 10)), 10)


class _Rec:
    """Fake Recording carrying the surface _render reads, including ``len()``.

    ``fake_recording`` covers the arrays; this adds ``frame()``, ``pose_connections``
    and ``__len__``, which only the GUI needs. A plain SimpleNamespace can't work here
    because ``len()`` resolves ``__len__`` on the type, not the instance.
    """

    def __init__(self, n=5, norm=None, visibility=1.0, annotations=None):
        rec = fake_recording(n=n, norm=norm, visibility=visibility,
                             annotations=annotations)
        self.__dict__.update(vars(rec))
        self._n = n
        self.pose_connections = np.array([[0, 1], [1, 2]], dtype=np.int32)

    def frame(self, i):
        return _frame()

    def __len__(self):
        return self._n


class RenderTests(unittest.TestCase):
    """_render must return a drawable frame for every state it can be put in."""

    def _render_one(self, rec, **kw):
        return _render(rec, 0, None, kw.pop("annotations", []), False, {}, **kw)

    def test_renders_with_pose_on_and_off(self):
        norm = np.full((3, NUM_LANDMARKS, 3), 0.5, dtype=np.float32)
        rec = _Rec(n=3, norm=norm)
        for show_pose in (True, False):
            out = self._render_one(rec, show_pose=show_pose,
                                   pose_present=np.ones(3, dtype=bool))
            self.assertEqual(out.shape, (120, 160, 3))

    def test_untracked_frame_renders(self):
        """An all-NaN pose frame must render (without a skeleton), not crash."""
        norm = np.full((3, NUM_LANDMARKS, 3), np.nan, dtype=np.float32)
        rec = _Rec(n=3, norm=norm)
        out = self._render_one(rec, show_pose=True,
                               pose_present=np.zeros(3, dtype=bool))
        self.assertEqual(out.shape, (120, 160, 3))

    def test_renders_with_annotations_and_legend(self):
        anns = [
            Annotation(index=0, label="sit_hold;arms=free;support=none",
                       start_ms=0, end_ms=1000),
            Annotation(index=1, label="calib;pose=upright", start_ms=0, end_ms=50),
        ]
        norm = np.full((3, NUM_LANDMARKS, 3), 0.5, dtype=np.float32)
        rec = _Rec(n=3, norm=norm)
        out = self._render_one(rec, annotations=anns, show_pose=True,
                               pose_present=np.ones(3, dtype=bool))
        self.assertEqual(out.shape, (120, 160, 3))

    def test_renders_with_no_annotations(self):
        norm = np.full((3, NUM_LANDMARKS, 3), 0.5, dtype=np.float32)
        rec = _Rec(n=3, norm=norm)
        out = self._render_one(rec, annotations=[], show_pose=True,
                               pose_present=np.ones(3, dtype=bool))
        self.assertEqual(out.shape, (120, 160, 3))


if __name__ == "__main__":
    unittest.main()
