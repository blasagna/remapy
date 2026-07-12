"""Tests for :mod:`face_blur.redact` — the shared redaction primitives."""

import unittest

import numpy as np

from face_blur import redact
from tests.fakes import solid_frame


class ValidateStyleTests(unittest.TestCase):
    def test_accepts_known_styles(self):
        for style in redact.BLUR_STYLES:
            redact.validate_style(style)  # must not raise

    def test_rejects_unknown_style(self):
        with self.assertRaises(ValueError):
            redact.validate_style("gaussian")


class PaddedBoundsTests(unittest.TestCase):
    def test_pads_by_fraction(self):
        frame = solid_frame(h=100, w=100)
        # box (40,40) size 20x20, pad 0.5 -> 10px each side.
        self.assertEqual(redact.padded_bounds(frame, 40, 40, 20, 20, 0.5), (30, 30, 70, 70))

    def test_clamps_to_frame(self):
        frame = solid_frame(h=100, w=100)
        bounds = redact.padded_bounds(frame, 0, 0, 20, 20, 1.0)
        x0, y0, x1, y1 = bounds
        self.assertEqual((x0, y0), (0, 0))  # cannot go negative
        self.assertLessEqual(x1, 100)
        self.assertLessEqual(y1, 100)

    def test_top_pad_overrides_top_only(self):
        frame = solid_frame(h=200, w=200)
        x0, y0, x1, y1 = redact.padded_bounds(frame, 80, 80, 20, 20, 0.5, top_pad=2.0)
        # side/bottom pad = 0.5*20 = 10; top pad = 2.0*20 = 40.
        self.assertEqual(x0, 70)
        self.assertEqual(y0, 80 - 40)
        self.assertEqual(x1, 110)
        self.assertEqual(y1, 110)

    def test_empty_region_returns_none(self):
        frame = solid_frame(h=100, w=100)
        # Zero-size box with zero pad collapses to an empty region.
        self.assertIsNone(redact.padded_bounds(frame, 50, 50, 0, 0, 0.0))


class RedactRegionTests(unittest.TestCase):
    def test_box_fills_solid_black(self):
        frame = solid_frame(h=40, w=40, value=200)
        redact.redact_region(frame, (10, 10, 30, 30), "box", mosaic_blocks=8)
        self.assertTrue(np.all(frame[10:30, 10:30] == 0))
        # Outside the box is untouched.
        self.assertTrue(np.all(frame[0:10, 0:10] == 200))

    def test_mosaic_changes_region_but_keeps_shape(self):
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 256, size=(40, 40, 3), dtype=np.uint8)
        before = frame.copy()
        redact.redact_region(frame, (5, 5, 35, 35), "mosaic", mosaic_blocks=4)
        self.assertEqual(frame.shape, before.shape)
        # The mosaic must actually alter the detailed region.
        self.assertFalse(np.array_equal(frame[5:35, 5:35], before[5:35, 5:35]))
        # And leave the rest alone.
        self.assertTrue(np.array_equal(frame[0:5, 0:5], before[0:5, 0:5]))

    def test_mosaic_handles_region_smaller_than_blocks(self):
        frame = solid_frame(h=20, w=20)
        # blocks (50) exceeds the 3x3 region; must clamp, not crash.
        redact.redact_region(frame, (0, 0, 3, 3), "mosaic", mosaic_blocks=50)


if __name__ == "__main__":
    unittest.main()
