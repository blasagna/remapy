"""Tests for :mod:`face_blur.pose_blur` — pose-keypoint face redaction."""

import unittest

import numpy as np

from face_blur.pose_blur import FACE_LANDMARKS, PoseFaceBlurrer
from tests.fakes import (
    FakeLandmark,
    face_landmarks_in_box,
    make_landmarks,
    pose_result,
    solid_frame,
)


class FaceLandmarksTests(unittest.TestCase):
    def test_face_landmarks_are_first_eleven(self):
        self.assertEqual(FACE_LANDMARKS, tuple(range(0, 11)))


class BlurTests(unittest.TestCase):
    def setUp(self):
        self.frame = solid_frame(h=200, w=200, value=200)

    def test_no_pose_is_noop(self):
        before = self.frame.copy()
        with PoseFaceBlurrer() as blurrer:
            blurrer.blur(self.frame, pose_result())
        self.assertTrue(np.array_equal(self.frame, before))

    def test_none_result_is_noop(self):
        before = self.frame.copy()
        with PoseFaceBlurrer() as blurrer:
            blurrer.blur(self.frame, None)
        self.assertTrue(np.array_equal(self.frame, before))

    def test_redacts_head_region_in_place(self):
        lms = face_landmarks_in_box(80, 80, 120, 110, w=200, h=200)
        with PoseFaceBlurrer(pad=0.0, top_pad=0.0) as blurrer:
            blurrer.blur(self.frame, pose_result(poses_norm=[lms]))
        # The keypoint bounding box (80..120, 80..110) must have been blacked out.
        self.assertTrue(np.all(self.frame[85:105, 85:115] == 0))
        # A far corner is untouched.
        self.assertTrue(np.all(self.frame[0:10, 0:10] == 200))

    def test_top_pad_extends_above_box(self):
        lms = face_landmarks_in_box(80, 100, 120, 130, w=200, h=200)
        with PoseFaceBlurrer(pad=0.0, top_pad=1.0) as blurrer:
            blurrer.blur(self.frame, pose_result(poses_norm=[lms]))
        # box height ~30, top_pad 1.0 -> ~30px of headroom above y=100 is redacted.
        self.assertTrue(np.all(self.frame[75:80, 90:110] == 0))

    def test_low_visibility_falls_back_to_all_keypoints(self):
        # All face keypoints below min_visibility -> fallback still redacts.
        lms = face_landmarks_in_box(80, 80, 120, 110, w=200, h=200, visibility=0.1)
        with PoseFaceBlurrer(pad=0.0, top_pad=0.0, min_visibility=0.5) as blurrer:
            blurrer.blur(self.frame, pose_result(poses_norm=[lms]))
        self.assertTrue(np.all(self.frame[85:105, 85:115] == 0))

    def test_too_few_landmarks_returns_none_bounds(self):
        # Only one usable landmark -> cannot form a box; frame untouched.
        before = self.frame.copy()
        lms = [FakeLandmark(x=0.5, y=0.5)]
        with PoseFaceBlurrer() as blurrer:
            blurrer.blur(self.frame, pose_result(poses_norm=[lms]))
        self.assertTrue(np.array_equal(self.frame, before))

    def test_multiple_poses_all_redacted(self):
        a = face_landmarks_in_box(10, 10, 40, 40, w=200, h=200)
        b = face_landmarks_in_box(150, 150, 180, 180, w=200, h=200)
        with PoseFaceBlurrer(pad=0.0, top_pad=0.0) as blurrer:
            blurrer.blur(self.frame, pose_result(poses_norm=[a, b]))
        self.assertTrue(np.all(self.frame[15:35, 15:35] == 0))
        self.assertTrue(np.all(self.frame[155:175, 155:175] == 0))

    def test_blur_after_close_raises(self):
        blurrer = PoseFaceBlurrer()
        blurrer.close()
        with self.assertRaises(RuntimeError):
            blurrer.blur(self.frame, pose_result(poses_norm=[make_landmarks()]))

    def test_invalid_style_rejected(self):
        with self.assertRaises(ValueError):
            PoseFaceBlurrer(style="swirl")


if __name__ == "__main__":
    unittest.main()
