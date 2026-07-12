"""Tests for :mod:`pose_estimation.angles` — joint-angle math."""

import math
import unittest

import numpy as np

from pose_estimation import angles
from pose_estimation.angles import JOINT_TRIPLETS, angle_between, joint_angles
from tests.fakes import FakeLandmark


class AngleBetweenTests(unittest.TestCase):
    def test_right_angle(self):
        a = np.array([1.0, 0.0, 0.0])
        joint = np.array([0.0, 0.0, 0.0])
        c = np.array([0.0, 1.0, 0.0])
        self.assertAlmostEqual(angle_between(a, joint, c), 90.0, places=6)

    def test_straight_line_is_180(self):
        a = np.array([-1.0, 0.0, 0.0])
        joint = np.array([0.0, 0.0, 0.0])
        c = np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(angle_between(a, joint, c), 180.0, places=6)

    def test_coincident_direction_is_zero(self):
        a = np.array([2.0, 0.0, 0.0])
        joint = np.array([0.0, 0.0, 0.0])
        c = np.array([5.0, 0.0, 0.0])
        self.assertAlmostEqual(angle_between(a, joint, c), 0.0, places=6)

    def test_zero_length_segment_is_nan(self):
        joint = np.array([1.0, 1.0, 1.0])
        # ``a`` coincides with the joint -> zero-length vector -> nan.
        self.assertTrue(math.isnan(angle_between(joint.copy(), joint, np.array([2.0, 2.0, 2.0]))))

    def test_clips_numerical_overshoot(self):
        # Nearly-collinear vectors can push cos slightly past 1.0; must not error.
        a = np.array([1e-8, 0.0, 0.0])
        joint = np.array([0.0, 0.0, 0.0])
        c = np.array([1.0, 0.0, 0.0])
        val = angle_between(a, joint, c)
        self.assertFalse(math.isnan(val))
        self.assertAlmostEqual(val, 0.0, places=4)


class JointAnglesTests(unittest.TestCase):
    def test_returns_all_named_joints(self):
        lms = [FakeLandmark(x=float(i), y=0.0, z=0.0) for i in range(33)]
        result = joint_angles(lms)
        self.assertEqual(set(result), set(JOINT_TRIPLETS))

    def test_known_geometry_gives_right_angle(self):
        # Build a pose where the left elbow subtends exactly 90 degrees.
        lms = [FakeLandmark() for _ in range(33)]
        i_shoulder, i_elbow, i_wrist = JOINT_TRIPLETS["left_elbow"]
        lms[i_elbow] = FakeLandmark(x=0.0, y=0.0, z=0.0)
        lms[i_shoulder] = FakeLandmark(x=0.0, y=1.0, z=0.0)
        lms[i_wrist] = FakeLandmark(x=1.0, y=0.0, z=0.0)
        self.assertAlmostEqual(joint_angles(lms)["left_elbow"], 90.0, places=5)

    def test_all_values_are_floats(self):
        lms = [FakeLandmark(x=float(i) * 0.1, y=float(i) * 0.2, z=0.0) for i in range(33)]
        for name, value in joint_angles(lms).items():
            self.assertIsInstance(value, float, name)


if __name__ == "__main__":
    unittest.main()
