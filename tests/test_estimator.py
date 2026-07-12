"""Tests for :mod:`pose_estimation.estimator`.

The model download and the native ``PoseLandmarker`` are mocked; ``detect`` is
exercised with a real (small) BGR frame so the BGR->RGB / ``mp.Image`` path runs.
"""

import unittest
from contextlib import contextmanager
from unittest import mock

from pose_estimation.estimator import POSE_CONNECTIONS
from tests.fakes import pose_result, solid_frame


@contextmanager
def patched_landmarker(result=None):
    """Patch the estimator's model + landmarker; yield the fake landmarker."""
    fake = mock.Mock()
    fake.timestamps = []

    def detect_for_video(_image, ts):
        fake.timestamps.append(ts)
        return result if result is not None else pose_result()

    fake.detect_for_video.side_effect = detect_for_video

    with mock.patch("pose_estimation.estimator.ensure_model", return_value="/fake/model.task"), \
            mock.patch("pose_estimation.estimator.PoseLandmarker") as PL:
        PL.create_from_options.return_value = fake
        yield fake


class PoseConnectionsTests(unittest.TestCase):
    def test_connections_are_index_pairs(self):
        self.assertTrue(len(POSE_CONNECTIONS) > 0)
        for pair in POSE_CONNECTIONS:
            self.assertEqual(len(pair), 2)
            self.assertTrue(all(isinstance(i, int) for i in pair))


class DetectTests(unittest.TestCase):
    def test_detect_forwards_timestamp_and_returns_result(self):
        sentinel = pose_result()
        with patched_landmarker(sentinel) as fake:
            from pose_estimation.estimator import PoseEstimator
            with PoseEstimator() as pose:
                out = pose.detect(solid_frame(), 42)
        self.assertIs(out, sentinel)
        self.assertEqual(fake.timestamps, [42])

    def test_detect_after_close_raises(self):
        with patched_landmarker():
            from pose_estimation.estimator import PoseEstimator
            pose = PoseEstimator()
            pose.close()
            with self.assertRaises(RuntimeError):
                pose.detect(solid_frame(), 0)

    def test_close_releases_landmarker(self):
        with patched_landmarker() as fake:
            from pose_estimation.estimator import PoseEstimator
            pose = PoseEstimator()
            pose.close()
        fake.close.assert_called_once()

    def test_options_forward_confidence_and_num_poses(self):
        with patched_landmarker(), \
                mock.patch("pose_estimation.estimator.PoseLandmarkerOptions") as Opts:
            from pose_estimation.estimator import PoseEstimator
            PoseEstimator(num_poses=2, min_detection_confidence=0.7)
            kwargs = Opts.call_args.kwargs
            self.assertEqual(kwargs["num_poses"], 2)
            self.assertEqual(kwargs["min_pose_detection_confidence"], 0.7)


if __name__ == "__main__":
    unittest.main()
