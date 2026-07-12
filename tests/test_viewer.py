"""Tests for :mod:`rerun_viewer.viewer`.

The ``rerun`` SDK is mocked entirely (``rerun_viewer.viewer.rr``), so nothing is
logged to a real recording, spawned, or saved. We assert on the calls made.
"""

import unittest
from unittest import mock

import numpy as np

from tests.fakes import make_landmarks, pose_result, solid_frame


class InitTests(unittest.TestCase):
    def test_spawn_by_default_forwards_memory_limit(self):
        with mock.patch("rerun_viewer.viewer.rr") as rr:
            from rerun_viewer.viewer import PoseRerunLogger
            PoseRerunLogger(spawn=True, memory_limit="50%")
        rr.init.assert_called_once()
        rr.spawn.assert_called_once_with(memory_limit="50%")
        rr.save.assert_not_called()

    def test_save_path_uses_save_not_spawn(self):
        with mock.patch("rerun_viewer.viewer.rr") as rr:
            from rerun_viewer.viewer import PoseRerunLogger
            PoseRerunLogger(save_path="out.rrd", spawn=True)
        rr.save.assert_called_once_with("out.rrd")
        rr.spawn.assert_not_called()

    def test_no_spawn_no_save_is_headless(self):
        with mock.patch("rerun_viewer.viewer.rr") as rr:
            from rerun_viewer.viewer import PoseRerunLogger
            PoseRerunLogger(spawn=False)
        rr.spawn.assert_not_called()
        rr.save.assert_not_called()


class LogFrameTests(unittest.TestCase):
    def _logger(self, rr):
        from rerun_viewer.viewer import PoseRerunLogger
        return PoseRerunLogger(spawn=False)

    def test_logs_image_and_fps(self):
        with mock.patch("rerun_viewer.viewer.rr") as rr:
            logger = self._logger(rr)
            logger.log_frame(0, 0.0, 30.0, solid_frame(), pose_result())
        logged = [c.args[0] for c in rr.log.call_args_list]
        self.assertIn("video/image", logged)
        self.assertIn("metrics/fps", logged)

    def test_no_pose_clears_skeletons(self):
        with mock.patch("rerun_viewer.viewer.rr") as rr:
            logger = self._logger(rr)
            logger.log_frame(1, 0.1, 30.0, solid_frame(), pose_result())
        logged = [c.args[0] for c in rr.log.call_args_list]
        self.assertIn("video/image/skeleton", logged)
        self.assertIn("pose3d", logged)
        # No skeleton keypoints / angles are logged when there's no pose.
        self.assertNotIn("video/image/keypoints", logged)

    def test_pose_present_logs_skeleton_and_angles(self):
        with mock.patch("rerun_viewer.viewer.rr") as rr:
            logger = self._logger(rr)
            lms = make_landmarks()
            result = pose_result(poses_norm=[lms], poses_world=[lms])
            logger.log_frame(2, 0.2, 29.0, solid_frame(), result)
        logged = [c.args[0] for c in rr.log.call_args_list]
        self.assertIn("video/image/skeleton", logged)
        self.assertIn("video/image/keypoints", logged)
        self.assertTrue(any(name.startswith("metrics/angles/") for name in logged))

    def test_sets_both_timelines(self):
        with mock.patch("rerun_viewer.viewer.rr") as rr:
            logger = self._logger(rr)
            logger.log_frame(5, 1.5, 30.0, solid_frame(), pose_result())
        self.assertEqual(rr.set_time.call_count, 2)


if __name__ == "__main__":
    unittest.main()
