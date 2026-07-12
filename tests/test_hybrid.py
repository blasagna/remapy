"""Tests for :mod:`face_blur.hybrid` — per-frame backend routing.

Both composed backends are replaced with mocks so the test verifies *which* path
each frame takes, independent of the real redaction implementations.
"""

import unittest
from unittest import mock

from tests.fakes import make_landmarks, pose_result, solid_frame


class HybridRoutingTests(unittest.TestCase):
    def _build(self):
        """Return a HybridFaceBlurrer with both backends mocked out."""
        with mock.patch("face_blur.hybrid.PoseFaceBlurrer") as Pose, \
                mock.patch("face_blur.hybrid.FaceBlurrer") as Det:
            from face_blur.hybrid import HybridFaceBlurrer
            hybrid = HybridFaceBlurrer(style="box")
        return hybrid, hybrid._pose, hybrid._detector

    def test_pose_present_uses_pose_backend(self):
        hybrid, pose, det = self._build()
        frame = solid_frame()
        result = pose_result(poses_norm=[make_landmarks()])
        hybrid.blur(frame, result)
        pose.blur.assert_called_once_with(frame, result)
        det.blur.assert_not_called()

    def test_no_pose_uses_detector_backend(self):
        hybrid, pose, det = self._build()
        frame = solid_frame()
        hybrid.blur(frame, pose_result())  # empty pose_landmarks
        det.blur.assert_called_once_with(frame)
        pose.blur.assert_not_called()

    def test_none_result_uses_detector_backend(self):
        hybrid, pose, det = self._build()
        frame = solid_frame()
        hybrid.blur(frame, None)
        det.blur.assert_called_once_with(frame)
        pose.blur.assert_not_called()

    def test_blur_returns_frame(self):
        hybrid, _pose, _det = self._build()
        frame = solid_frame()
        self.assertIs(hybrid.blur(frame, None), frame)

    def test_close_closes_both_backends(self):
        hybrid, pose, det = self._build()
        hybrid.close()
        pose.close.assert_called_once()
        det.close.assert_called_once()

    def test_context_manager_closes_both(self):
        with mock.patch("face_blur.hybrid.PoseFaceBlurrer") as Pose, \
                mock.patch("face_blur.hybrid.FaceBlurrer") as Det:
            from face_blur.hybrid import HybridFaceBlurrer
            with HybridFaceBlurrer() as hybrid:
                pass
        hybrid._pose.close.assert_called_once()
        hybrid._detector.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
