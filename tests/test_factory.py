"""Tests for :mod:`face_blur.factory` — ``--blur-method`` backend selection."""

import unittest
from unittest import mock

from face_blur import factory
from face_blur.factory import BLUR_METHODS, build_blurrer
from face_blur.hybrid import HybridFaceBlurrer
from face_blur.pose_blur import PoseFaceBlurrer


class BuildBlurrerTests(unittest.TestCase):
    def test_methods_constant(self):
        self.assertEqual(BLUR_METHODS, ("detector", "pose", "hybrid"))

    def test_pose_method_needs_no_model(self):
        blurrer = build_blurrer("pose", style="mosaic")
        self.assertIsInstance(blurrer, PoseFaceBlurrer)
        self.assertEqual(blurrer.style, "mosaic")

    def test_detector_method_constructs_face_blurrer(self):
        # FaceBlurrer construction is mocked to avoid the model download.
        with mock.patch("face_blur.blur.ensure_model", return_value="/fake.tflite"), \
                mock.patch("face_blur.blur.FaceDetector"):
            blurrer = build_blurrer("detector", style="box", model_path="/x")
        from face_blur.blur import FaceBlurrer
        self.assertIsInstance(blurrer, FaceBlurrer)

    def test_hybrid_method_constructs_hybrid(self):
        with mock.patch("face_blur.blur.ensure_model", return_value="/fake.tflite"), \
                mock.patch("face_blur.blur.FaceDetector"):
            blurrer = build_blurrer("hybrid", style="box")
        self.assertIsInstance(blurrer, HybridFaceBlurrer)

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            build_blurrer("blurry")

    def test_all_advertised_methods_build(self):
        with mock.patch("face_blur.blur.ensure_model", return_value="/fake.tflite"), \
                mock.patch("face_blur.blur.FaceDetector"):
            for method in factory.BLUR_METHODS:
                self.assertIsNotNone(build_blurrer(method))


if __name__ == "__main__":
    unittest.main()
