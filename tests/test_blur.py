"""Tests for :mod:`face_blur.blur` — the ``FaceDetector`` redaction backend.

The model download (:func:`face_blur.model.ensure_model`) and the native
``FaceDetector`` are mocked, so no file is downloaded and no model is loaded.
"""

import unittest
from contextlib import contextmanager
from unittest import mock

import numpy as np

from tests.fakes import detection, detector_result, solid_frame


@contextmanager
def patched_detector(result):
    """Patch ``FaceBlurrer``'s model + detector; yield the fake detector.

    ``result`` is returned from every ``detect_for_video`` call. The fake records
    the timestamps it was called with on ``.timestamps``.
    """
    fake = mock.Mock()
    fake.timestamps = []

    def detect_for_video(_image, ts):
        fake.timestamps.append(ts)
        return result

    fake.detect_for_video.side_effect = detect_for_video

    with mock.patch("face_blur.blur.ensure_model", return_value="/fake/model.tflite"), \
            mock.patch("face_blur.blur.FaceDetector") as FD:
        FD.create_from_options.return_value = fake
        yield fake


class ConstructionTests(unittest.TestCase):
    def test_invalid_style_rejected_before_model_load(self):
        from face_blur.blur import FaceBlurrer
        with self.assertRaises(ValueError):
            FaceBlurrer(style="swirl")


class BlurTests(unittest.TestCase):
    def _import(self):
        from face_blur.blur import FaceBlurrer
        return FaceBlurrer

    def test_redacts_detected_face(self):
        FaceBlurrer = self._import()
        frame = solid_frame(h=100, w=100, value=200)
        result = detector_result(detection(40, 40, 20, 20))
        with patched_detector(result):
            with FaceBlurrer(style="box", pad=0.0) as blurrer:
                blurrer.blur(frame)
        self.assertTrue(np.all(frame[40:60, 40:60] == 0))
        self.assertTrue(np.all(frame[0:10, 0:10] == 200))

    def test_no_detections_leaves_frame_untouched(self):
        FaceBlurrer = self._import()
        frame = solid_frame(value=200)
        before = frame.copy()
        with patched_detector(detector_result()):
            with FaceBlurrer() as blurrer:
                blurrer.blur(frame)
        self.assertTrue(np.array_equal(frame, before))

    def test_timestamps_strictly_increase(self):
        FaceBlurrer = self._import()
        frame = solid_frame()
        with patched_detector(detector_result()) as fake:
            with FaceBlurrer() as blurrer:
                for _ in range(5):
                    blurrer.blur(frame)
        ts = fake.timestamps
        self.assertEqual(len(ts), 5)
        self.assertTrue(all(b > a for a, b in zip(ts, ts[1:])), ts)

    def test_explicit_timestamp_is_forwarded(self):
        FaceBlurrer = self._import()
        frame = solid_frame()
        with patched_detector(detector_result()) as fake:
            with FaceBlurrer() as blurrer:
                blurrer.blur(frame, timestamp_ms=1234)
        self.assertEqual(fake.timestamps, [1234])

    def test_pose_result_argument_is_ignored(self):
        FaceBlurrer = self._import()
        frame = solid_frame(value=200)
        result = detector_result(detection(10, 10, 10, 10))
        with patched_detector(result):
            with FaceBlurrer(pad=0.0) as blurrer:
                # Passing a bogus pose_result must not change behavior.
                blurrer.blur(frame, pose_result=object())
        self.assertTrue(np.all(frame[10:20, 10:20] == 0))

    def test_out_of_frame_detection_is_skipped(self):
        FaceBlurrer = self._import()
        frame = solid_frame(h=50, w=50, value=200)
        before = frame.copy()
        # A zero-size box yields empty bounds -> padded_bounds returns None.
        with patched_detector(detector_result(detection(25, 25, 0, 0))):
            with FaceBlurrer(pad=0.0) as blurrer:
                blurrer.blur(frame)
        self.assertTrue(np.array_equal(frame, before))

    def test_blur_after_close_raises(self):
        FaceBlurrer = self._import()
        with patched_detector(detector_result()):
            blurrer = FaceBlurrer()
            blurrer.close()
            self.assertTrue(True)  # close ran
            with self.assertRaises(RuntimeError):
                blurrer.blur(solid_frame())

    def test_close_releases_detector(self):
        FaceBlurrer = self._import()
        with patched_detector(detector_result()) as fake:
            blurrer = FaceBlurrer()
            blurrer.close()
        fake.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
