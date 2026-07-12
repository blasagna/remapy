"""Tests for :mod:`video_capture.capture`.

The underlying ``cv2.VideoCapture`` is replaced with :class:`tests.fakes.FakeCapture`
so no real camera or file is touched.
"""

import unittest
from unittest import mock

import cv2

from tests.fakes import FakeCapture, solid_frame
from video_capture.capture import CaptureError, VideoCapture


class OpenTests(unittest.TestCase):
    def test_open_returns_self_and_applies_size(self):
        fake = FakeCapture(frames=[solid_frame()])
        with mock.patch("cv2.VideoCapture", return_value=fake) as ctor:
            cap = VideoCapture(source=0, width=1280, height=720)
            self.assertIs(cap.open(), cap)
            ctor.assert_called_once_with(0)
        # Requested size forwarded to the device.
        self.assertIn((cv2.CAP_PROP_FRAME_WIDTH, 1280), fake.set_calls)
        self.assertIn((cv2.CAP_PROP_FRAME_HEIGHT, 720), fake.set_calls)

    def test_open_without_size_does_not_set_props(self):
        fake = FakeCapture()
        with mock.patch("cv2.VideoCapture", return_value=fake):
            VideoCapture(source="movie.mp4").open()
        self.assertEqual(fake.set_calls, [])

    def test_open_failure_raises_and_releases(self):
        fake = FakeCapture(opened=False)
        with mock.patch("cv2.VideoCapture", return_value=fake):
            with self.assertRaises(CaptureError):
                VideoCapture(source=3).open()
        self.assertTrue(fake.released)


class ReadTests(unittest.TestCase):
    def test_read_before_open_raises(self):
        with self.assertRaises(CaptureError):
            VideoCapture().read()

    def test_read_returns_frame(self):
        frame = solid_frame()
        fake = FakeCapture(frames=[frame])
        with mock.patch("cv2.VideoCapture", return_value=fake):
            cap = VideoCapture().open()
            self.assertIs(cap.read(), frame)

    def test_read_failure_raises(self):
        fake = FakeCapture(frames=[])  # read() -> (False, None)
        with mock.patch("cv2.VideoCapture", return_value=fake):
            cap = VideoCapture().open()
            with self.assertRaises(CaptureError):
                cap.read()

    def test_read_none_frame_raises(self):
        fake = FakeCapture(frames=[None])  # opened but delivers a None frame
        with mock.patch("cv2.VideoCapture", return_value=fake):
            cap = VideoCapture().open()
            with self.assertRaises(CaptureError):
                cap.read()


class FramesTests(unittest.TestCase):
    def test_frames_yields_until_exhausted(self):
        frames = [solid_frame(value=v) for v in (10, 20, 30)]
        fake = FakeCapture(frames=list(frames))
        with mock.patch("cv2.VideoCapture", return_value=fake):
            cap = VideoCapture().open()
            got = list(cap.frames())
        self.assertEqual(len(got), 3)

    def test_frames_stops_at_bad_read(self):
        fake = FakeCapture(frames=[solid_frame(), None, solid_frame()])
        with mock.patch("cv2.VideoCapture", return_value=fake):
            cap = VideoCapture().open()
            self.assertEqual(len(list(cap.frames())), 1)

    def test_frames_before_open_raises(self):
        with self.assertRaises(CaptureError):
            list(VideoCapture().frames())


class ResolutionAndLifecycleTests(unittest.TestCase):
    def test_resolution_reads_device_props(self):
        fake = FakeCapture(props={
            cv2.CAP_PROP_FRAME_WIDTH: 640.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
        })
        with mock.patch("cv2.VideoCapture", return_value=fake):
            cap = VideoCapture().open()
            self.assertEqual(cap.resolution, (640, 480))

    def test_resolution_before_open_raises(self):
        with self.assertRaises(CaptureError):
            _ = VideoCapture().resolution

    def test_release_is_idempotent(self):
        fake = FakeCapture()
        with mock.patch("cv2.VideoCapture", return_value=fake):
            cap = VideoCapture().open()
            cap.release()
            cap.release()  # second call must not raise
        self.assertTrue(fake.released)

    def test_context_manager_opens_and_releases(self):
        fake = FakeCapture(frames=[solid_frame()])
        with mock.patch("cv2.VideoCapture", return_value=fake):
            with VideoCapture() as cap:
                self.assertIsNotNone(cap.read())
        self.assertTrue(fake.released)


if __name__ == "__main__":
    unittest.main()
