"""Tests for :mod:`list_devices.devices`.

``cv2.VideoCapture`` and the Linux sysfs helpers are mocked so probing does not
touch real hardware.
"""

import unittest
from unittest import mock

import cv2

from list_devices import devices
from list_devices.devices import DeviceInfo, enumerate_devices, probe_index
from tests.fakes import FakeCapture, solid_frame


def _readable_cap(default=(640, 480), maxres=(1920, 1080), fps=30.0):
    """A FakeCapture that opens, reads a frame, and reports resolutions.

    ``set`` on the width/height properties bumps the reported value up to the
    device's max (mimicking the driver clamping an oversized request).
    """
    props = {
        cv2.CAP_PROP_FRAME_WIDTH: float(default[0]),
        cv2.CAP_PROP_FRAME_HEIGHT: float(default[1]),
        cv2.CAP_PROP_FPS: fps,
    }
    cap = FakeCapture(frames=[solid_frame()], props=props)

    real_set = cap.set

    def clamped_set(prop, value):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            value = min(value, maxres[0])
        elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
            value = min(value, maxres[1])
        return real_set(prop, value)

    cap.set = clamped_set
    return cap


class ProbeIndexTests(unittest.TestCase):
    def test_unopenable_index_returns_none(self):
        with mock.patch("cv2.VideoCapture", return_value=FakeCapture(opened=False)):
            self.assertIsNone(probe_index(0))

    def test_open_but_no_frame_returns_none(self):
        # Opens, but read() yields nothing -> phantom node, filtered out.
        with mock.patch("cv2.VideoCapture", return_value=FakeCapture(frames=[])):
            self.assertIsNone(probe_index(1))

    def test_readable_index_returns_deviceinfo(self):
        cap = _readable_cap(default=(640, 480), maxres=(1920, 1080), fps=25.0)
        with mock.patch("cv2.VideoCapture", return_value=cap):
            info = probe_index(0)
        self.assertIsInstance(info, DeviceInfo)
        self.assertEqual(info.index, 0)
        self.assertEqual((info.default_width, info.default_height), (640, 480))
        self.assertEqual((info.max_width, info.max_height), (1920, 1080))
        self.assertEqual(info.fps, 25.0)
        self.assertEqual(info.backend, "FAKE")

    def test_max_res_never_below_default(self):
        # Driver ignores the oversized request (returns default); max >= default.
        cap = _readable_cap(default=(1280, 720), maxres=(1280, 720))
        with mock.patch("cv2.VideoCapture", return_value=cap):
            info = probe_index(0)
        self.assertEqual((info.max_width, info.max_height), (1280, 720))

    def test_capture_is_released(self):
        cap = _readable_cap()
        with mock.patch("cv2.VideoCapture", return_value=cap):
            probe_index(0)
        self.assertTrue(cap.released)


class DeviceInfoTests(unittest.TestCase):
    def test_source_arg_is_string_index(self):
        info = DeviceInfo(2, 640, 480, 1920, 1080, 30.0, "V4L2")
        self.assertEqual(info.source_arg, "2")


class EnumerateDevicesTests(unittest.TestCase):
    def test_collects_only_readable_indices(self):
        found = {0: DeviceInfo(0, 640, 480, 640, 480, 30.0, "FAKE")}

        def fake_probe(i):
            info = found.get(i)
            # Return a fresh copy so mutation (name/node) does not leak between tests.
            return DeviceInfo(**vars(info)) if info else None

        with mock.patch.object(devices, "probe_index", side_effect=fake_probe), \
                mock.patch.object(devices, "_v4l2_names", return_value={}), \
                mock.patch.object(devices, "_highest_v4l2_index", return_value=None), \
                mock.patch("os.path.exists", return_value=False):
            result = enumerate_devices(max_index=3)
        self.assertEqual([d.index for d in result], [0])

    def test_attaches_v4l2_name_and_node(self):
        def fake_probe(i):
            if i == 0:
                return DeviceInfo(0, 640, 480, 640, 480, 30.0, "V4L2")
            return None

        with mock.patch.object(devices, "probe_index", side_effect=fake_probe), \
                mock.patch.object(devices, "_v4l2_names", return_value={0: "HD Webcam"}), \
                mock.patch.object(devices, "_highest_v4l2_index", return_value=None), \
                mock.patch("os.path.exists", return_value=True):
            result = enumerate_devices(max_index=0)
        self.assertEqual(result[0].name, "HD Webcam")
        self.assertEqual(result[0].node, "/dev/video0")

    def test_scan_extended_to_highest_v4l2_index(self):
        probed = []

        def fake_probe(i):
            probed.append(i)
            return None

        with mock.patch.object(devices, "probe_index", side_effect=fake_probe), \
                mock.patch.object(devices, "_v4l2_names", return_value={}), \
                mock.patch.object(devices, "_highest_v4l2_index", return_value=12):
            enumerate_devices(max_index=3)
        # Scan widened from 3 up to the highest present node (12).
        self.assertEqual(max(probed), 12)

    def test_restores_opencv_log_level(self):
        with mock.patch.object(devices, "probe_index", return_value=None), \
                mock.patch.object(devices, "_v4l2_names", return_value={}), \
                mock.patch.object(devices, "_highest_v4l2_index", return_value=None), \
                mock.patch.object(cv2.utils.logging, "getLogLevel", return_value=99), \
                mock.patch.object(cv2.utils.logging, "setLogLevel") as set_level:
            enumerate_devices(max_index=1)
        # Last call restores the original level captured before the scan.
        self.assertEqual(set_level.call_args_list[-1].args[0], 99)


if __name__ == "__main__":
    unittest.main()
