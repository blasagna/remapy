"""Tests for the Feather Sense host integration.

Covers the serial decode/poll layer (``adafruit_feather_sense.stream``), the
recorder's ``/feather`` datasets + reader, and the rerun viewer's sensor logging.
No board, serial port, camera, or rerun viewer is touched: serial is a
``FakeSerial`` fed pre-baked protocol frames, HDF5 is a real temp file, and the
``rerun`` SDK is mocked.
"""

import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest import mock

import numpy as np

# adafruit_feather_sense.stream inserts the dir holding feather_protocol on sys.path.
from adafruit_feather_sense.stream import FeatherSenseStream, SensorRecord
import feather_protocol as fp
from recording.reader import Recording
from recording.recorder import HDF5Recorder
from tests.fakes import FakeSerial, pose_result, solid_frame


def _baked_stream():
    """A byte blob of one frame per stream, including an error frame."""
    blob = b""
    blob += fp.encode(fp.MSG_ACCEL, 100, fp.to_raw(fp.MSG_ACCEL, (0.0, 0.1, 9.81)))
    blob += fp.encode(fp.MSG_GRAVITY, 100, fp.to_raw(fp.MSG_GRAVITY, (0.0, 0.1, 9.80)))
    blob += fp.encode(fp.MSG_LINEAR_ACCEL, 100, fp.to_raw(fp.MSG_LINEAR_ACCEL, (0.0, 0.0, 0.01)))
    blob += fp.encode(fp.MSG_ENV, 200, fp.to_raw(fp.MSG_ENV, (25.0, 40.0, 1010.0)))
    blob += fp.encode(fp.MSG_ALTITUDE, 200, fp.to_raw(fp.MSG_ALTITUDE, (75.5,)))
    blob += fp.encode(fp.MSG_BATTERY, 300, fp.to_raw(fp.MSG_BATTERY, (4.05, 82.0)), extra_u8=1)
    blob += fp.encode_error(400, fp.MSG_MAG, "boom")
    return blob


def _drain(stream, fake):
    records = []
    for _ in range(500):
        recs = stream.poll()
        records.extend(recs)
        if fake.in_waiting == 0 and not recs:
            break
    return records


class StreamDecodeTests(unittest.TestCase):
    def test_poll_decodes_si_units_and_names(self):
        fake = FakeSerial(_baked_stream(), chunk=6)
        stream = FeatherSenseStream(serial_obj=fake)
        recs = _drain(stream, fake)

        names = [r.name for r in recs]
        self.assertEqual(
            names,
            ["accel", "gravity", "linear_accel", "env", "altitude", "battery", "error"],
        )
        by = {r.name: r for r in recs}
        # int32 wire values converted back to SI floats.
        self.assertAlmostEqual(by["accel"].values[2], 9.81, places=2)
        self.assertAlmostEqual(by["env"].values[2], 1010.0, places=2)
        self.assertAlmostEqual(by["altitude"].values[0], 75.5, places=2)
        # battery usb flag is a passthrough int, not scaled.
        self.assertEqual(by["battery"].values[2], 1)
        self.assertEqual(stream.errors, 0)

    def test_error_record_resolves_source_name(self):
        fake = FakeSerial(_baked_stream(), chunk=64)
        stream = FeatherSenseStream(serial_obj=fake)
        err = [r for r in _drain(stream, fake) if r.name == "error"][0]
        self.assertEqual(err.values[0], "mag")  # source name, not the raw id
        self.assertEqual(err.values[1], "boom")

    def test_cross_read_buffering(self):
        """Byte-at-a-time reads must reassemble the same records."""
        blob = _baked_stream()
        one = _drain(FeatherSenseStream(serial_obj=FakeSerial(blob, chunk=64)),
                     FakeSerial(blob, chunk=64))  # reference count
        fake = FakeSerial(blob, chunk=1)
        recs = _drain(FeatherSenseStream(serial_obj=fake), fake)
        self.assertEqual([r.name for r in recs], [r.name for r in one])

    def test_resyncs_after_banner(self):
        # A CircuitPython banner (text, no 0x00) glues onto the first frame, so
        # that frame is sacrificed; the stream resyncs at the next delimiter and
        # the following frame decodes cleanly (errors counted, not raised).
        blob = b"Auto-reload is on. code.py output:\r\n"
        blob += fp.encode(fp.MSG_GYRO, 1, fp.to_raw(fp.MSG_GYRO, (0.0, 0.0, 0.0)))
        blob += fp.encode(fp.MSG_ACCEL, 2, fp.to_raw(fp.MSG_ACCEL, (0.0, 0.0, 9.8)))
        fake = FakeSerial(blob, chunk=9)
        stream = FeatherSenseStream(serial_obj=fake)
        recs = _drain(stream, fake)
        self.assertEqual([r.name for r in recs], ["accel"])
        self.assertGreaterEqual(stream.errors, 1)


class OpenIfAvailableTests(unittest.TestCase):
    def test_returns_stream_when_frames_present(self):
        fake = FakeSerial(_baked_stream(), chunk=16)
        with mock.patch("adafruit_feather_sense.stream.serial.Serial", return_value=fake), \
                mock.patch("adafruit_feather_sense.stream.autodetect_port", return_value="/dev/ttyACM9"):
            stream = FeatherSenseStream.open_if_available(probe_timeout=0.3)
        self.assertIsNotNone(stream)
        # Records seen during probing are not lost — they come back on first poll.
        self.assertTrue(any(r.name == "accel" for r in stream.poll()))

    def test_returns_none_when_silent(self):
        fake = FakeSerial(b"", chunk=8)
        with mock.patch("adafruit_feather_sense.stream.serial.Serial", return_value=fake), \
                mock.patch("adafruit_feather_sense.stream.autodetect_port", return_value="/dev/ttyACM9"):
            stream = FeatherSenseStream.open_if_available(probe_timeout=0.2)
        self.assertIsNone(stream)
        self.assertTrue(fake.closed)

    def test_returns_none_when_no_port(self):
        with mock.patch("adafruit_feather_sense.stream.autodetect_port", return_value=None):
            self.assertIsNone(FeatherSenseStream.open_if_available(probe_timeout=0.1))


class RecorderFeatherTests(unittest.TestCase):
    def _record(self, path):
        no_pose = pose_result()  # no landmarks
        with HDF5Recorder(path) as rec:
            rec.append(solid_frame(), 0, no_pose)
            rec.append_sensor("accel", 100, [0.0, 0.1, 9.8], ("x", "y", "z"))
            rec.append_sensor("accel", 130, [0.0, 0.2, 9.7], ("x", "y", "z"))
            rec.append_sensor("altitude", 110, [75.5], ("altitude_m",))
            rec.append_sensor("error", 400, ["mag", "boom"], ("source", "message"))
            rec.append(solid_frame(), 33, no_pose)

    def test_roundtrip(self):
        with TemporaryDirectory() as d:
            path = Path(d) / "s.h5"
            self._record(path)
            with Recording(path) as r:
                self.assertEqual(set(r.feather), {"accel", "altitude", "error"})
                accel = r.feather["accel"]
                self.assertEqual(accel.values.shape, (2, 3))
                self.assertEqual(list(accel.timestamps_ms), [100, 130])
                self.assertEqual(accel.fields, ["x", "y", "z"])
                self.assertAlmostEqual(float(accel.values[0, 2]), 9.8, places=4)
                self.assertEqual(r.feather["altitude"].values.shape, (1, 1))
                err = r.feather["error"]
                self.assertEqual(list(err.timestamps_ms), [400])
                self.assertEqual(err.source[0], "mag")
                self.assertEqual(err.message[0], "boom")
                self.assertEqual(len(r), 2)  # frames unaffected

    def test_absent_feather_is_empty_dict(self):
        with TemporaryDirectory() as d:
            path = Path(d) / "s.h5"
            with HDF5Recorder(path) as rec:
                rec.append(solid_frame(), 0, pose_result())
            with Recording(path) as r:
                self.assertEqual(r.feather, {})


class ViewerLogSensorsTests(unittest.TestCase):
    def _records(self):
        return [
            SensorRecord("accel", fp.MSG_ACCEL, 100, [0.1, 0.2, 9.8], ("x", "y", "z")),
            SensorRecord("altitude", fp.MSG_ALTITUDE, 100, [75.5], ("altitude_m",)),
            SensorRecord("error", fp.MSG_ERROR, 120, ("mag", "boom"), ("source", "message")),
        ]

    def test_logs_scalars_per_field_and_error_text(self):
        with mock.patch("rerun_viewer.viewer.rr") as rr:
            from rerun_viewer.viewer import PoseRerunLogger
            logger = PoseRerunLogger(spawn=False, feather=True)
            logger.log_sensors(self._records(), elapsed_s=1.0)
        logged = [c.args[0] for c in rr.log.call_args_list]
        self.assertIn("feather/accel/x", logged)
        self.assertIn("feather/accel/z", logged)
        self.assertIn("feather/altitude/altitude_m", logged)
        self.assertIn("feather/error", logged)

    def test_blueprint_builds_with_and_without_feather(self):
        from rerun_viewer.viewer import _build_blueprint
        # Should not raise for any combination.
        for layout in ("split", "tabs"):
            for feather in (False, True):
                self.assertIsNotNone(_build_blueprint(layout, feather=feather))


if __name__ == "__main__":
    unittest.main()
