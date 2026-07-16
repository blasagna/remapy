"""Tests for the Feather Sense host integration.

Covers the serial decode/poll layer (``adafruit_feather_sense.stream``), the
host-side motion derivation (``adafruit_feather_sense.motion``), the recorder's
``/feather`` datasets + reader, and the rerun viewer's sensor logging. No board,
serial port, camera, or rerun viewer is touched: serial is a ``FakeSerial`` fed
pre-baked protocol frames, HDF5 is a real temp file, and the ``rerun`` SDK is
mocked.
"""

import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest import mock

import h5py
import numpy as np

# adafruit_feather_sense.stream inserts the dir holding feather_protocol on sys.path.
from adafruit_feather_sense import open_feather
from adafruit_feather_sense.motion import GravityFilter, derive_motion
from adafruit_feather_sense.stream import FeatherSenseStream, FrameRecordDecoder, SensorRecord
import feather_protocol as fp
from recording.reader import Recording
from recording.recorder import HDF5Recorder
from tests.fakes import FakeSerial, pose_result, solid_frame


def _baked_stream():
    """A byte blob of one frame per wire stream, including an error frame.

    Only raw streams appear: gravity/linear_accel are host-derived and never
    transmitted (see adafruit_feather_sense.motion).
    """
    blob = b""
    blob += fp.encode(fp.MSG_ACCEL, 100, fp.to_raw(fp.MSG_ACCEL, (0.0, 0.1, 9.81)))
    blob += fp.encode(fp.MSG_GYRO, 150, fp.to_raw(fp.MSG_GYRO, (0.01, -0.02, 0.03)))
    blob += fp.encode(fp.MSG_MAG, 200, fp.to_raw(fp.MSG_MAG, (-35.94, 25.17, -28.97)))
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
        self.assertEqual(names, ["accel", "gyro", "mag", "battery", "error"])
        by = {r.name: r for r in recs}
        # int32 wire values converted back to SI floats, each via its own scale.
        self.assertAlmostEqual(by["accel"].values[2], 9.81, places=2)
        self.assertAlmostEqual(by["gyro"].values[0], 0.01, places=3)
        self.assertAlmostEqual(by["mag"].values[0], -35.94, places=2)
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


class FrameRecordDecoderTests(unittest.TestCase):
    """The transport-agnostic decode shared by the serial and BLE streams."""

    def test_feed_yields_si_records(self):
        dec = FrameRecordDecoder()
        recs = dec.feed(_baked_stream())
        self.assertEqual([r.name for r in recs], ["accel", "gyro", "mag", "battery", "error"])
        by = {r.name: r for r in recs}
        self.assertAlmostEqual(by["accel"].values[2], 9.81, places=2)
        self.assertEqual(by["error"].values, ("mag", "boom"))  # source resolved to name
        self.assertEqual(dec.errors, 0)

    def test_feed_accepts_partial_chunks(self):
        blob = _baked_stream()
        whole = FrameRecordDecoder().feed(blob)
        piecemeal, dec = [], FrameRecordDecoder()
        for b in blob:
            piecemeal.extend(dec.feed(bytes([b])))
        self.assertEqual([r.name for r in piecemeal], [r.name for r in whole])


class OpenFeatherDispatchTests(unittest.TestCase):
    """The open_feather factory routes to the right backend (both mocked)."""

    def test_serial_transport(self):
        sentinel = object()
        with mock.patch(
            "adafruit_feather_sense.stream.FeatherSenseStream.open_if_available",
            return_value=sentinel,
        ) as m:
            result = open_feather("serial", port="/dev/ttyACM0")
        self.assertIs(result, sentinel)
        m.assert_called_once_with("/dev/ttyACM0")

    def test_ble_transport(self):
        sentinel = object()
        with mock.patch(
            "adafruit_feather_sense.ble_stream.FeatherSenseBLEStream.open_if_available",
            return_value=sentinel,
        ) as m:
            result = open_feather("ble", address="AA:BB:CC:DD:EE:FF")
        self.assertIs(result, sentinel)
        m.assert_called_once_with("AA:BB:CC:DD:EE:FF")

    def test_none_passthrough(self):
        with mock.patch(
            "adafruit_feather_sense.stream.FeatherSenseStream.open_if_available",
            return_value=None,
        ):
            self.assertIsNone(open_feather("serial"))

    def test_unknown_transport_raises(self):
        with self.assertRaises(ValueError):
            open_feather("carrier-pigeon")


class MotionDerivationTests(unittest.TestCase):
    """The host-side gravity/linear split that replaced the board's filter."""

    def test_first_sample_seeds_gravity_leaving_zero_linear(self):
        filt = GravityFilter()
        gravity, linear = filt.update(0, (0.0, 0.1, 9.81))
        self.assertEqual(gravity, (0.0, 0.1, 9.81))
        self.assertEqual(linear, (0.0, 0.0, 0.0))

    def test_sustained_tilt_bleeds_into_gravity(self):
        """Held long enough, a constant reading is all gravity again."""
        filt = GravityFilter(tau_s=0.05)
        filt.update(0, (0.0, 0.0, 9.81))
        for i in range(1, 200):  # 2s of a new orientation at 100 Hz
            gravity, linear = filt.update(i * 10, (9.81, 0.0, 0.0))
        np.testing.assert_allclose(gravity, [9.81, 0.0, 0.0], atol=1e-2)
        np.testing.assert_allclose(linear, [0.0, 0.0, 0.0], atol=1e-2)

    def test_brief_transient_lands_in_linear(self):
        filt = GravityFilter(tau_s=0.5)
        filt.update(0, (0.0, 0.0, 9.81))
        gravity, linear = filt.update(10, (2.0, 0.0, 9.81))  # 10ms spike
        self.assertAlmostEqual(linear[0], 2.0, places=1)  # ~all of it
        self.assertAlmostEqual(gravity[0], 0.0, places=1)  # gravity unmoved

    def test_non_advancing_timestamp_holds_estimate(self):
        """Duplicate timestamps / u32 clock wrap must not corrupt gravity."""
        filt = GravityFilter()
        first, _ = filt.update(100, (0.0, 0.0, 9.81))
        same, _ = filt.update(100, (5.0, 5.0, 5.0))
        wrapped, _ = filt.update(50, (5.0, 5.0, 5.0))
        self.assertEqual(same, first)
        self.assertEqual(wrapped, first)

    def test_derive_motion_matches_incremental_updates(self):
        """The batch helper and the live filter must agree sample for sample."""
        ts = [0, 20, 40, 60, 80]
        accel = [(0.0, 0.1, 9.8), (0.5, 0.1, 9.7), (0.0, 0.2, 9.9), (0.1, 0.0, 9.8), (0.0, 0.0, 9.8)]
        batch_g, batch_l = derive_motion(ts, accel)
        filt = GravityFilter()
        for i, (t, xyz) in enumerate(zip(ts, accel)):
            g, lin = filt.update(t, xyz)
            self.assertEqual(g, batch_g[i])
            self.assertEqual(lin, batch_l[i])

    def test_empty_stream_derives_nothing(self):
        self.assertEqual(derive_motion([], []), ([], []))


class RecorderFeatherTests(unittest.TestCase):
    def _record(self, path):
        no_pose = pose_result()  # no landmarks
        with HDF5Recorder(path) as rec:
            rec.append(solid_frame(), 0, no_pose)
            rec.append_sensor("accel", 100, [0.0, 0.1, 9.8], ("x", "y", "z"))
            rec.append_sensor("accel", 130, [0.0, 0.2, 9.7], ("x", "y", "z"))
            rec.append_sensor("gyro", 110, [0.01, 0.02, 0.03], ("x", "y", "z"))
            rec.append_sensor("error", 400, ["mag", "boom"], ("source", "message"))
            rec.append(solid_frame(), 33, no_pose)

    def test_roundtrip(self):
        with TemporaryDirectory() as d:
            path = Path(d) / "s.h5"
            self._record(path)
            with Recording(path) as r:
                accel = r.feather["accel"]
                self.assertEqual(accel.values.shape, (2, 3))
                self.assertEqual(list(accel.timestamps_ms), [100, 130])
                self.assertEqual(accel.fields, ["x", "y", "z"])
                self.assertAlmostEqual(float(accel.values[0, 2]), 9.8, places=4)
                self.assertFalse(accel.derived)
                self.assertEqual(r.feather["gyro"].values.shape, (1, 3))
                err = r.feather["error"]
                self.assertEqual(list(err.timestamps_ms), [400])
                self.assertEqual(err.source[0], "mag")
                self.assertEqual(err.message[0], "boom")
                self.assertEqual(len(r), 2)  # frames unaffected

    def test_only_raw_streams_are_stored(self):
        """gravity/linear_accel must not be written — they are derived on read."""
        with TemporaryDirectory() as d:
            path = Path(d) / "s.h5"
            self._record(path)
            with h5py.File(path, "r") as f:
                self.assertEqual(set(f["feather"]), {"accel", "gyro", "error"})

    def test_motion_derived_on_read(self):
        with TemporaryDirectory() as d:
            path = Path(d) / "s.h5"
            self._record(path)
            with Recording(path) as r:
                self.assertEqual(
                    set(r.feather), {"accel", "gyro", "error", "gravity", "linear_accel"}
                )
                grav, lin = r.feather["gravity"], r.feather["linear_accel"]
                self.assertTrue(grav.derived and lin.derived)
                # Derived streams inherit the raw accel's timeline exactly.
                self.assertEqual(list(grav.timestamps_ms), [100, 130])
                self.assertEqual(grav.values.shape, (2, 3))
                self.assertEqual(lin.fields, ["x", "y", "z"])
                # First sample seeds gravity from accel, so linear starts at 0.
                np.testing.assert_allclose(lin.values[0], [0.0, 0.0, 0.0], atol=1e-6)
                # dt=30ms against tau=0.5s => gravity barely moves, so the
                # accel step lands almost entirely in linear.
                self.assertAlmostEqual(float(lin.values[1, 1]), 0.0943, places=3)

    def test_motion_retunable_after_capture(self):
        """A shorter tau tracks accel faster, leaving less signal in linear."""
        with TemporaryDirectory() as d:
            path = Path(d) / "s.h5"
            self._record(path)
            with Recording(path) as r:
                default = r.feather["linear_accel"].values[1, 1]
                fast = r.motion(tau_s=0.01)["linear_accel"].values[1, 1]
                self.assertLess(abs(float(fast)), abs(float(default)))

    def test_absent_feather_is_empty_dict(self):
        with TemporaryDirectory() as d:
            path = Path(d) / "s.h5"
            with HDF5Recorder(path) as rec:
                rec.append(solid_frame(), 0, pose_result())
            with Recording(path) as r:
                self.assertEqual(r.feather, {})
                self.assertEqual(r.motion(), {})  # nothing to derive from


class ViewerLogSensorsTests(unittest.TestCase):
    def _records(self):
        return [
            SensorRecord("accel", fp.MSG_ACCEL, 100, [0.1, 0.2, 9.8], ("x", "y", "z")),
            SensorRecord("gyro", fp.MSG_GYRO, 100, [0.01, 0.02, 0.03], ("x", "y", "z")),
            SensorRecord("error", fp.MSG_ERROR, 120, ("mag", "boom"), ("source", "message")),
        ]

    def _logged(self, records):
        with mock.patch("rerun_viewer.viewer.rr") as rr:
            from rerun_viewer.viewer import PoseRerunLogger
            logger = PoseRerunLogger(spawn=False, feather=True)
            logger.log_sensors(records, elapsed_s=1.0)
        return [c.args[0] for c in rr.log.call_args_list]

    def test_logs_scalars_per_field_and_error_text(self):
        logged = self._logged(self._records())
        self.assertIn("feather/accel/x", logged)
        self.assertIn("feather/accel/z", logged)
        self.assertIn("feather/gyro/x", logged)
        self.assertIn("feather/error", logged)

    def test_accel_also_logs_derived_gravity_and_linear(self):
        """The board no longer sends these; the viewer filters them from accel."""
        logged = self._logged(self._records())
        for entity in ("feather/gravity/x", "feather/gravity/z",
                       "feather/linear_accel/x", "feather/linear_accel/z"):
            self.assertIn(entity, logged)

    def test_no_accel_means_no_derived_plots(self):
        gyro_only = [SensorRecord("gyro", fp.MSG_GYRO, 100, [0.0, 0.0, 0.0], ("x", "y", "z"))]
        logged = self._logged(gyro_only)
        self.assertFalse([e for e in logged if e.startswith("feather/gravity")])
        self.assertFalse([e for e in logged if e.startswith("feather/linear_accel")])

    def test_blueprint_builds_with_and_without_feather(self):
        from rerun_viewer.viewer import _build_blueprint
        # Should not raise for any combination.
        for layout in ("split", "tabs"):
            for feather in (False, True):
                self.assertIsNotNone(_build_blueprint(layout, feather=feather))


if __name__ == "__main__":
    unittest.main()
