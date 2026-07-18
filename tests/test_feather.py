"""Tests for the Feather Sense host integration.

Covers the serial decode/poll layer (``adafruit_feather_sense.stream``), the
host-side motion derivation (``adafruit_feather_sense.motion``), the recorder's
``/feather`` datasets + reader, and the rerun viewer's sensor logging. No board,
serial port, camera, or rerun viewer is touched: serial is a ``FakeSerial`` fed
pre-baked protocol frames, HDF5 is a real temp file, and the ``rerun`` SDK is
mocked.

The board-side exception is ``status_led``: its ``band_for`` thresholds/hysteresis
are pure, and ``StatusLED``'s write path runs against an injected ``FakePixel``.
Both stay importable here because ``StatusLED`` defers its ``board``/``neopixel``
imports into ``__init__``.
"""

import importlib
import random
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest import mock

import h5py
import numpy as np

# adafruit_feather_sense.stream inserts the dir holding feather_protocol on sys.path.
from adafruit_feather_sense import open_feather
from adafruit_feather_sense.motion import GravityFilter, derive_motion
from adafruit_feather_sense.status_led import (
    GREEN,
    PULSE_PERIOD_MS,
    RED,
    YELLOW,
    StatusLED,
    band_for,
    pulse_level,
)
from adafruit_feather_sense.stream import (
    FeatherSenseStream,
    FrameRecordDecoder,
    RateTracker,
    SensorRecord,
)
import feather_protocol as fp
from telemetry import Telemetry  # board-side, but imports `sensors` lazily
from recording.reader import Recording
from recording.recorder import HDF5Recorder
from tests.fakes import FakePixel, FakeSerial, pose_result, solid_frame


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


def _cobs_encode_reference(data):
    """The original per-byte COBS encoder, kept verbatim as a test oracle.

    `fp.cobs_encode` was rewritten for speed on the board; "faster" is only
    acceptable if it is also *identical*, and this is what identical means.
    """
    out = bytearray()
    code_index = 0
    out.append(0)
    code = 1
    for byte in data:
        if byte != 0:
            out.append(byte)
            code += 1
            if code == 0xFF:
                out[code_index] = code
                code_index = len(out)
                out.append(0)
                code = 1
        else:
            out[code_index] = code
            code_index = len(out)
            out.append(0)
            code = 1
    out[code_index] = code
    return bytes(out)


class CobsEncodeTests(unittest.TestCase):
    """The split-based rewrite must match the byte loop it replaced, exactly."""

    def _cases(self):
        rng = random.Random(20260716)
        cases = [b"", b"\x00", b"\x01", b"\x00\x00", b"\x01\x00\x02"]
        # The block-split boundary: a 0xFF code means "254 non-zero bytes, no
        # zero after". Unreachable for real frames, so only a test will ever
        # exercise it.
        for n in (253, 254, 255, 256, 507, 508, 509, 1000):
            cases.append(b"\x01" * n)
            cases.append(b"\x01" * n + b"\x00")
            cases.append(b"\x00" + b"\x01" * n)
        for _ in range(2000):
            n = rng.randint(0, 600)
            cases.append(bytes(rng.randint(0, 255) for _ in range(n)))
        for _ in range(2000):  # zero-heavy, like a real fixed-point payload
            n = rng.randint(0, 40)
            cases.append(bytes(rng.choice([0, 0, 0, rng.randint(1, 255)]) for _ in range(n)))
        return cases

    def test_identical_to_the_reference_encoder(self):
        for data in self._cases():
            self.assertEqual(fp.cobs_encode(data), _cobs_encode_reference(data), data.hex())

    def test_roundtrips_and_never_emits_a_zero(self):
        for data in self._cases():
            encoded = fp.cobs_encode(data)
            self.assertNotIn(0, encoded, data.hex())
            self.assertEqual(fp.cobs_decode(encoded), data, data.hex())

    def test_matches_an_independent_implementation(self):
        # The `cobs` PyPI package as a third-party oracle. It diverges from ours
        # at a 254-byte zero-free run (a documented COBS variation), which real
        # frames never reach -- max payload is ~207 bytes.
        cobs = importlib.import_module("cobs.cobs")
        rng = random.Random(7)
        for _ in range(2000):
            data = bytes(rng.randint(0, 255) for _ in range(rng.randint(0, 200)))
            self.assertEqual(fp.cobs_encode(data), cobs.encode(data), data.hex())


class EncodeXyzTests(unittest.TestCase):
    """The fused encode must be byte-identical to to_raw + generic encode."""

    def test_identical_to_the_generic_path(self):
        rng = random.Random(1234)
        for msg_type in (fp.MSG_ACCEL, fp.MSG_GYRO, fp.MSG_MAG):
            scale = fp.SCALES[msg_type][0]
            for _ in range(500):
                xyz = tuple(rng.uniform(-200.0, 200.0) for _ in range(3))
                ts = rng.randint(0, 2**32 - 1)
                self.assertEqual(
                    fp.encode_xyz(msg_type, ts, xyz, scale),
                    fp.encode(msg_type, ts, fp.to_raw(msg_type, xyz)),
                )

    def test_decodes_back_to_the_input(self):
        frame = fp.encode_xyz(fp.MSG_ACCEL, 1234, (0.5, -1.25, 9.81), 1000)
        recs = FrameRecordDecoder().feed(frame)
        self.assertEqual(recs[0].name, "accel")
        self.assertEqual(recs[0].timestamp_ms, 1234)
        self.assertAlmostEqual(recs[0].values[2], 9.81, places=3)

    def test_timestamp_wraps_rather_than_overflowing_the_pack(self):
        # ~49.7 days of uptime wraps the u32; it must not raise.
        frame = fp.encode_xyz(fp.MSG_ACCEL, 2**32 + 5, (0.0, 0.0, 0.0), 1000)
        self.assertEqual(FrameRecordDecoder().feed(frame)[0].timestamp_ms, 5)


class _FakeHub:
    """A SensorHub stand-in: constant readings, optionally failing one stream."""

    def __init__(self, failing=None):
        self.failing = failing
        self.reads = {}

    def _count(self, name):
        self.reads[name] = self.reads.get(name, 0) + 1
        if self.failing == name:
            raise RuntimeError("boom")

    def read_imu(self):
        self._count("imu")
        return (0.0, 0.0, 9.81), (0.0, 0.0, 0.0)

    def read_mag(self):
        self._count("mag")
        return (0.0, 0.0, 0.0)

    def read_battery(self):
        self._count("battery")
        return (4.0, 80.0, 1)


class TelemetryScheduleTests(unittest.TestCase):
    """The board's sampling schedule, driven by a fake clock and a fake hub.

    Board-side code, but reachable on the host because `Telemetry` defers its
    `sensors` import. This is the only coverage the sampling loop has.
    """

    def _run(self, telemetry, duration_ms, step_ms=1, lateness_ms=0, start_ms=0):
        """Drive `pump` over a simulated span; return decoded frames by stream."""
        frames = []
        now = start_ms
        end = start_ms + duration_ms
        while now < end:
            next_due = telemetry.pump(now + lateness_ms, frames.append)
            now = max(next_due, now + step_ms)
        dec = FrameRecordDecoder()
        out = {}
        for rec in dec.feed(b"".join(frames)):
            out.setdefault(rec.name, []).append(rec)
        return out

    def test_hits_nominal_rate(self):
        tel = Telemetry(hub=_FakeHub(), imu_hz=100, mag_hz=20)
        got = self._run(tel, 1000)
        # A slot fires at t=0 and every 10 ms after, so 1000 ms holds 100.
        self.assertEqual(len(got["accel"]), 100)
        self.assertEqual(len(got["gyro"]), 100)
        self.assertEqual(len(got["mag"]), 20)

    def test_sustained_lateness_does_not_erode_the_rate(self):
        # The regression test for the original `due = now + interval`. Every slot
        # is served 3 ms late, which that form baked into the next period (13 ms
        # instead of 10) and so lost ~23 % of the rate outright. Advancing from
        # the deadline absorbs the lateness instead of compounding it.
        tel = Telemetry(hub=_FakeHub(), imu_hz=100, mag_hz=20)
        got = self._run(tel, 1000, lateness_ms=3)
        self.assertEqual(len(got["accel"]), 100)

    def test_rate_holds_at_high_uptime(self):
        # The float32 `time.monotonic()` decay this schedule exists to dodge:
        # at ~5.6 h uptime the old float clock quantized to ~2 ms and the rate
        # sagged to 43.5 Hz. Integer ms has no such term — a large start offset
        # must change nothing.
        tel = Telemetry(hub=_FakeHub(), imu_hz=100, mag_hz=20)
        got = self._run(tel, 1000, start_ms=20_110_454)
        self.assertEqual(len(got["accel"]), 100)

    def test_a_long_stall_drops_the_backlog_instead_of_bursting(self):
        tel = Telemetry(hub=_FakeHub(), imu_hz=100, mag_hz=20)
        frames = []
        tel.pump(0, frames.append)
        frames.clear()
        # Nothing ran for 500 ms (a BLE reconnect). Catching up honestly would
        # mean 50 stale accel samples; the clamp emits one and resyncs.
        tel.pump(500, frames.append)
        got = {}
        for rec in FrameRecordDecoder().feed(b"".join(frames)):
            got.setdefault(rec.name, []).append(rec)
        self.assertEqual(len(got["accel"]), 1)

    def test_a_failing_sensor_emits_an_error_and_spares_the_others(self):
        tel = Telemetry(hub=_FakeHub(failing="mag"), imu_hz=100, mag_hz=20)
        got = self._run(tel, 1000)
        self.assertNotIn("mag", got)
        self.assertEqual(len(got["error"]), 20)  # one per attempted mag read
        self.assertEqual(got["error"][0].values, ("mag", "boom"))
        self.assertEqual(len(got["accel"]), 100)  # unaffected

    def test_accel_and_gyro_share_one_timestamp(self):
        # The point of the burst read: they are one sample, so they must not
        # carry two different times.
        tel = Telemetry(hub=_FakeHub(), imu_hz=100, mag_hz=20)
        got = self._run(tel, 200)
        pairs = list(zip(got["accel"], got["gyro"]))
        self.assertTrue(pairs)
        for accel, gyro in pairs:
            self.assertEqual(accel.timestamp_ms, gyro.timestamp_ms)

    def test_one_imu_read_serves_both_streams(self):
        hub = _FakeHub()
        tel = Telemetry(hub=hub, imu_hz=100, mag_hz=20)
        got = self._run(tel, 1000)
        self.assertEqual(len(got["accel"]), 100)
        self.assertEqual(len(got["gyro"]), 100)
        self.assertEqual(hub.reads["imu"], 100)  # not 200

    def test_on_battery_fires_from_the_battery_slot(self):
        seen = []
        tel = Telemetry(
            hub=_FakeHub(),
            imu_hz=100,
            battery_hz=10,
            on_battery=lambda pct, usb: seen.append((pct, usb)),
        )
        self._run(tel, 1000)
        self.assertEqual(len(seen), 10)
        self.assertAlmostEqual(seen[0][0], 80.0)

    def test_on_battery_is_told_the_usb_state(self):
        # The LED needs it: a reading taken on USB is an upper bound, not a
        # level, so the consumer has to know which kind it got.
        seen = []
        tel = Telemetry(
            hub=_FakeHub(),
            imu_hz=100,
            battery_hz=10,
            on_battery=lambda pct, usb: seen.append(usb),
        )
        self._run(tel, 1000)
        self.assertEqual(set(seen), {_FakeHub().read_battery()[2]})

    def test_on_pulse_runs_on_its_own_slot_and_emits_nothing(self):
        # The charging ramp can't ride the 0.2 Hz battery slot, so it gets its
        # own — but it drives a display, so it must not put a frame on the wire.
        seen = []
        tel = Telemetry(hub=_FakeHub(), imu_hz=100, mag_hz=0, battery_hz=0, on_pulse=seen.append)
        got = self._run(tel, 1000)
        self.assertEqual(len(seen), 15)  # default pulse_hz
        self.assertEqual(sorted(got), ["accel", "gyro"])

    def test_no_on_pulse_leaves_the_schedule_untouched(self):
        # `pump` walks the schedule every iteration, so an unused slot must not
        # be in it at all.
        tel = Telemetry(hub=_FakeHub(), imu_hz=100)
        self.assertEqual(len(tel._schedule), 3)
        with_pulse = Telemetry(hub=_FakeHub(), imu_hz=100, on_pulse=lambda _ms: None)
        self.assertEqual(len(with_pulse._schedule), 4)

    def test_a_raising_on_battery_costs_only_its_own_frame(self):
        # `on_battery` fires inside the battery reader, so an exception lands in
        # pump's handler: the battery frame becomes an error frame. Everything
        # else must be untouched.
        def boom(_pct, _usb):
            raise RuntimeError("led died")

        tel = Telemetry(hub=_FakeHub(), imu_hz=100, battery_hz=10, on_battery=boom)
        got = self._run(tel, 1000)
        self.assertEqual(len(got["accel"]), 100)
        self.assertNotIn("battery", got)
        self.assertEqual(len(got["error"]), 10)

    def test_zero_hz_disables_a_stream_without_dividing_by_zero(self):
        tel = Telemetry(hub=_FakeHub(), imu_hz=100, mag_hz=0)
        got = self._run(tel, 1000)
        self.assertEqual(sorted(got), ["accel", "battery", "gyro"])
        self.assertEqual(len(got["accel"]), 100)


class RateTrackerTests(unittest.TestCase):
    """The --stats accounting. This is the project's acceptance instrument, so a
    bias here silently rewrites every measured number in the README."""

    def _report(self, elapsed, add):
        """Run one window of `add(tracker)` over a faked `elapsed` wall time."""
        clock = [1000.0]
        with mock.patch("adafruit_feather_sense.stream.time.monotonic", lambda: clock[0]):
            tracker = RateTracker()
            add(tracker)
            clock[0] += elapsed
            return tracker.report()

    def test_host_rate_divides_by_true_elapsed_not_the_nominal_interval(self):
        # The regression test for the original bug: 100 samples counted over a
        # window that really ran 1.1 s is 90.9/s, not the "100/s" a raw count
        # printed. The reader loop only ever overshoots 1.0 s (a blocking read
        # overshoots by its own timeout), so the error was always in the
        # flattering direction.
        out = self._report(1.1, lambda t: [t.add("accel", i * 10) for i in range(100)])
        self.assertIn("host=  90.9/s", out)
        # ...and `dev` recovers the truth the host window obscured: the device
        # timestamps are 10 ms apart, so the board really was at 100 Hz.
        self.assertIn("dev= 100.0/s", out)

    def test_device_rate_is_independent_of_host_timing(self):
        # Same samples, same device clock, but the host took 2x as long to read
        # them: `dev` must not move, because the board's spacing didn't.
        fast = self._report(1.0, lambda t: [t.add("accel", i * 10) for i in range(101)])
        slow = self._report(2.0, lambda t: [t.add("accel", i * 10) for i in range(101)])
        self.assertIn("dev= 100.0/s", fast)
        self.assertIn("dev= 100.0/s", slow)
        self.assertIn("host= 101.0/s", fast)
        self.assertIn("host=  50.5/s", slow)

    def test_max_gap_exposes_a_stall_that_the_average_hides(self):
        # 50 samples at 10 ms then one 500 ms stall: the mean rate still looks
        # respectable, so the gap is the only thing that reveals the burst. This
        # is the BLE "link saturated" signature.
        def add(t):
            for i in range(50):
                t.add("accel", i * 10)
            t.add("accel", 990)

        out = self._report(1.0, add)
        self.assertIn("gap max= 500.0 ms", out)

    def test_window_resets_between_reports(self):
        clock = [1000.0]
        with mock.patch("adafruit_feather_sense.stream.time.monotonic", lambda: clock[0]):
            tracker = RateTracker()
            tracker.add("accel", 0)
            clock[0] += 1.0
            self.assertIn("accel", tracker.report())
            self.assertFalse(tracker.due())  # clock restarted
            clock[0] += 1.0
            self.assertNotIn("accel", tracker.report())  # counts did not carry over

    def test_single_sample_reports_host_rate_but_no_device_rate(self):
        # One sample spans no interval, so a device rate would be a fabrication.
        out = self._report(1.0, lambda t: t.add("accel", 5))
        self.assertIn("n=   1", out)
        self.assertNotIn("dev=", out)


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


class StatusLEDBandTests(unittest.TestCase):
    """The board's battery indicator: the pure band/hysteresis logic.

    `StatusLED` defers its `board`/`neopixel` imports into __init__, which is
    what keeps this module importable here; the write logic is exercised with an
    injected `FakePixel` in `StatusLEDUpdateTests`.
    """

    def test_bands_across_the_range(self):
        self.assertEqual(band_for(0.0), RED)
        self.assertEqual(band_for(10.0), RED)
        self.assertEqual(band_for(40.0), YELLOW)
        self.assertEqual(band_for(80.0), GREEN)
        self.assertEqual(band_for(100.0), GREEN)

    def test_bare_thresholds_apply_when_nothing_is_lit(self):
        self.assertEqual(band_for(24.9), RED)
        self.assertEqual(band_for(25.0), YELLOW)
        self.assertEqual(band_for(59.9), YELLOW)
        self.assertEqual(band_for(60.0), GREEN)

    def test_hysteresis_resists_climbing_out_of_a_band(self):
        # Charging over USB: the level rises, but the band should not flip until
        # it clears the edge by the hysteresis margin.
        self.assertEqual(band_for(26.0, RED), RED)
        self.assertEqual(band_for(28.0, RED), YELLOW)
        self.assertEqual(band_for(61.0, YELLOW), YELLOW)
        self.assertEqual(band_for(63.0, YELLOW), GREEN)

    def test_hysteresis_resists_falling_out_of_a_band(self):
        # Discharging: same margin in the other direction. Edges are exclusive
        # below, so sitting *on* the shifted edge still holds the band.
        self.assertEqual(band_for(58.0, GREEN), GREEN)
        self.assertEqual(band_for(57.0, GREEN), GREEN)
        self.assertEqual(band_for(56.9, GREEN), YELLOW)
        self.assertEqual(band_for(23.0, YELLOW), YELLOW)
        self.assertEqual(band_for(22.0, YELLOW), YELLOW)
        self.assertEqual(band_for(21.9, YELLOW), RED)

    def test_a_reading_resting_on_an_edge_does_not_oscillate(self):
        # The flicker case hysteresis exists for: hold at a bare threshold and
        # the displayed color must stay put once chosen.
        color = band_for(25.0)
        for _ in range(10):
            self.assertEqual(band_for(25.0, color), color)


class _BatteryHub:
    """Minimal SensorHub stand-in: only `read_battery` is reached here."""

    def __init__(self, percent=90.0, usb=1):
        self.percent = percent
        self.usb = usb
        self.reads = 0

    def read_battery(self):
        self.reads += 1
        return (4.1, self.percent, self.usb)


class StatusLEDUpdateTests(unittest.TestCase):
    """The write path, via an injected pixel (no board, no neopixel)."""

    def test_lights_the_band_for_the_level(self):
        px = FakePixel()
        StatusLED(_BatteryHub(), pixel=px).update(90.0)
        self.assertEqual(px.fills, [GREEN])
        self.assertEqual(px.shows, 1)

    def test_writes_only_when_the_band_changes(self):
        # show() bit-bangs the pixel with interrupts off; repainting an unchanged
        # color would spend loop time to display nothing new.
        px = FakePixel()
        led = StatusLED(_BatteryHub(), pixel=px)
        for pct in (90.0, 88.0, 75.0, 61.0):
            led.update(pct)
        self.assertEqual(px.fills, [GREEN])
        self.assertEqual(px.shows, 1)

    def test_rewrites_when_the_band_does_change(self):
        px = FakePixel()
        led = StatusLED(_BatteryHub(), pixel=px)
        led.update(90.0)
        led.update(10.0)
        led.update(40.0)
        self.assertEqual(px.fills, [GREEN, RED, YELLOW])
        self.assertEqual(px.shows, 3)

    def test_update_never_raises_on_a_failing_pixel(self):
        class BadPixel:
            def fill(self, color):
                raise RuntimeError("pixel is on fire")

            def show(self):
                pass

        StatusLED(_BatteryHub(), pixel=BadPixel()).update(50.0)  # must not raise

    def test_no_pixel_is_a_silent_no_op(self):
        # The host case (and a board missing the neopixel lib): stream lives on.
        led = StatusLED(_BatteryHub())
        self.assertIsNone(led._pixel)
        led.update(50.0)
        led.tick(0.0)
        self.assertIsNone(led._color)

    def test_tick_self_drives_but_rate_limits(self):
        # Used only in the BLE advertising gap, where pump() isn't running.
        hub = _BatteryHub(percent=10.0)
        led = StatusLED(hub, interval_s=5.0, pixel=FakePixel())
        led.tick(0.0)
        led.tick(1.0)
        led.tick(4.9)
        self.assertEqual(hub.reads, 1)
        self.assertEqual(led._color, RED)
        led.tick(5.0)
        self.assertEqual(hub.reads, 2)

    def test_charging_pulses_the_band_color(self):
        # Same hue, ramping brightness — the color still carries the level, the
        # animation carries "charging".
        px = FakePixel()
        led = StatusLED(_BatteryHub(), pixel=px)
        led.update(10.0, usb=0)  # learn a real level first: red
        led.update(10.0, usb=1)
        px.fills.clear()
        for ms in range(0, PULSE_PERIOD_MS, 50):
            led.pulse(ms)
        self.assertTrue(px.fills)
        # Every frame is red at some brightness, never another hue.
        for r, g, b in px.fills:
            self.assertEqual((g, b), (0, 0))
            self.assertLessEqual(r, RED[0])
        self.assertGreater(max(f[0] for f in px.fills), min(f[0] for f in px.fills))

    def test_the_pulse_never_goes_fully_dark(self):
        # A ramp that hits zero reads as a fault, and loses the color for half
        # the cycle.
        levels = [pulse_level(ms) for ms in range(0, PULSE_PERIOD_MS, 10)]
        self.assertGreater(min(levels), 0.0)
        self.assertAlmostEqual(max(levels), 1.0)

    def test_pulse_is_a_no_op_off_usb(self):
        # The slot runs regardless; on battery it must not touch the pixel.
        px = FakePixel()
        led = StatusLED(_BatteryHub(), pixel=px)
        led.update(90.0, usb=0)
        px.fills.clear()
        for ms in range(0, PULSE_PERIOD_MS, 50):
            led.pulse(ms)
        self.assertEqual(px.fills, [])

    def test_usb_selects_the_animation_not_the_band(self):
        # The regression this replaces: the LED used to cap the band while
        # charging, which (the ceiling living in RAM, the board almost always
        # booting plugged in) pinned it to yellow forever. `VOLTAGE_MONITOR`
        # reads the battery terminal, not the charger — measured at 4.00 V/80 %
        # and 4.09 V/90 % on two packs — so the reading is the band, plugged in
        # or not.
        for usb in (0, 1):
            for pct, want in ((10.0, RED), (40.0, YELLOW), (90.0, GREEN)):
                led = StatusLED(_BatteryHub(), pixel=FakePixel())
                led.update(pct, usb=usb)
                self.assertEqual(led._color, want, "pct=%s usb=%s" % (pct, usb))

    def test_an_almost_full_pack_reads_green_while_charging(self):
        # The user-visible bug: a nearly-full battery on the charger showed
        # amber. It must show the band it actually is.
        led = StatusLED(_BatteryHub(), pixel=FakePixel())
        led.update(89.5, usb=1)  # the measured reading from the real board
        self.assertEqual(led._color, GREEN)
        self.assertTrue(led._charging)

    def test_a_low_pack_still_reads_red_while_charging(self):
        led = StatusLED(_BatteryHub(), pixel=FakePixel())
        led.update(8.0, usb=1)
        self.assertEqual(led._color, RED)

    def test_the_battery_slot_does_not_stamp_over_a_ramp(self):
        # `update` fires at 0.2 Hz mid-pulse; if it repainted full brightness
        # every time, the ramp would glitch once every 5 s.
        px = FakePixel()
        led = StatusLED(_BatteryHub(), pixel=px)
        led.update(10.0, usb=0)
        led.update(10.0, usb=1)
        led.pulse(PULSE_PERIOD_MS // 2)  # peak
        led.pulse(PULSE_PERIOD_MS // 4 + PULSE_PERIOD_MS)  # partway down
        px.fills.clear()
        led.update(10.0, usb=1)  # same band, mid-ramp
        self.assertEqual(px.fills, [])

    def test_pulse_never_raises_on_a_failing_pixel(self):
        class BadPixel:
            def fill(self, color):
                raise RuntimeError("pixel is on fire")

            def show(self):
                pass

        led = StatusLED(_BatteryHub(), pixel=BadPixel())
        led.update(50.0, usb=1)
        led.pulse(0)  # must not raise

    def test_tick_survives_a_failing_battery_read(self):
        class BadHub:
            def read_battery(self):
                raise RuntimeError("adc gone")

        led = StatusLED(BadHub(), pixel=FakePixel())
        led.tick(0.0)  # must not raise
        self.assertIsNone(led._color)


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
