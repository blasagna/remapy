"""Host-side, non-blocking reader for the Feather Sense USB serial stream.

Wraps a serial port + :class:`feather_protocol.FrameDecoder` and exposes a
:meth:`FeatherSenseStream.poll` that returns whatever sensor records have arrived
since the last call, converted to SI units. It is designed to be *pumped* from an
existing loop (e.g. the camera loop in the rerun/recording apps) without blocking
on serial I/O.

Use :func:`FeatherSenseStream.open_if_available` to obtain a stream only when a
Feather Sense is actually plugged in and streaming (it probes for a valid frame),
returning ``None`` otherwise — so callers can transparently run with or without
the device.

This lives with the CircuitPython sources but only ever runs on the host; it
imports the shared :mod:`feather_protocol` (added to ``sys.path`` below so the
module resolves whether this file is imported as a package member or run
directly).
"""

import os
import sys
import time
from collections import namedtuple

import serial
from serial.tools import list_ports

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feather_protocol as fp  # noqa: E402

# One decoded, SI-converted sample.
#   name:         stream name, e.g. "accel" (fp.MSG_INFO)
#   msg_type:     the protocol type byte
#   timestamp_ms: device timestamp (ms since board boot, wraps ~49.7 days)
#   values:       list of SI floats; for "error", (source_type:int, message:str)
#   fields:       field labels aligned with values
SensorRecord = namedtuple("SensorRecord", "name msg_type timestamp_ms values fields")


def autodetect_port():
    """Return the most likely CircuitPython serial port, or None."""
    candidates = list(list_ports.comports())
    for port in candidates:
        desc = "%s %s" % (port.description or "", port.manufacturer or "")
        if "adafruit" in desc.lower() or "circuitpython" in desc.lower():
            return port.device
    for port in candidates:
        if "ttyACM" in port.device or "usbmodem" in port.device:
            return port.device
    return candidates[0].device if candidates else None


def to_si(msg_type, values):
    """Transform raw int32 wire values into SI-unit floats via the scale table.

    Fields without a scale (battery usb flag, error source/message) pass through
    unchanged. This is where the fixed-point integers become physical units.
    """
    scales = fp.SCALES.get(msg_type, ())
    return [val / scales[i] if i < len(scales) else val for i, val in enumerate(values)]


def _record(msg_type, timestamp_ms, values):
    name, fields = fp.MSG_INFO.get(msg_type, ("type0x%02x" % msg_type, ()))
    if msg_type == fp.MSG_ERROR:
        # Resolve the source-stream name so downstream consumers stay
        # independent of feather_protocol: values = (source_name, message).
        source_id, message = values
        source_name = fp.MSG_INFO.get(source_id, ("0x%02x" % source_id,))[0]
        return SensorRecord(name, msg_type, timestamp_ms, (source_name, message), fields)
    return SensorRecord(name, msg_type, timestamp_ms, to_si(msg_type, values), fields)


class RateTracker:
    """Per-stream sample-rate accounting for the reader CLIs' ``--stats`` mode.

    Reports two rates per stream, because they answer different questions:

    - **host** — samples counted over the *true* elapsed wall time. What arrived
      here. Divide by measured elapsed, never by the nominal report interval: a
      reader loop gated on ``>= 1.0`` s always overshoots (a blocking read
      overshoots by its own timeout), so counting `n` per "1 s" over-reports by
      however much the window ran long.
    - **dev** — derived from the *device* timestamps, which are stamped at read
      on the board. This measures what the board actually did, independent of
      host scheduling, USB buffering or BLE batching. It is the number to quote
      for a sample rate; ``host`` only tells you the link kept up.

    ``gap`` is the largest interval between consecutive device timestamps in the
    window. A rate at target with an outsized gap means the stream stalled and
    caught up in a burst — invisible in either average.
    """

    def __init__(self, interval_s=1.0):
        self.interval_s = interval_s
        self._start = time.monotonic()
        self._streams = {}  # name -> [count, first_ts, last_ts, max_gap_ms]

    def add(self, name, timestamp_ms=None):
        slot = self._streams.get(name)
        if slot is None:
            self._streams[name] = [1, timestamp_ms, timestamp_ms, 0]
            return
        slot[0] += 1
        if timestamp_ms is None:
            return
        if slot[1] is None:
            slot[1] = timestamp_ms
        elif slot[2] is not None:
            gap = timestamp_ms - slot[2]
            if gap > slot[3]:
                slot[3] = gap
        slot[2] = timestamp_ms

    def due(self):
        return time.monotonic() - self._start >= self.interval_s

    def report(self, errors=0):
        """Return the formatted report for this window and start a new one."""
        elapsed = time.monotonic() - self._start
        lines = ["--- %.3f s ---  errors=%d" % (elapsed, errors)]
        for name in sorted(self._streams):
            count, first_ts, last_ts, max_gap = self._streams[name]
            host_hz = count / elapsed if elapsed > 0 else 0.0
            line = "  %-9s n=%4d  host=%6.1f/s" % (name, count, host_hz)
            # (count - 1) intervals span first..last: an unbiased rate estimate
            # that needs no clock sync between the board and this host.
            if count > 1 and first_ts is not None and last_ts is not None and last_ts > first_ts:
                dev_hz = (count - 1) * 1000.0 / (last_ts - first_ts)
                line += "  dev=%6.1f/s  gap max=%6.1f ms" % (dev_hz, max_gap)
            lines.append(line)
        self._streams.clear()
        self._start = time.monotonic()
        return "\n".join(lines)


class FrameRecordDecoder:
    """Turn raw transport bytes into SI-converted :class:`SensorRecord`s.

    Wraps a :class:`feather_protocol.FrameDecoder` (COBS/TLV framing) and the
    per-type SI conversion, so every transport — USB serial or BLE — shares one
    decode path. ``feed(bytes) -> list[SensorRecord]``.
    """

    def __init__(self):
        self._decoder = fp.FrameDecoder()

    def feed(self, data):
        return [_record(mt, ts, values) for mt, ts, values in self._decoder.feed(data)]

    @property
    def errors(self):
        return self._decoder.errors


class FeatherSenseStream:
    """Pump a Feather Sense serial stream from an existing loop.

    Either pass an already-open serial-like object (``serial_obj``, handy for
    tests) or a ``port`` (``None`` = auto-detect). Prefer
    :meth:`open_if_available` when the device may be absent.
    """

    def __init__(self, port=None, baud=115200, serial_obj=None):
        if serial_obj is not None:
            self._ser = serial_obj
            self.port = getattr(serial_obj, "port", None)
        else:
            self.port = port or autodetect_port()
            if not self.port:
                raise RuntimeError("no serial port found for the Feather Sense")
            # timeout=0 -> reads never block; we drain in_waiting each poll.
            self._ser = serial.Serial(self.port, baud, timeout=0)
        self._decoder = FrameRecordDecoder()
        self._pending = []  # records decoded during a probe, returned on first poll

    def poll(self):
        """Return a list of SensorRecords that have arrived since the last poll.

        Non-blocking: reads only what the OS has already buffered.
        """
        out = self._pending
        self._pending = []
        waiting = getattr(self._ser, "in_waiting", 0)
        data = self._ser.read(waiting) if waiting else self._ser.read()
        if data:
            out.extend(self._decoder.feed(data))
        return out

    @property
    def errors(self):
        """Count of malformed frames skipped by the decoder."""
        return self._decoder.errors

    def close(self):
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:  # pragma: no cover - best-effort close
                pass
            self._ser = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    @classmethod
    def open_if_available(cls, port=None, baud=115200, probe_timeout=2.5):
        """Open the stream only if a Feather Sense is present and streaming.

        Opens the port and waits up to ``probe_timeout`` seconds for at least one
        valid frame (tolerating any CircuitPython banner text on connect). On
        success returns a live :class:`FeatherSenseStream` whose first
        :meth:`poll` includes the frames seen during probing; otherwise closes up
        and returns ``None``.
        """
        try:
            resolved = port or autodetect_port()
            if not resolved:
                return None
            stream = cls(port=resolved, baud=baud)
        except Exception:
            return None
        deadline = time.monotonic() + probe_timeout
        while time.monotonic() < deadline:
            records = stream.poll()
            if records:
                stream._pending = records + stream._pending
                return stream
            time.sleep(0.05)
        stream.close()
        return None
