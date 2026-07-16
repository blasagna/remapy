"""Transport-agnostic sensor sampling loop for the Feather Sense (board side).

`Telemetry` owns the `SensorHub`, the per-sensor schedule, and the frame
encoding; it knows nothing about *how* frames leave the board. Each transport
entry point (`board/serial/code.py`, `board/ble/code.py`) just supplies an
``emit(frame_bytes)`` callback and drives `pump()` in a loop, so the two builds
share identical sampling behavior and differ only in the sink and sample rates.

**The schedule runs on integer milliseconds, never `time.monotonic()` floats.**
CircuitPython builds single-precision floats, so `time.monotonic()`'s resolution
decays with uptime: its ULP is ~2 ms at 5.6 h and ~4 ms at 10 h. Feeding that
into a 10-20 ms schedule quantizes every deadline, and because a slot reschedules
from the *observed* time, the quantization is absorbed as a permanently longer
period rather than corrected. Measured on this board: 43.5 Hz against a 50 Hz
nominal at 5.6 h uptime, versus 48.9 Hz moments after a reflash — a rate that
silently decays the longer the board runs, while timestamps stay correct so
nothing looks wrong. `time.monotonic_ns()` is an integer and never loses
precision; ms also matches the wire timestamp unit.

Runs on the board, but `SensorHub` is imported lazily (see `__init__`) so this
module — and so the whole schedule — is importable and testable on the host.
"""

import time

import feather_protocol as fp


def _now_ms():
    return (time.monotonic_ns() // 1_000_000) & 0xFFFFFFFF


def _interval_ms(hz):
    """Sample interval in whole ms, at least 1."""
    return max(1, int(round(1000.0 / hz)))


class Telemetry:
    """Sample the sensors on schedule and emit COBS-framed TLV records.

    Sample rates are constructor args so a bandwidth-limited transport (BLE) can
    run slower than USB serial. Each sample is one frame.

    Only raw signals are sampled — gravity and linear acceleration are derived
    downstream on the host (see ``motion.py``), so one accelerometer read costs
    one frame rather than three.

    ``on_battery(percent)`` is an optional sink for consumers that want the
    battery level without paying for their own poll: it fires from the existing
    battery slot, right after the frame is encoded. It exists because a method
    call in the caller's hot loop is *not* free here — driving the status LED
    from `code.py` at every iteration measured ~0.6 accel samples/s (~1.2 %) in
    guard checks alone, versus ~0 riding this slot. It must not raise: an
    exception propagates into `pump`'s handler, costing that battery frame.
    Telemetry knows nothing about what consumes this (see ``status_led.py``).
    """

    def __init__(self, hub=None, imu_hz=50, mag_hz=20, battery_hz=0.2, on_battery=None):
        if hub is None:
            # Deferred so the host can import this module without `board`/`busio`
            # (the pattern `status_led.py` uses). Injecting a fake hub is what
            # makes the schedule testable off-board.
            from sensors import SensorHub

            hub = SensorHub()
        self._hub = hub

        # Hoisted out of the readers: a dict lookup per frame, ~200 times a
        # second, for values fixed at construction.
        accel_scale = fp.SCALES[fp.MSG_ACCEL][0]
        gyro_scale = fp.SCALES[fp.MSG_GYRO][0]
        mag_scale = fp.SCALES[fp.MSG_MAG][0]

        def imu():
            # One burst, one instant, one timestamp -> two frames. Accel and gyro
            # were always scheduled together (they share `imu_hz`), so this is a
            # slot fewer to walk as well as a read fewer to pay for.
            accel_xyz, gyro_xyz = hub.read_imu()
            now = _now_ms()
            return (
                fp.encode_xyz(fp.MSG_ACCEL, now, accel_xyz, accel_scale),
                fp.encode_xyz(fp.MSG_GYRO, now, gyro_xyz, gyro_scale),
            )

        def mag():
            return fp.encode_xyz(fp.MSG_MAG, _now_ms(), hub.read_mag(), mag_scale)

        def battery():
            v, pct, usb = hub.read_battery()
            frame = fp.encode(
                fp.MSG_BATTERY, _now_ms(), fp.to_raw(fp.MSG_BATTERY, (v, pct)), extra_u8=usb
            )
            if on_battery is not None:
                on_battery(pct)
            return frame

        # [interval_ms, next_due_ms, reader, source_type] — integer ms throughout.
        # A stream at 0 Hz is left out of the list rather than parked at a
        # far-future deadline: `pump` walks this every iteration, so a disabled
        # slot should cost nothing at all.
        self._schedule = [
            [_interval_ms(hz), 0, reader, source_type]
            for hz, reader, source_type in (
                (imu_hz, imu, fp.MSG_ACCEL),  # emits accel + gyro
                (mag_hz, mag, fp.MSG_MAG),
                (battery_hz, battery, fp.MSG_BATTERY),
            )
            if hz > 0
        ]

    def pump(self, now_ms, emit):
        """Emit every frame due at ``now_ms`` (integer ms, see `_now_ms`).

        ``emit(frame_bytes)`` is the transport sink. A failing sensor never kills
        the loop — it emits an ``error`` frame instead. Returns the absolute
        ``next_due_ms``; the caller derives its own sleep from that, which keeps
        this function pure (no clock read) and so testable with a fake clock.
        """
        next_due = now_ms + 1000
        for item in self._schedule:
            interval, due, reader, source_type = item
            if now_ms >= due:
                try:
                    result = reader()
                    if isinstance(result, (bytes, bytearray)):
                        emit(result)
                    else:
                        for frame in result:  # a reader may emit several frames
                            emit(frame)
                except Exception as exc:  # noqa: BLE001 - a flaky sensor must not stop the stream
                    try:
                        emit(fp.encode_error(_now_ms(), source_type, exc))
                    except Exception:  # noqa: BLE001 - error reporting must not either
                        pass
                # Advance from the *deadline*, not from `now_ms`: this slot was
                # served late (work + clock granularity), and rescheduling from
                # the observed time would bake that lateness into every
                # subsequent period instead of catching it up. That is what held
                # the loop below nominal.
                due += interval
                if due <= now_ms - interval:
                    # Fell far enough behind that catching up would emit a burst
                    # of stale samples (a long stall: BLE reconnect, a slow mag
                    # read). Drop the backlog and resync to the current time.
                    due = now_ms + interval
                item[1] = due
            next_due = min(next_due, item[1])
        return next_due
