"""Transport-agnostic sensor sampling loop for the Feather Sense (board side).

`Telemetry` owns the `SensorHub`, the per-sensor schedule, and the frame
encoding; it knows nothing about *how* frames leave the board. Each transport
entry point (`board/serial/code.py`, `board/ble/code.py`) just supplies an
``emit(frame_bytes)`` callback and drives `pump()` in a loop, so the two builds
share identical sampling behavior and differ only in the sink and sample rates.

Runs on the board (imports `sensors`); pure aside from that.
"""

import time

import feather_protocol as fp
from sensors import SensorHub


def _now_ms():
    return (time.monotonic_ns() // 1_000_000) & 0xFFFFFFFF


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
        self._hub = hub if hub is not None else SensorHub()
        hub = self._hub

        def accel():
            return fp.encode(fp.MSG_ACCEL, _now_ms(), fp.to_raw(fp.MSG_ACCEL, hub.read_accel()))

        def gyro():
            return fp.encode(fp.MSG_GYRO, _now_ms(), fp.to_raw(fp.MSG_GYRO, hub.read_gyro()))

        def mag():
            return fp.encode(fp.MSG_MAG, _now_ms(), fp.to_raw(fp.MSG_MAG, hub.read_mag()))

        def battery():
            v, pct, usb = hub.read_battery()
            frame = fp.encode(
                fp.MSG_BATTERY, _now_ms(), fp.to_raw(fp.MSG_BATTERY, (v, pct)), extra_u8=usb
            )
            if on_battery is not None:
                on_battery(pct)
            return frame

        # [interval_s, next_due_s, reader, source_type]
        self._schedule = [
            [1.0 / imu_hz, 0.0, accel, fp.MSG_ACCEL],
            [1.0 / imu_hz, 0.0, gyro, fp.MSG_GYRO],
            [1.0 / mag_hz, 0.0, mag, fp.MSG_MAG],
            [1.0 / battery_hz, 0.0, battery, fp.MSG_BATTERY],
        ]

    def pump(self, now, emit):
        """Emit every frame that is due at time ``now`` (seconds, monotonic).

        ``emit(frame_bytes)`` is the transport sink. A failing sensor never kills
        the loop — it emits an ``error`` frame instead. Returns the number of
        seconds until the next sample is due (for the caller to sleep).
        """
        next_wake = now + 1.0
        for item in self._schedule:
            interval, due, reader, source_type = item
            if now >= due:
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
                item[1] = now + interval
            next_wake = min(next_wake, item[1])
        return max(0.0, next_wake - time.monotonic())
