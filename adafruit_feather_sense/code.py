"""Feather Sense USB serial telemetry (CircuitPython entry point).

Streams every onboard sensor over USB serial as COBS-framed TLV records (see
``feather_protocol``). Each stream is sampled on its own schedule, and each
sample is emitted as an independent frame, so fast motion data and slow
environmental data coexist on one link without bundling.

Deploy: copy this file plus ``sensors.py`` and ``feather_protocol.py`` to the
CIRCUITPY drive root. Read it on the host with ``read_stream.py``.

Future BLE: swap the ``emit`` sink for a BLE characteristic write — the
scheduler, ``SensorHub`` and ``feather_protocol.encode`` are transport-agnostic.
"""

import sys
import time

import feather_protocol as fp
from sensors import SensorHub


def _now_ms():
    return (time.monotonic_ns() // 1_000_000) & 0xFFFFFFFF


def _emit(frame):
    # Raw binary out the USB CDC data channel.
    sys.stdout.buffer.write(frame)


def _emit_error(source_type, exc):
    # Stream a caught sampling/encoding failure; never let error reporting
    # itself take down the loop.
    try:
        _emit(fp.encode_error(_now_ms(), source_type, exc))
    except Exception:  # noqa: BLE001
        pass


def main():
    hub = SensorHub()

    # Each reader converts SI sensor values to scaled int32 (fp.to_raw) and
    # encodes a frame (or several). Different rates on purpose.
    def accel():
        # One accelerometer read, decomposed into three time-aligned frames:
        # total (raw, includes gravity), estimated gravity, and linear (motion).
        ts = _now_ms()
        total, gravity, linear = hub.read_motion()
        return (
            fp.encode(fp.MSG_ACCEL, ts, fp.to_raw(fp.MSG_ACCEL, total)),
            fp.encode(fp.MSG_GRAVITY, ts, fp.to_raw(fp.MSG_GRAVITY, gravity)),
            fp.encode(fp.MSG_LINEAR_ACCEL, ts, fp.to_raw(fp.MSG_LINEAR_ACCEL, linear)),
        )

    def gyro():
        return fp.encode(fp.MSG_GYRO, _now_ms(), fp.to_raw(fp.MSG_GYRO, hub.read_gyro()))

    def mag():
        return fp.encode(fp.MSG_MAG, _now_ms(), fp.to_raw(fp.MSG_MAG, hub.read_mag()))

    def env():
        return fp.encode(fp.MSG_ENV, _now_ms(), fp.to_raw(fp.MSG_ENV, hub.read_env()))

    def altitude():
        return fp.encode(fp.MSG_ALTITUDE, _now_ms(), fp.to_raw(fp.MSG_ALTITUDE, (hub.read_altitude(),)))

    def battery():
        v, pct, usb = hub.read_battery()
        return fp.encode(fp.MSG_BATTERY, _now_ms(), fp.to_raw(fp.MSG_BATTERY, (v, pct)), extra_u8=usb)

    # [interval_s, next_due_s, reader, source_type]
    schedule = [
        [1 / 50, 0.0, accel, fp.MSG_ACCEL],       # 50 Hz
        [1 / 50, 0.0, gyro, fp.MSG_GYRO],         # 50 Hz
        [1 / 20, 0.0, mag, fp.MSG_MAG],           # 20 Hz
        [1.0, 0.0, env, fp.MSG_ENV],              # 1 Hz
        [1.0, 0.0, altitude, fp.MSG_ALTITUDE],    # 1 Hz
        [5.0, 0.0, battery, fp.MSG_BATTERY],      # 0.2 Hz
    ]

    while True:
        now = time.monotonic()
        next_wake = now + 1.0
        for item in schedule:
            interval, due, reader, source_type = item
            if now >= due:
                try:
                    result = reader()
                    if isinstance(result, (bytes, bytearray)):
                        _emit(result)
                    else:
                        for frame in result:  # a reader may emit several frames
                            _emit(frame)
                except Exception as exc:  # noqa: BLE001 - a flaky sensor must not kill the stream
                    _emit_error(source_type, exc)
                item[1] = now + interval
            next_wake = min(next_wake, item[1])
        sleep_for = next_wake - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)


main()
