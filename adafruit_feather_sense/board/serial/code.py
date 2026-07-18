"""Feather Sense USB serial telemetry (CircuitPython entry point).

Streams the board's raw motion sensors (acceleration, angular rate, magnetic
field) plus battery state over USB serial as COBS-framed TLV records (see
``feather_protocol``), using the shared ``Telemetry`` sampling loop. Each sample
is an independent frame, so streams at different rates coexist on one link
without bundling.

Deploy (USB serial build): copy this file as ``code.py`` plus ``feather_protocol.py``,
``sensors.py``, ``telemetry.py`` and ``status_led.py`` to the CIRCUITPY drive root.
Read it on the host with ``read_stream.py`` (or the rerun/recording apps, default
transport).

For the BLE build see ``board/ble/code.py``.
"""

import sys
import time

import supervisor

from sensors import SensorHub
from status_led import StatusLED
from telemetry import Telemetry, _now_ms

IMU_HZ = 100  # ~4.4 KB/s of frames; USB CDC is nowhere near a constraint

# CircuitPython's status bar writes an xterm title escape sequence to the
# console — `ESC]0;<snake> BLE:Off | code.py...` — and on this build the console
# *is* the data channel, so that text lands in the middle of the binary stream.
# It cost exactly one decode error per host attach: the host splits on 0x00,
# hands the escape sequence to `cobs_decode`, and correctly rejects it. Nothing
# framed is lost (the stream resyncs at the next delimiter) but a permanent
# `errors=1` trains you to ignore the error counter, which is the real damage.
# Must run before the first frame is emitted.
supervisor.status_bar.console = False


def _emit(frame):
    # Raw binary out the USB CDC data channel.
    sys.stdout.buffer.write(frame)


def main():
    # One number drives both, and they must agree: the hub needs it to clock the
    # sensor above the poll, the schedule needs it to pace the poll.
    hub = SensorHub(imu_hz=IMU_HZ)
    led = StatusLED(hub)
    # The LED rides the battery slot's existing read rather than polling from
    # this loop: calling it per-iteration measured ~1 accel sample/s. The
    # charging pulse needs a faster tick than that slot's 0.2 Hz, so it gets its
    # own scheduled slot — still not a per-iteration call.
    telemetry = Telemetry(hub=hub, imu_hz=IMU_HZ, on_battery=led.update, on_pulse=led.pulse)
    while True:
        next_due_ms = telemetry.pump(_now_ms(), _emit)
        delay = (next_due_ms - _now_ms()) / 1000
        if delay > 0:
            time.sleep(delay)


main()
