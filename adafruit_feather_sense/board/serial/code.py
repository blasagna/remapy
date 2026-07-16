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

from sensors import SensorHub
from status_led import StatusLED
from telemetry import Telemetry


def _emit(frame):
    # Raw binary out the USB CDC data channel.
    sys.stdout.buffer.write(frame)


def main():
    hub = SensorHub()
    led = StatusLED(hub)
    # The LED rides the battery slot's existing read rather than polling from
    # this loop: calling it per-iteration measured ~1 accel sample/s.
    telemetry = Telemetry(hub=hub, on_battery=led.update)  # full USB rates (IMU 50 Hz)
    while True:
        delay = telemetry.pump(time.monotonic(), _emit)
        if delay > 0:
            time.sleep(delay)


main()
