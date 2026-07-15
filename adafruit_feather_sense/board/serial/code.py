"""Feather Sense USB serial telemetry (CircuitPython entry point).

Streams every onboard sensor over USB serial as COBS-framed TLV records (see
``feather_protocol``), using the shared ``Telemetry`` sampling loop. Each sample
is an independent frame, so fast motion data and slow environmental data coexist
on one link without bundling.

Deploy (USB serial build): copy this file as ``code.py`` plus ``feather_protocol.py``,
``sensors.py`` and ``telemetry.py`` to the CIRCUITPY drive root. Read it on the
host with ``read_stream.py`` (or the rerun/recording apps, default transport).

For the BLE build see ``board/ble/code.py``.
"""

import sys
import time

from telemetry import Telemetry


def _emit(frame):
    # Raw binary out the USB CDC data channel.
    sys.stdout.buffer.write(frame)


def main():
    telemetry = Telemetry()  # full USB rates (IMU 50 Hz)
    while True:
        delay = telemetry.pump(time.monotonic(), _emit)
        if delay > 0:
            time.sleep(delay)


main()
