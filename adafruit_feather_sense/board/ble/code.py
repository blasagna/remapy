"""Feather Sense BLE telemetry (CircuitPython entry point).

Streams the same COBS-framed TLV records as the USB serial build, but over the
**Nordic UART Service** (a BLE "serial pipe") instead of USB. The board
advertises as ``FeatherSense``; a host connects and subscribes to the UART TX
characteristic. The wire protocol is identical, so the host decodes it with the
same ``FrameDecoder`` (see ``ble_stream.py`` / ``read_ble.py``).

Because BLE offers only ~1-2 KB/s, this build samples the IMU at a reduced rate
(``Telemetry(imu_hz=20, mag_hz=10)``); ``uart.write`` back-pressures the loop to
the link's capacity. USB stays at full rate (``board/serial/code.py``).

Deploy (BLE build): ``circup install adafruit_ble`` once, then copy this file as
``code.py`` plus ``feather_protocol.py``, ``sensors.py`` and ``telemetry.py`` to
the CIRCUITPY drive root.
"""

import time

from adafruit_ble import BLERadio
from adafruit_ble.advertising.standard import ProvideServicesAdvertisement
from adafruit_ble.services.nordic import UARTService

from telemetry import Telemetry


def main():
    ble = BLERadio()
    ble.name = "FeatherSense"
    uart = UARTService()
    advertisement = ProvideServicesAdvertisement(uart)
    telemetry = Telemetry(imu_hz=20, mag_hz=10)  # reduced profile for the BLE link

    while True:
        ble.start_advertising(advertisement)
        while not ble.connected:
            pass
        ble.stop_advertising()
        # Stream until the central disconnects; a write raising means the link
        # dropped mid-frame, so fall back out to re-advertise.
        try:
            while ble.connected:
                delay = telemetry.pump(time.monotonic(), uart.write)
                if delay > 0:
                    time.sleep(delay)
        except Exception:  # noqa: BLE001 - disconnected mid-write -> re-advertise
            pass


main()
