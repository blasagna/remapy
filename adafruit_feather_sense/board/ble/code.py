"""Feather Sense BLE telemetry (CircuitPython entry point).

Streams the same COBS-framed TLV records as the USB serial build (raw motion +
battery), but over the **Nordic UART Service** (a BLE "serial pipe") instead of
USB. The board
advertises as ``FeatherSense``; a host connects and subscribes to the UART TX
characteristic. The wire protocol is identical, so the host decodes it with the
same ``FrameDecoder`` (see ``ble_stream.py`` / ``read_ble.py``).

Because BLE offers only ~1-2 KB/s, this build samples the IMU at a reduced rate
(``Telemetry(imu_hz=20, mag_hz=10)``); ``uart.write`` back-pressures the loop to
the link's capacity. USB stays at full rate (``board/serial/code.py``).

Deploy (BLE build): ``circup install adafruit_ble`` once, then copy this file as
``code.py`` plus ``feather_protocol.py``, ``sensors.py``, ``telemetry.py`` and
``status_led.py`` to the CIRCUITPY drive root.
"""

import time

from adafruit_ble import BLERadio
from adafruit_ble.advertising.standard import ProvideServicesAdvertisement
from adafruit_ble.services.nordic import UARTService

from sensors import SensorHub
from status_led import StatusLED
from telemetry import Telemetry


def main():
    ble = BLERadio()
    ble.name = "FeatherSense"
    uart = UARTService()
    advertisement = ProvideServicesAdvertisement(uart)
    hub = SensorHub()
    led = StatusLED(hub)
    # While streaming, the LED rides the battery slot — no call in the hot loop.
    telemetry = Telemetry(
        hub=hub, imu_hz=20, mag_hz=10, on_battery=led.update
    )  # reduced profile for the BLE link

    while True:
        ble.start_advertising(advertisement)
        # `pump` (and so the battery slot driving the LED) doesn't run while
        # advertising — which is exactly when the board is on battery with nobody
        # reading it — so self-drive the LED here. This loop has nothing else to
        # do, so `tick`'s cost is free, and it rate-limits itself.
        while not ble.connected:
            led.tick(time.monotonic())
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
