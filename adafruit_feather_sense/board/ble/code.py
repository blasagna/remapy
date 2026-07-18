"""Feather Sense BLE telemetry (CircuitPython entry point).

Streams the same COBS-framed TLV records as the USB serial build (raw motion +
battery), but over the **Nordic UART Service** (a BLE "serial pipe") instead of
USB. The board
advertises as ``FeatherSense``; a host connects and subscribes to the UART TX
characteristic. The wire protocol is identical, so the host decodes it with the
same ``FrameDecoder`` (see ``ble_stream.py`` / ``read_ble.py``).

This build samples the IMU at a reduced rate (``imu_hz=50`` against USB's 100) —
BLE is the constraint here, not the loop, which sustains 100 Hz on the serial
build from the same code. ``uart.write`` back-pressures the loop to the link's
capacity, so an over-ambitious rate stalls *every* stream rather than erroring:
the failure mode is a rate quietly below nominal, not an exception. Measured, the
link saturates at ~100 Hz IMU (~4.4 KB/s), so 50 Hz runs with ~2x margin.

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
from telemetry import Telemetry, _now_ms


IMU_HZ = 50  # ~2.2 KB/s of frames, measured to stream cleanly; see the module docstring


def main():
    ble = BLERadio()
    ble.name = "FeatherSense"
    uart = UARTService()
    advertisement = ProvideServicesAdvertisement(uart)
    hub = SensorHub(imu_hz=IMU_HZ)
    led = StatusLED(hub)
    # While streaming, the LED rides the battery slot — no call in the hot loop;
    # the charging pulse rides its own slot, which the 0.2 Hz battery slot is far
    # too slow to animate.
    telemetry = Telemetry(
        hub=hub, imu_hz=IMU_HZ, mag_hz=10, on_battery=led.update, on_pulse=led.pulse
    )

    while True:
        ble.start_advertising(advertisement)
        # `pump` (and so the battery + pulse slots driving the LED) doesn't run
        # while advertising — which is exactly when the board is on battery with
        # nobody reading it — so self-drive the LED here. This loop has nothing
        # else to do, so `tick`'s cost is free; it rate-limits the battery read
        # and pulses on every spin.
        while not ble.connected:
            led.tick(time.monotonic())
        ble.stop_advertising()
        # No connection-interval tuning here on purpose: requesting the 7.5 ms
        # minimum measured *identical* to leaving the negotiated default alone
        # (50.0 Hz either way), so it was removed rather than kept as a
        # plausible-looking no-op.
        # Stream until the central disconnects; a write raising means the link
        # dropped mid-frame, so fall back out to re-advertise.
        try:
            while ble.connected:
                next_due_ms = telemetry.pump(_now_ms(), uart.write)
                delay = (next_due_ms - _now_ms()) / 1000
                if delay > 0:
                    time.sleep(delay)
        except Exception:  # noqa: BLE001 - disconnected mid-write -> re-advertise
            pass


main()
