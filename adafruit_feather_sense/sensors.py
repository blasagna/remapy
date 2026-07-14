"""Onboard sensor access for the Adafruit Feather Bluefruit Sense (nRF52840).

`SensorHub` initialises every sensor on the board's internal I2C bus and exposes
one read method per data stream. It intentionally holds *no* protocol/transport
knowledge, so the same hub can back the USB serial stream today and a BLE
transport later.

Runs only on the board (imports `board`, `busio`, `analogio`, `supervisor`).
"""

import time

import analogio
import board
import busio
import supervisor

import adafruit_bmp280
import adafruit_lis3mdl
import adafruit_sht31d

# The Feather Sense shipped with the LSM6DS33 originally and the pin-compatible
# LSM6DS3TR-C from Jan 2024 onward. Try the former, fall back to the latter.
from adafruit_lsm6ds.lsm6ds33 import LSM6DS33
from adafruit_lsm6ds.lsm6ds3trc import LSM6DS3TRC

# LiPo voltage endpoints for a crude state-of-charge estimate.
_BATT_EMPTY_V = 3.2
_BATT_FULL_V = 4.2

# Time constant (seconds) of the low-pass filter that estimates the gravity
# vector from raw acceleration. Larger = steadier gravity, slower to follow a
# reorientation; smaller = follows tilt faster but leaks more motion into it.
_GRAVITY_TAU_S = 0.5


class SensorHub:
    """Owns the I2C bus and every sensor; provides per-stream reads."""

    def __init__(self):
        self.i2c = busio.I2C(board.SCL, board.SDA)

        # IMU: accelerometer + gyroscope (same chip, two data streams).
        try:
            self.imu = LSM6DS33(self.i2c)
            self.imu_model = "LSM6DS33"
        except (ValueError, RuntimeError, OSError):
            self.imu = LSM6DS3TRC(self.i2c)
            self.imu_model = "LSM6DS3TRC"

        self.magnetometer = adafruit_lis3mdl.LIS3MDL(self.i2c)
        self.humidity = adafruit_sht31d.SHT31D(self.i2c)
        self.barometer = adafruit_bmp280.Adafruit_BMP280_I2C(self.i2c)
        # Sea-level pressure used to derive altitude; adjust to local QNH for
        # an accurate absolute altitude.
        self.barometer.sea_level_pressure = 1013.25

        self._battery = analogio.AnalogIn(board.VOLTAGE_MONITOR)

        # Gravity-vector estimate state (see read_motion / _GRAVITY_TAU_S).
        self._gravity = None
        self._gravity_t = None

    # --- IMU -----------------------------------------------------------------
    def read_accel(self):
        """(x, y, z) total acceleration in m/s^2 (includes gravity)."""
        return self.imu.acceleration

    def read_motion(self):
        """Read the accelerometer once and decompose it.

        Returns ``(total, gravity, linear)``, each an (x, y, z) tuple in m/s^2:

        - ``total``   — the raw accelerometer output (includes gravity),
        - ``gravity`` — a low-pass estimate of the gravity vector,
        - ``linear``  — ``total - gravity`` (acceleration from actual motion).

        The LSM6DS does no on-chip fusion, so gravity is estimated here with a
        single-pole low-pass filter whose coefficient adapts to the real elapsed
        time between calls (robust to the variable loop rate). This isolates
        gravity well for tilt and brief motion; sustained linear acceleration
        will slowly bleed into the gravity estimate (a gyro-fused orientation
        filter would remove that, at more cost).
        """
        total = self.imu.acceleration
        now = time.monotonic()
        if self._gravity is None:
            self._gravity = list(total)
        else:
            dt = now - self._gravity_t
            alpha = dt / (_GRAVITY_TAU_S + dt) if dt > 0 else 0.0
            self._gravity = [g + alpha * (t - g) for g, t in zip(self._gravity, total)]
        self._gravity_t = now
        gravity = tuple(self._gravity)
        linear = tuple(t - g for t, g in zip(total, gravity))
        return total, gravity, linear

    def read_gyro(self):
        """(x, y, z) angular rate in rad/s."""
        return self.imu.gyro

    # --- Magnetometer --------------------------------------------------------
    def read_mag(self):
        """(x, y, z) magnetic field in microtesla."""
        return self.magnetometer.magnetic

    # --- Environment ---------------------------------------------------------
    def read_env(self):
        """(temperature_C, relative_humidity_%, pressure_hPa).

        Temperature is taken from the BMP280 (co-located with the pressure
        reading); the SHT31-D supplies humidity.
        """
        return (
            self.barometer.temperature,
            self.humidity.relative_humidity,
            self.barometer.pressure,
        )

    def read_altitude(self):
        """Barometric altitude in metres (relative to `sea_level_pressure`)."""
        return self.barometer.altitude

    # --- Power ---------------------------------------------------------------
    def read_battery(self):
        """(voltage_V, percent, usb_connected) for the LiPo / USB power state.

        `board.VOLTAGE_MONITOR` sits behind a 2:1 divider, so the true battery
        voltage is twice the measured pin voltage. `percent` is a rough linear
        LiPo estimate; `usb_connected` reflects whether the board sees USB power.
        """
        raw = self._battery.value  # 0..65535
        pin_v = raw / 65535 * self._battery.reference_voltage
        voltage = pin_v * 2
        percent = (voltage - _BATT_EMPTY_V) / (_BATT_FULL_V - _BATT_EMPTY_V) * 100
        percent = max(0.0, min(100.0, percent))
        usb_connected = 1 if supervisor.runtime.usb_connected else 0
        return voltage, percent, usb_connected
