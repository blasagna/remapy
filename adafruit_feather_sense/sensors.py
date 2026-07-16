"""Onboard sensor access for the Adafruit Feather Bluefruit Sense (nRF52840).

`SensorHub` initialises the motion sensors on the board's internal I2C bus and
exposes one read method per data stream. It intentionally holds *no*
protocol/transport knowledge, so the same hub backs the USB serial stream and
the BLE transport.

Only **raw** signals are read here — acceleration, angular rate, magnetic field
and battery state. Two deliberate omissions:

- *Derived motion* (the gravity vector and linear acceleration) is computed on
  the host (see ``motion.py``), so the loop budget goes to sampling rather than
  filtering, and the filter's time constant stays a read-time choice.
- *Environmental sensing* (BMP280 temperature/pressure/altitude, SHT31-D
  humidity) was removed: those chips only answer a forced-mode conversion, which
  blocked the loop for ~150 ms of every second — ~15 % of the wall clock, and
  ~7.6 lost IMU samples per second — to serve 1 Hz data this project does not
  use. Re-adding them means putting the BMP280 in ``MODE_NORMAL`` first.

Runs only on the board (imports `board`, `busio`, `analogio`, `supervisor`).
"""

import analogio
import board
import busio
import supervisor

import adafruit_lis3mdl

# The Feather Sense shipped with the LSM6DS33 originally and the pin-compatible
# LSM6DS3TR-C from Jan 2024 onward. Try the former, fall back to the latter.
from adafruit_lsm6ds.lsm6ds33 import LSM6DS33
from adafruit_lsm6ds.lsm6ds3trc import LSM6DS3TRC

# LiPo voltage endpoints for a crude state-of-charge estimate.
_BATT_EMPTY_V = 3.2
_BATT_FULL_V = 4.2


class SensorHub:
    """Owns the I2C bus and every sensor; provides per-stream reads."""

    def __init__(self):
        # busio.I2C defaults to 100 kHz; both chips here run fast mode. Measured
        # ~1.8x on every read (accel 1.42 -> 0.77 ms, gyro 1.57 -> 0.88 ms, mag
        # 3.61 -> 1.95 ms), which is the cheapest throughput win available.
        self.i2c = busio.I2C(board.SCL, board.SDA, frequency=400_000)

        # IMU: accelerometer + gyroscope (same chip, two data streams).
        try:
            self.imu = LSM6DS33(self.i2c)
            self.imu_model = "LSM6DS33"
        except (ValueError, RuntimeError, OSError):
            self.imu = LSM6DS3TRC(self.i2c)
            self.imu_model = "LSM6DS3TRC"

        self.magnetometer = adafruit_lis3mdl.LIS3MDL(self.i2c)

        self._battery = analogio.AnalogIn(board.VOLTAGE_MONITOR)

    # --- IMU -----------------------------------------------------------------
    def read_accel(self):
        """(x, y, z) total acceleration in m/s^2.

        Raw accelerometer output: this **includes gravity** (~9.8 m/s^2 on one
        axis at rest), since the LSM6DS does no on-chip fusion. The host splits
        it into gravity + linear components (see ``motion.py``).
        """
        return self.imu.acceleration

    def read_gyro(self):
        """(x, y, z) angular rate in rad/s."""
        return self.imu.gyro

    # --- Magnetometer --------------------------------------------------------
    def read_mag(self):
        """(x, y, z) magnetic field in microtesla."""
        return self.magnetometer.magnetic

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
