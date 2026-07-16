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

import struct
from math import radians

import analogio
import board
import busio
import supervisor

import adafruit_lis3mdl

from adafruit_lsm6ds import AccelRange, GyroRange, Rate

# The Feather Sense shipped with the LSM6DS33 originally and the pin-compatible
# LSM6DS3TR-C from Jan 2024 onward. Try the former, fall back to the latter.
from adafruit_lsm6ds.lsm6ds33 import LSM6DS33
from adafruit_lsm6ds.lsm6ds3trc import LSM6DS3TRC

# LiPo voltage endpoints for a crude state-of-charge estimate.
_BATT_EMPTY_V = 3.2
_BATT_FULL_V = 4.2

# Gyro X output register. The LSM6DS lays out gyro (0x22) then accelerometer
# (0x28) as six contiguous little-endian int16s, so one 12-byte burst read gets
# both. Datasheet layout, not driver internals — the driver reads the same two
# addresses as separate `Struct(_LSM6DS_OUTX_L_G, "<hhh")` / `(_LSM6DS_OUTX_L_A,
# "<hhh")` descriptors.
_OUTX_L_G = 0x22
_IMU_BURST_FMT = "<hhhhhh"  # gx, gy, gz, ax, ay, az

# LSB -> SI, mirroring the driver's own `_scale_xl_data` / `_scale_gyro_data`.
_MILLI_G_TO_ACCEL = 0.00980665

# Selectable output data rates, ascending. The IMU must be clocked well above the
# poll rate or the loop re-reads samples the chip hasn't refreshed; see
# `_odr_for`.
_ODR_CHOICES = (
    (52.0, Rate.RATE_52_HZ),
    (104.0, Rate.RATE_104_HZ),
    (208.0, Rate.RATE_208_HZ),
    (416.0, Rate.RATE_416_HZ),
    (833.0, Rate.RATE_833_HZ),
)


def _odr_for(imu_hz):
    """Return (rate_const, rate_hz): the slowest ODR that oversamples `imu_hz` 2x.

    2x is the ratio the 50 Hz build ran at on the driver's 104 Hz default, and it
    is the floor for the poll to see a fresh sample every time.

    Deliberately not *more* than needed: a higher ODR widens the analog
    bandwidth, buying noise for no extra information at a fixed poll rate.
    Measured at rest, 104 -> 208 cost **1.42x** RMS on the gyro (the textbook
    sqrt(2)) but only **1.05x** on the accel, whose at-rest noise is dominated by
    ambient vibration rather than bandwidth. So the cost is real but per-stream —
    don't assume sqrt(2) everywhere, and don't reach for 416 "for headroom".
    """
    target = 2.0 * imu_hz
    for rate_hz, rate in _ODR_CHOICES:
        if rate_hz >= target:
            return rate, rate_hz
    return _ODR_CHOICES[-1][1], _ODR_CHOICES[-1][0]


class SensorHub:
    """Owns the I2C bus and every sensor; provides per-stream reads."""

    def __init__(self, imu_hz=50):
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

        # The driver defaults both axes to 104 Hz, which is only 1.04x a 100 Hz
        # poll — close enough that the loop would read stale samples. Pin it to
        # the poll rate instead of inheriting a default that silently fits one
        # rate and not another.
        odr, self.imu_odr_hz = _odr_for(imu_hz)
        self.imu.accelerometer_data_rate = odr
        self.imu.gyro_data_rate = odr

        # Fixed at init from the *public* range properties, so `read_imu` is a
        # burst + multiply with no per-read driver calls. Mirrors the driver's
        # own scale math; `_bdu` (which the driver sets) is what makes a
        # multi-byte burst read internally consistent.
        self._accel_scale = AccelRange.lsb[self.imu.accelerometer_range] * _MILLI_G_TO_ACCEL
        self._gyro_scale = radians(GyroRange.lsb[self.imu.gyro_range] / 1000)
        self._imu_reg = bytes((_OUTX_L_G,))
        self._imu_buf = bytearray(12)

        self.magnetometer = adafruit_lis3mdl.LIS3MDL(self.i2c)

        self._battery = analogio.AnalogIn(board.VOLTAGE_MONITOR)

    # --- IMU -----------------------------------------------------------------
    def read_imu(self):
        """((ax, ay, az), (gx, gy, gz)) — acceleration m/s^2, angular rate rad/s.

        Both streams from **one** 12-byte burst, replacing the driver's separate
        `.acceleration` / `.gyro` properties (two I2C transactions reading two
        halves of the same contiguous block). Two reasons, and the second is the
        real one:

        - It halves the IMU's I2C cost per cycle.
        - It makes accel and gyro **one sample, at one instant**, sharing a
          timestamp. Read separately they are ~1 ms apart and independently
          stamped, which is a lie about simultaneity that any downstream sensor
          fusion would quietly inherit.

        Acceleration is raw and **includes gravity** (~9.8 m/s^2 on one axis at
        rest) — the LSM6DS does no on-chip fusion, and the host splits it into
        gravity + linear components (see ``motion.py``).
        """
        with self.imu.i2c_device as i2c:
            i2c.write_then_readinto(self._imu_reg, self._imu_buf)
        gx, gy, gz, ax, ay, az = struct.unpack_from(_IMU_BURST_FMT, self._imu_buf)
        a_s = self._accel_scale
        g_s = self._gyro_scale
        return (ax * a_s, ay * a_s, az * a_s), (gx * g_s, gy * g_s, gz * g_s)

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
