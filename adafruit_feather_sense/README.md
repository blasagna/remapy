# Adafruit Feather Sense — sensor telemetry

A CircuitPython application that samples every onboard sensor of the
**Adafruit Feather Bluefruit Sense (nRF52840)** and streams the readings over
USB serial, plus a host-side `pyserial` reader. Transport starts as USB serial;
the protocol and sensor layers are transport-agnostic so a BLE transport can be
added later without touching them.

- Board guide: https://learn.adafruit.com/adafruit-feather-sense/
- Board: `feather_bluefruit_sense`, CircuitPython **10.2.1**, mounted at `/media/bob/CIRCUITPY`.

## Sensors

All sensors sit on the board's internal I2C bus (`board.SCL`/`board.SDA`).

| Stream | Chip | Library | Units |
|--------|------|---------|-------|
| Acceleration | LSM6DS33 **or** LSM6DS3TR-C | `adafruit_lsm6ds` | m/s² |
| Angular rate | (same IMU) | `adafruit_lsm6ds` | rad/s |
| Magnetic field | LIS3MDL | `adafruit_lis3mdl` | µT |
| Temperature / Pressure / Altitude | BMP280 | `adafruit_bmp280` | °C / hPa / m |
| Humidity | SHT31-D | `adafruit_sht31d` | %RH |
| Battery / power state | `board.VOLTAGE_MONITOR`, `supervisor` | (stdlib) | V / % / bool |

**IMU chip note:** Adafruit swapped the LSM6DS33 for the pin-compatible
LSM6DS3TR-C in Jan 2024. `SensorHub` tries `LSM6DS33` first and falls back to
`LSM6DS3TRC`, so either board revision works. Battery voltage is read through a
2:1 divider on `VOLTAGE_MONITOR` (true voltage = 2 × pin voltage); `percent` is a
crude linear LiPo estimate (3.2 V → 0 %, 4.2 V → 100 %).

*Not sampled:* the APDS9960 (light/color/gesture/proximity) and the PDM
microphone. Both are easy to add later as new message types.

## Wire protocol — TLV over COBS

Each sensor sample is emitted as **one independent frame**, so streams can be
sampled at different rates without bundling. Defined once in
[`feather_protocol.py`](feather_protocol.py) (pure `struct`, runs on both
CircuitPython and CPython — the board and the host import the same file).

```
frame   = cobs_encode(payload) + 0x00
payload = [ type:u8 ][ length:u8 ][ value ]        # Type-Length-Value
value   = [ timestamp_ms:u32 ][ data: N × int32 ]   # little-endian
```

**No floating point on the wire.** Sensor values are transmitted as scaled
fixed-point `int32`; the physical (SI) value is `raw_int / scale`. The device
converts SI → int with `feather_protocol.to_raw()`, and the host converts back
int → SI in `read_stream.py` (`to_si()`), both using the one shared `SCALES`
table. This keeps frames compact and the numeric format explicit.

COBS (Consistent Overhead Byte Stuffing) removes every `0x00` from the payload,
so the single trailing `0x00` is an unambiguous frame delimiter. The host splits
on `0x00` and decodes each chunk; a partial/corrupt frame is dropped and the
next delimiter resyncs the stream (self-synchronising framing).

| Type | Name | Payload after timestamp | Scale (SI = raw / scale) |
|------|------|-------------------------|--------------------------|
| `0x01` | accel | 3×i32 (x, y, z) — **total, includes gravity** | 1000 → m/s² |
| `0x02` | gyro | 3×i32 (x, y, z) | 10000 → rad/s |
| `0x03` | mag | 3×i32 (x, y, z) | 100 → µT |
| `0x04` | env | 3×i32 (temp, humidity, pressure) | 100 → °C, %RH, hPa |
| `0x05` | altitude | 1×i32 (altitude) | 1000 → m |
| `0x06` | battery | 2×i32 (voltage, percent) + 1×u8 (usb) | 1000 → V, 100 → % |
| `0x07` | error | u8 source-type + UTF-8 text | — (not scaled) |
| `0x08` | gravity | 3×i32 (x, y, z) — estimated gravity vector | 1000 → m/s² |
| `0x09` | linear_accel | 3×i32 (x, y, z) — gravity removed | 1000 → m/s² |

`timestamp_ms` = `time.monotonic_ns() // 1_000_000` (u32, wraps ≈ 49.7 days).
No CRC — COBS framing + reliable USB CDC is sufficient; a CRC could be added as a
future extension. Adding a new message type never breaks existing parsers.

**Error stream (`0x07`):** each sensor read+encode is wrapped in the device
loop; on any exception it emits an `error` frame carrying the *source* stream's
type byte and the exception text (`source=mag  OSError: ...`) instead of silently
dropping the sample. Other streams keep flowing. The reader prints these inline
and counts them under `--stats`.

**Acceleration decomposition (`accel` / `gravity` / `linear_accel`):** the
LSM6DS is a raw IMU with no on-chip fusion, so it only reports *total*
acceleration (`accel`, which includes gravity — ~9.8 m/s² on one axis at rest).
`SensorHub.read_motion()` does one accelerometer read per sample and derives two
more streams from it: `gravity`, a single-pole low-pass estimate of the gravity
vector (time constant `_GRAVITY_TAU_S`, filter coefficient adapts to the real
loop `dt`), and `linear_accel = accel − gravity`, the motion-only component
(≈ 0 at rest). All three share one timestamp. This is a lightweight estimate:
sustained (non-transient) linear acceleration slowly bleeds into the gravity
estimate; a gyro-fused orientation filter (e.g. Madgwick/Mahony) would remove
that at more compute cost — a possible future improvement.

## Files

| File | Runs on | Purpose |
|------|---------|---------|
| [`feather_protocol.py`](feather_protocol.py) | board **+** host | TLV encode/decode, COBS, `FrameDecoder` |
| [`sensors.py`](sensors.py) | board | `SensorHub` — inits all sensors, one read method per stream |
| [`code.py`](code.py) | board | entry point: per-stream rate scheduler → COBS frames over USB serial |
| [`read_stream.py`](read_stream.py) | host | `pyserial` reader: decode, int→SI transform (`to_si`), pretty-print / `--stats` / `--raw` |

Default sample rates (in `code.py`): accel 50 Hz (emits `accel` + `gravity` +
`linear_accel` together), gyro 50 Hz, mag 20 Hz, env 1 Hz, altitude 1 Hz,
battery 0.2 Hz. (Observed throughput is a bit lower — ~32/32/13 Hz — because the
streams share one loop; adjust intervals there.)

## Deploy

Install the sensor drivers to the board with `circup` (from the repo root):

```bash
pixi run circup install adafruit_lsm6ds adafruit_lis3mdl adafruit_sht31d adafruit_bmp280
```

`adafruit_bus_device` and `adafruit_register` are pulled in automatically.
Resulting `pixi run circup freeze` on the board:

```
adafruit_bmp280==3.3.12
adafruit_lis3mdl==1.2.8
adafruit_sht31d==2.3.31
adafruit_bus_device==5.2.17
adafruit_lsm6ds==4.6.3
adafruit_register==1.11.3
```

Then copy the three device files to the CIRCUITPY drive root (the board
auto-reloads and starts streaming):

```bash
cp feather_protocol.py sensors.py code.py /media/bob/CIRCUITPY/ && sync
```

## Read the stream (host)

```bash
pixi run python adafruit_feather_sense/read_stream.py            # auto-detect port, decode
pixi run python adafruit_feather_sense/read_stream.py --stats    # per-stream rates once/sec
pixi run python adafruit_feather_sense/read_stream.py --raw      # hexdump bytes (inspect framing)
pixi run python adafruit_feather_sense/read_stream.py --only accel,battery
pixi run python adafruit_feather_sense/read_stream.py --port /dev/ttyACM0
```

Example decoded output (board at rest, on USB power):

```
accel     x=0.024  y=0.564  z=9.870      # z ≈ g, board flat
gyro      x=0.060  y=-0.053 z=-0.056     # ≈ 0, stationary
mag       x=-35.94 y=25.17  z=-28.97
env       temperature_c=26.36  humidity_pct=43.83  pressure_hpa=1004.34
altitude  altitude_m=74.41
battery   voltage_v=4.103  percent=90.27  usb_connected=1
```

## Notes / gotchas

- CircuitPython's `struct` module exposes only the `pack`/`unpack` **functions**,
  not the `struct.Struct` **class** — `feather_protocol.py` uses the functions so
  it works on both runtimes.
- Opening the serial port while the board is idling at the REPL requires a
  Ctrl-D (`0x04`) to soft-reload `code.py`; once it's looping it streams
  continuously. The host reader tolerates the leading REPL banner text (it
  decodes as one dropped frame, then resyncs).

## Future: BLE

`SensorHub` and `feather_protocol.encode` are transport-free, so a future
`ble_stream.py` can reuse both and only swap the sink (e.g. a Nordic UART or the
Adafruit BLE sensor service — `adafruit_ble` / `adafruit_ble_adafruit` are in the
bundle) for the `sys.stdout.buffer.write` call in `code.py`.
