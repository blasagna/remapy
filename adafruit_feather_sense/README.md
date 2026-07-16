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

## Transports

The same COBS-framed TLV protocol runs over two interchangeable transports:

- **USB serial** (default) — frames written to `sys.stdout.buffer`; host reads with `pyserial`.
- **BLE** — frames written to the **Nordic UART Service**; host reads with `bleak`. Lower
  bandwidth (~1–2 KB/s), so the board build samples the IMU slower.

The sampling loop (`telemetry.py`) is transport-agnostic; each board build is a thin `code.py`
that only supplies the emit sink. On the host, both transports expose the **same stream
interface** (`poll()` / `errors` / `close()` / `.port` / `open_if_available`), so the apps select
one with `--feather-transport {serial,ble}` and nothing else changes.

## Files

| File | Runs on | Purpose |
|------|---------|---------|
| [`feather_protocol.py`](feather_protocol.py) | board **+** host | TLV encode/decode, COBS, `FrameDecoder` |
| [`sensors.py`](sensors.py) | board | `SensorHub` — inits all sensors, one read method per stream |
| [`telemetry.py`](telemetry.py) | board | `Telemetry.pump(now, emit)` — the shared sample/schedule/encode loop (rates configurable) |
| [`board/serial/code.py`](board/serial/code.py) | board | USB serial entry: `emit = sys.stdout.buffer.write`, full rates |
| [`board/ble/code.py`](board/ble/code.py) | board | BLE entry: Nordic UART peripheral (`FeatherSense`), `emit = uart.write`, reduced rates |
| [`stream.py`](stream.py) | host | serial `FeatherSenseStream` + shared `FrameRecordDecoder` (bytes→SI records) |
| [`ble_stream.py`](ble_stream.py) | host | BLE `FeatherSenseBLEStream` (bleak on a background thread; same interface) |
| [`read_stream.py`](read_stream.py) | host | serial reader CLI: decode, int→SI (`to_si`), pretty-print / `--stats` / `--raw` |
| [`read_ble.py`](read_ble.py) | host | BLE reader CLI (bleak): `--address` / `--name` / `--stats` / `--only` |
| [`__init__.py`](__init__.py) | host | `open_feather(transport, ...)` — transport-selecting factory used by the apps |

## Sample rates

Rates are `Telemetry(...)` constructor args, so each board build picks its own. Nominal vs.
**measured** on the serial build (12 s capture, board at rest, CircuitPython 10.2.1):

| Stream | Nominal (serial) | Measured | Nominal (BLE) |
|--------|------------------|----------|---------------|
| accel / gravity / linear_accel | 50 Hz (`imu_hz`) | ~41 Hz | 20 Hz |
| gyro | 50 Hz (`imu_hz`) | ~41 Hz | 20 Hz |
| mag | 20 Hz (`mag_hz`) | ~16 Hz | 10 Hz |
| env / altitude | 1 Hz (`env_hz`) | 1.0 Hz | 1 Hz |
| battery | 0.2 Hz (`battery_hz`) | 0.2 Hz | 0.2 Hz |

The IMU/mag streams land ~18–20 % under nominal: all six share one cooperative `pump()` loop, so
each sensor's I2C read and frame encode delays whatever is due next. The slow streams hit their
rates exactly (huge timing slack). **This is jitter in sample *spacing*, not wrong times** —
timestamps are stamped at read, so recorded data stays accurate.

One accelerometer read fans out into **three** frames sharing a timestamp (`accel`, `gravity`,
`linear_accel`), so ~41 Hz of accel sampling is ~123 frames/s on the wire; `gyro` is a separate
read at the same `imu_hz`.

To close the gap: raise `imu_hz` above 50 to compensate for loop overhead (one-line change,
timestamps stay honest), or drop `mag_hz` to free loop time. The BLE build is deliberately slower
(`imu_hz=20, mag_hz=10`) — `uart.write` back-pressures the loop to the ~1–2 KB/s link anyway.

### Why not interrupts?

**CircuitPython has no user-defined interrupt handlers** — no `attachInterrupt`, no MicroPython
`Pin.irq(handler=...)`. It's a firmware-wide design decision (Python callbacks from an ISR are
unsafe with the GC/heap), so there is no Python-level workaround. The interrupt-backed modules
that do exist (`countio`, `keypad`, `rotaryio`, `pulseio`, `alarm.pin.PinAlarm`) service the
interrupt in C and hand back a queue/counter; none apply to an I2C sensor. `alarm.pin.PinAlarm`
is the only true hardware interrupt available, and only for waking from sleep.

Interrupts also wouldn't fix the rate gap above — that's **throughput** (I2C transactions + USB
writes), not the scheduling latency interrupts solve. A data-ready IRQ would save a few hundred µs
of notice but still cost one I2C read per sample.

The real lever would be the IMU's **~8 KB FIFO**: buffer at a hardware-clocked ODR, drain many
samples per burst read. That amortizes per-transaction overhead *and* gives better timestamp
regularity than the loop can (spacing comes from the chip's clock). Catch: the installed
`adafruit_lsm6ds` (4.6.3) exposes **no FIFO support** — the only interrupt-adjacent symbol is
`_route_int1`, used solely for the pedometer step-counter — so it means driving `FIFO_CTRL1–5` /
`FIFO_STATUS` by hand over the existing I2C connection. Unverified: whether the Feather Sense
routes the IMU's INT1 to a GPIO at all (check `dir(board)` at the REPL).

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

Then copy the shared modules plus the **chosen transport's** `code.py` to the CIRCUITPY drive
root (the board auto-reloads and starts streaming):

```bash
# USB serial build (default)
cp feather_protocol.py sensors.py telemetry.py board/serial/code.py /media/bob/CIRCUITPY/ && sync

# BLE build — install the BLE library once, then deploy the ble entry
pixi run circup install adafruit_ble
cp feather_protocol.py sensors.py telemetry.py board/ble/code.py /media/bob/CIRCUITPY/ && sync
```

Only one `code.py` runs at a time; swap builds by re-copying the other transport's `code.py`.

**Eject cleanly — CIRCUITPY corrupts easily.** Unplugging the board without unmounting can leave
the FAT filesystem damaged (CircuitPython drives are unusually prone to this when yanked
mid-write). Unmount before unplugging:

```bash
udisksctl unmount -b /dev/sdb1     # confirm the device first: mount | grep -i circuitpy
```

Symptoms of a corrupt drive: files that `ls` fine but return `Input/output error` on read, and a
drive that mounts read-only (or flips to read-only seconds after a write). Confirm in the kernel
log:

```bash
journalctl -k | grep -i fat-fs
# FAT-fs (sdb1): Volume was not properly unmounted. Some data may be corrupt. Please run fsck.
# FAT-fs (sdb1): error, fat_get_cluster: invalid cluster chain (i_pos 125)
# FAT-fs (sdb1): Filesystem has been set read-only
```

Repair it — **replugging does not fix this**; only fsck rewrites the FAT (a replug clears the
read-only mount, which looks healthy but the bad cluster chain is still on flash):

```bash
udisksctl unmount -b /dev/sdb1 && sudo fsck.vfat -a -v /dev/sdb1
```

Then remount and verify: a clean mount logs **no** `Volume was not properly unmounted` warning
(that message is the FAT dirty bit, which fsck clears — its absence is how you know the repair
took), and a `touch` on the drive succeeds with no new cluster errors. fsck truncates or deletes
the corrupt file, so re-copy the build afterwards; `lib/` and the circup-installed drivers
normally survive untouched. Any edit made *directly on the board* that isn't mirrored in the repo
is unrecoverable. Last resort if fsck can't repair it: `storage.erase_filesystem()` at the REPL
wipes CIRCUITPY clean (erases `lib/` — re-run the `circup install` above).

## Read the stream (host)

USB serial (`pyserial`):

```bash
pixi run python adafruit_feather_sense/read_stream.py            # auto-detect port, decode
pixi run python adafruit_feather_sense/read_stream.py --stats    # per-stream rates once/sec
pixi run python adafruit_feather_sense/read_stream.py --raw      # hexdump bytes (inspect framing)
pixi run python adafruit_feather_sense/read_stream.py --only accel,battery
pixi run python adafruit_feather_sense/read_stream.py --port /dev/ttyACM0
```

BLE (`bleak`; PC Bluetooth on, board running the BLE build):

```bash
pixi run python adafruit_feather_sense/read_ble.py              # scan for "FeatherSense", decode
pixi run python adafruit_feather_sense/read_ble.py --stats
pixi run python adafruit_feather_sense/read_ble.py --address AA:BB:CC:DD:EE:FF
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

## Host library & app integration

Both transports present the **same interface** for consuming the stream from an existing loop
without blocking:

- `poll()` returns the `SensorRecord`s (SI-converted; `error` records carry the source *name*)
  that arrived since the last call — pump it once per iteration.
- `open_if_available(...)` probes for a real frame and returns the stream, or `None` when the
  device is absent — so callers run with or without the board.
- `FrameRecordDecoder` (in [`stream.py`](stream.py)) is the shared bytes→records decode used by
  both `FeatherSenseStream` (serial) and `FeatherSenseBLEStream` (BLE).

`open_feather(transport, *, port=None, address=None)` in [`__init__.py`](__init__.py) picks the
backend (lazy-importing bleak/pyserial). The **rerun viewer** and **HDF5 recorder** apps use it —
flags `--feather` / `--no-feather` / `--feather-transport {serial,ble}` / `--feather-port` (serial)
/ `--feather-address` (BLE):

```bash
pixi run rerun   --feather                         # camera + pose + a "Feather Sense" plots tab (serial)
pixi run record  --feather -o session.h5           # sensor streams saved under /feather/<name> (serial)
pixi run record  --feather --feather-transport ble -o session.h5   # same, over BLE
```

Recorded data reads back via `recording.reader.Recording.feather` (a `{stream: arrays}` dict),
identical regardless of transport.

## Notes / gotchas

- CircuitPython's `struct` module exposes only the `pack`/`unpack` **functions**,
  not the `struct.Struct` **class** — `feather_protocol.py` uses the functions so
  it works on both runtimes.
- Opening the serial port while the board is idling at the REPL requires a
  Ctrl-D (`0x04`) to soft-reload `code.py`; once it's looping it streams
  continuously. The host reader tolerates the leading REPL banner text (it
  decodes as one dropped frame, then resyncs).
- Piping `read_stream.py` into another command (`| head`, `| grep`) can show
  **nothing** — Python block-buffers stdout when it isn't a TTY, and `--stats`
  repaints a live line that a pipe never sees. Looks like a dead board; isn't.
  Use `python -u` when piping.

