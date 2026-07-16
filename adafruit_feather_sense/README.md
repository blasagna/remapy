# Adafruit Feather Sense — sensor telemetry

A CircuitPython application that samples the **raw motion sensors** of the
**Adafruit Feather Bluefruit Sense (nRF52840)** and streams the readings over USB
serial or BLE, plus host-side readers. The protocol and sensor layers are
transport-agnostic; both transports are interchangeable.

The board sends **only what it measures** — acceleration, angular rate, magnetic
field, battery. Anything derivable from those (gravity, linear acceleration) is
reconstructed on the host, so the device's loop budget goes to sampling. See
[Derived motion](#derived-motion-gravity--linear_accel).

- Board guide: https://learn.adafruit.com/adafruit-feather-sense/
- Board: `feather_bluefruit_sense`, CircuitPython **10.2.1**, mounted at `/media/bob/CIRCUITPY`.

## Sensors

All sensors sit on the board's internal I2C bus (`board.SCL`/`board.SDA`).

| Stream | Chip | Library | Units |
|--------|------|---------|-------|
| Acceleration | LSM6DS33 **or** LSM6DS3TR-C | `adafruit_lsm6ds` | m/s² |
| Angular rate | (same IMU) | `adafruit_lsm6ds` | rad/s |
| Magnetic field | LIS3MDL | `adafruit_lis3mdl` | µT |
| Battery / power state | `board.VOLTAGE_MONITOR`, `supervisor` | (stdlib) | V / % / bool |

**IMU chip note:** Adafruit swapped the LSM6DS33 for the pin-compatible
LSM6DS3TR-C in Jan 2024. `SensorHub` tries `LSM6DS33` first and falls back to
`LSM6DS3TRC`, so either board revision works. Battery voltage is read through a
2:1 divider on `VOLTAGE_MONITOR` (true voltage = 2 × pin voltage); `percent` is a
crude linear LiPo estimate (3.2 V → 0 %, 4.2 V → 100 %).

*Not sampled:* the APDS9960 (light/color/gesture/proximity) and the PDM
microphone. Both are easy to add later as new message types.

**Environmental sensing was removed** (BMP280 temperature/pressure/altitude,
SHT31-D humidity). Those chips only answer a *forced-mode conversion* — a
blocking ~45 ms per BMP280 read — and `env` + `altitude` together stalled the
loop for **~152 ms of every second** (~15 % of the wall clock, ~7.6 lost IMU
samples/s) to serve 1 Hz data nothing downstream consumed. Removing them took the
IMU from 41 Hz to **48.3 Hz** (measured). If you re-add them: put the BMP280 in
`MODE_NORMAL` (it then converts continuously in the background and a read becomes
a ~1 ms register fetch), lower `overscan_pressure` from the `X16` default, and
derive altitude from the `env` pressure on the host rather than paying a second
conversion for it.

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
| `0x04` | battery | 2×i32 (voltage, percent) + 1×u8 (usb) | 1000 → V, 100 → % |
| `0x05` | error | u8 source-type + UTF-8 text | — (not scaled) |
| `0x06`† | gravity | 3×float (x, y, z) — **host-derived, never on the wire** | — (built in SI) |
| `0x07`† | linear_accel | 3×float (x, y, z) — **host-derived, never on the wire** | — (built in SI) |

† `0x06`/`0x07` are *pseudo-types*: no board ever transmits them. They exist so
host-derived samples (see [`motion.py`](motion.py)) flow through the same
`SensorRecord`/naming machinery as decoded ones, and so have no `SCALES` entry.

**Codes carry no compatibility guarantee.** They are assigned densely, and the
board and host ship from this one file — so a renumbering is resolved by
reflashing, and the retired env/altitude types were deleted rather than reserved.
Recordings are unaffected either way: `/feather` groups are keyed by stream
*name*, not by type code. If you add a stream, append the next free code.

`timestamp_ms` = `time.monotonic_ns() // 1_000_000` (u32, wraps ≈ 49.7 days).
No CRC — COBS framing + reliable USB CDC is sufficient; a CRC could be added as a
future extension. Adding a new message type never breaks existing parsers.

**Error stream (`0x07`):** each sensor read+encode is wrapped in the device
loop; on any exception it emits an `error` frame carrying the *source* stream's
type byte and the exception text (`source=mag  OSError: ...`) instead of silently
dropping the sample. Other streams keep flowing. The reader prints these inline
and counts them under `--stats`.

## Derived motion (`gravity` / `linear_accel`)

The LSM6DS is a raw IMU with no on-chip fusion, so it only reports *total*
acceleration (`accel`, which includes gravity — ~9.8 m/s² on one axis at rest).
The two useful decompositions are derived **on the host** by
[`motion.py`](motion.py) from that one raw stream:

- `gravity` — a single-pole low-pass estimate of the gravity vector
  (`GRAVITY_TAU_S`; the filter coefficient adapts to the real sample `dt`),
- `linear_accel` — `accel − gravity`, the motion-only component (≈ 0 at rest).

`GravityFilter.update(timestamp_ms, xyz)` drives the live path (the rerun
viewer); `derive_motion(timestamps, values, tau_s)` does a whole recorded stream
at once. The filter is a sequential IIR, so it is a loop by construction and does
not vectorise.

**Why the host and not the board?** It used to run in `SensorHub.read_motion()`,
which made one accelerometer read cost *three* encoded frames (`accel` +
`gravity` + `linear_accel`) — and `encode` is the single most expensive step in
the loop at ~1.275 ms/frame. Moving it off the board removed two of every four
frames per cycle. It also buys accuracy and flexibility:

- **Timing.** The filter now keys off the *device* timestamp rather than the
  board's `time.monotonic()` at read, so it sees the true sample spacing.
- **Re-tunable.** `tau_s` becomes a read-time choice. `Recording.motion(tau_s=…)`
  re-derives an existing recording with a different constant — impossible once a
  filter is baked into the capture. This is the same rule the recorder follows
  for joint angles: store the minimal raw signal, recompute the rest on read.

Accuracy caveat (unchanged): this is a lightweight estimate. Sustained
(non-transient) linear acceleration slowly bleeds into the gravity estimate; a
gyro-fused orientation filter (e.g. Madgwick/Mahony) would remove that at more
compute cost — cheaper to reconsider now that it runs on a host CPU rather than
an nRF52840.

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
| [`sensors.py`](sensors.py) | board | `SensorHub` — inits the raw sensors, one read method per stream |
| [`telemetry.py`](telemetry.py) | board | `Telemetry.pump(now, emit)` — the shared sample/schedule/encode loop (rates configurable) |
| [`motion.py`](motion.py) | host | `GravityFilter` / `derive_motion` — gravity + linear_accel from raw accel |
| [`board/serial/code.py`](board/serial/code.py) | board | USB serial entry: `emit = sys.stdout.buffer.write`, full rates |
| [`board/ble/code.py`](board/ble/code.py) | board | BLE entry: Nordic UART peripheral (`FeatherSense`), `emit = uart.write`, reduced rates |
| [`stream.py`](stream.py) | host | serial `FeatherSenseStream` + shared `FrameRecordDecoder` (bytes→SI records) |
| [`ble_stream.py`](ble_stream.py) | host | BLE `FeatherSenseBLEStream` (bleak on a background thread; same interface) |
| [`read_stream.py`](read_stream.py) | host | serial reader CLI: decode, int→SI (`to_si`), pretty-print / `--stats` / `--raw` |
| [`read_ble.py`](read_ble.py) | host | BLE reader CLI (bleak): `--address` / `--name` / `--stats` / `--only` |
| [`__init__.py`](__init__.py) | host | `open_feather(transport, ...)` — transport-selecting factory used by the apps |

## Sample rates

Rates are `Telemetry(...)` constructor args, so each board build picks its own. Nominal vs.
**measured** on the serial build (15 s capture, board at rest, CircuitPython 10.2.1):

| Stream | Nominal (serial) | Measured | % of nominal | Nominal (BLE) |
|--------|------------------|----------|--------------|---------------|
| accel | 50 Hz (`imu_hz`) | **48.8 Hz** | 97.5 % | 20 Hz |
| gyro | 50 Hz (`imu_hz`) | **48.8 Hz** | 97.5 % | 20 Hz |
| mag | 20 Hz (`mag_hz`) | **19.8 Hz** | 98.8 % | 10 Hz |
| battery | 0.2 Hz (`battery_hz`) | 0.2 Hz | 100 % | 0.2 Hz |

How it got there — each step measured on the board:

| Change | accel | mag |
|--------|-------|-----|
| original (env/altitude on board, gravity+linear encoded on board, 100 kHz I2C) | 41.0 Hz | 16.0 Hz |
| + environmental sensing removed | 48.3 Hz | 19.2 Hz |
| + I2C at 400 kHz | **48.8 Hz** | **19.8 Hz** |

The residual ~2.5 % is the cooperative `pump()` loop: all four streams share it, so each sensor's
I2C read and frame encode delays whatever is due next. **This is jitter in sample *spacing*, not
wrong times** — timestamps are stamped at read, so recorded data stays accurate.

The original 41 Hz was **not** loop saturation (it still slept 17 % of the time) — it was
*blackout*. The 1 Hz `env`/`altitude` reads blocked for ~152 ms/s, and 152 ms of dead time removes
~7.6 of the 50 accel slots per second: `848 ms / 20 ms = 42.4 Hz`, against 41 measured.

**Why 400 kHz bought only +0.5 Hz.** It halves read time, but by then the loop was no longer the
constraint — the rate is capped by `imu_hz`, and the saved time became *sleep*, not throughput.
Its real value is **headroom**: the loop went from 17 % idle (original) to **58 % idle**, with
`pump()` busy only ~4.9 s of every 12. That is the budget to raise `imu_hz`, add streams, or
absorb a slower transport — not a bigger number on this table.

To go past 50 Hz you must raise `imu_hz` itself (there is now room); it is capped by the schedule,
not the hardware. The BLE build stays deliberately slower (`imu_hz=20, mag_hz=10`) — `uart.write`
back-pressures the loop to the ~1–2 KB/s link regardless.

### Where the loop time goes

Measured per phase on the board — useful before optimising the wrong thing:

| Phase | Cost | Note |
|-------|------|------|
| `encode` (to_raw + pack + COBS) | **1.275 ms/frame** | the most expensive step |
| ↳ `cobs_encode` (20 B) | 0.708 ms | 56 % of encode — a pure-Python byte loop |
| ↳ `to_raw` | 0.397 ms | 31 % of encode |
| `imu.acceleration` | 1.420 ms @ 100 kHz | 0.772 ms @ 400 kHz (**1.84×**) |
| `imu.gyro` | 1.566 ms @ 100 kHz | 0.879 ms @ 400 kHz (**1.78×**) |
| `mag.magnetic` | 3.608 ms @ 100 kHz | 1.950 ms @ 400 kHz (**1.85×**) |
| `write(1 frame, 20 B)` | 0.126 ms | USB emit is only ~2 % of wall time |
| `write(4 frames batched)` | 0.173 ms | batching saves ~1.6 % — not worth a queue |

Note the schedule's "reader" closures include their `encode`, so an in-situ `read:accel` (~2.6 ms
at 400 kHz) is read + encode, not the bare sensor read.

Three conclusions worth keeping:

- **The transport is not the bottleneck.** Batching writes buys ~1.6 % of the loop. Don't bother.
- **`encode` is.** Which is why moving the derived `gravity`/`linear_accel` frames to the host
  mattered: it deleted two of every four frames per cycle. If more is ever needed, `cobs_encode`
  is a pure-Python per-byte loop and the obvious next target.
- **A faster read only helps while the loop is the constraint.** 400 kHz was worth ~1.8× per read
  but only +0.5 Hz, because by then `imu_hz` was the cap. It bought idle, not rate.

Reproduce any of this by timing phases on the board with a temporary `code.py` (loop each phase a
few hundred times — CircuitPython's monotonic clock ticks at only ~1024 Hz, so a single read is
unmeasurable), or instrument the real loop by wrapping the `Telemetry._schedule` reader slots and
the emit sink, then subtracting: `wall = reads + emit + pump overhead + sleep`.

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
pixi run circup install adafruit_lsm6ds adafruit_lis3mdl
```

`adafruit_bus_device` and `adafruit_register` are pulled in automatically.
Resulting `pixi run circup freeze` on the board:

```
adafruit_lis3mdl==1.2.8
adafruit_bus_device==5.2.17
adafruit_lsm6ds==4.6.3
adafruit_register==1.11.3
```

`adafruit_bmp280` / `adafruit_sht31d` are **no longer needed** (environmental sensing was
removed); delete them from the board's `lib/` if present.

Then copy the shared modules plus the **chosen transport's** `code.py` to the CIRCUITPY drive
root (the board auto-reloads and starts streaming):

```bash
# USB serial build (default)
cp feather_protocol.py sensors.py telemetry.py board/serial/code.py /media/bob/CIRCUITPY/ && sync

# BLE build — install the BLE library once, then deploy the ble entry
pixi run circup install adafruit_ble
cp feather_protocol.py sensors.py telemetry.py board/ble/code.py /media/bob/CIRCUITPY/ && sync
```

**Do not copy `motion.py`** — it is host-only, like `stream.py`/`read_stream.py`. Only one
`code.py` runs at a time; swap builds by re-copying the other transport's `code.py`.

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

Example decoded output (board at rest, on USB power) — the raw wire streams:

```
accel     x=0.024  y=0.564  z=9.870      # z ≈ g, board flat (gravity included)
gyro      x=0.060  y=-0.053 z=-0.056     # ≈ 0, stationary
mag       x=-35.94 y=25.17  z=-28.97
battery   voltage_v=4.103  percent=90.27  usb_connected=1
```

`gravity` / `linear_accel` are absent here by design — the readers print what the board sends;
the apps derive those two (see [Derived motion](#derived-motion-gravity--linear_accel)).

## Host library & app integration

Both transports present the **same interface** for consuming the stream from an existing loop
without blocking:

- `poll()` returns the `SensorRecord`s (SI-converted; `error` records carry the source *name*)
  that arrived since the last call — pump it once per iteration.
- `open_if_available(...)` probes for a real frame and returns the stream, or `None` when the
  device is absent — so callers run with or without the board.
- `FrameRecordDecoder` (in [`stream.py`](stream.py)) is the shared bytes→records decode used by
  both `FeatherSenseStream` (serial) and `FeatherSenseBLEStream` (BLE). It decodes **only what
  arrives** — it does not synthesize the derived streams, so a recorder fed from it stores raw
  signals only.

**Deriving gravity / linear_accel.** Each consumer applies [`motion.py`](motion.py) itself, at the
point that suits it:

| Consumer | How | Result |
|----------|-----|--------|
| `rerun_viewer` (live) | one `GravityFilter` per logger, fed each `accel` record | derived plots alongside the raw ones |
| `recording` (capture) | — | stores **raw accel only** |
| `recording.reader.Recording` (offline) | `derive_motion` over the stored accel | `feather["gravity"]` / `["linear_accel"]`, flagged `derived=True` |

```python
with Recording("session.h5") as r:
    r.feather["accel"]         # stored, derived=False
    r.feather["gravity"]       # derived on read at motion.GRAVITY_TAU_S
    r.motion(tau_s=0.1)        # re-derive from the same raw accel, steadier/faster
```

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

