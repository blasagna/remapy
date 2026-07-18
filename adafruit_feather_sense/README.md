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

**One burst, one sample.** Accel and gyro are two streams from one chip, and
`SensorHub.read_imu()` reads them as **one 12-byte I2C burst** at `0x22` (gyro
`OUTX_L_G` and accel `OUTX_L_A` are contiguous; the driver's `_bdu` bit is what
makes a multi-byte read internally consistent) rather than through the driver's
separate `.acceleration`/`.gyro` properties. It halves the IMU's I2C cost, but
the reason it is worth reaching past the driver is **simultaneity**: read
separately the two are ~1 ms apart and independently timestamped, which is a lie
about a single physical sample instant that any downstream fusion would inherit.
They now share one timestamp exactly (pinned by a test, and verified end-to-end
over 463 recorded pairs at max |Δt| = 0 ms).

**IMU output data rate is set, not inherited.** `SensorHub(imu_hz=...)` picks the
slowest ODR that oversamples the poll **2×** (`_odr_for`: 50 Hz → 104, 100 Hz →
208). The driver defaults both axes to 104 Hz, which happened to be a fine 2.08×
for the old 50 Hz build and a useless 1.04× for a 100 Hz one — the loop would
re-read samples the chip had not refreshed.

Deliberately not *higher* than 2×, because ODR is not free. At rest, holding
`imu_hz=100` and changing only the ODR (12 s, board on a desk):

| ODR | accel RMS σ | gyro RMS σ |
|-----|-------------|------------|
| 104 | 0.00968 m/s² | 0.00147 rad/s |
| **208** (shipped) | 0.01021 m/s² | 0.00209 rad/s |
| ratio | **1.05×** | **1.42×** |

The gyro tracks the textbook √2-per-doubling almost exactly (1.42 vs 1.414);
the accel barely moves (1.05×), because its at-rest noise is dominated by
ambient vibration rather than sensor bandwidth — so **don't assume √2 applies to
every stream, it didn't here.** 208 is still the right trade at 100 Hz: stale
duplicate samples are a worse defect than +0.0006 rad/s (~0.035 °/s) of gyro
noise. But going to 416 "for headroom" would buy another √2 on the gyro for
nothing.

These are also the **at-rest baseline** for the IMU signal (accel mean |a| =
9.96 m/s², i.e. gravity, as a sanity check on the burst read's scaling).

*Not sampled:* the APDS9960 (light/color/gesture/proximity) and the PDM
microphone. Both are easy to add later as new message types.

## Status LED

`status_led.py` lights the onboard NeoPixel by battery level and **pulses it while charging**, so
the board reports its own state while untethered rather than only over the wire. It is **display
only** — nothing about it reaches the protocol, and the host never sees it.

| Band | Level | Sticky edge |
|------|-------|-------------|
| 🔴 red | < 25 % | must reach 28 % to leave red |
| 🟡 yellow | 25 – 60 % | must reach 63 % to go green, drop below 22 % to go red |
| 🟢 green | > 60 % | must drop below 57 % to go yellow |

| Power | Pixel |
|-------|-------|
| on battery | solid, at the band color |
| charging (USB attached) | **the same band color, ramping** 12 % → 100 % → 12 % over 2 s |

Level is read from the same `read_battery()` estimate as the `battery` stream. Nothing latches:
the band moves **both** ways. The ±3 % hysteresis on each edge is what stops a reading resting on
a threshold from flickering between two colors. Color always means level and only level —
charging is carried by the animation, so the two are readable independently.

**The stream comes first**, and the design is the measured consequence of that. The level rides
the **existing 0.2 Hz battery slot** via `Telemetry(hub=hub, on_battery=led.update)`, so the
sampling loop gains **no per-iteration call**. The ramp obviously cannot animate at 0.2 Hz, so it
gets its **own scheduled slot** — `Telemetry(on_pulse=led.pulse)` at `PULSE_HZ` (15 Hz) — which is
still not a per-iteration call: it costs `pump` one more list entry to walk, plus 15 calls a
second, and it is off the schedule entirely when unset. Off USB `pulse` returns on a single
attribute test. Measured on the board (15 s serial capture, charging so the ramp was actually
running): **accel 100.0/s device-side, dead on nominal**, mag 20.0/s, no error frames. That
**bounds** the cost rather than measuring it — the 100 Hz build runs at ~34 % of the loop ceiling,
so anything under ~2 ms/s of extra work disappears into the idle margin and cannot show up as a
rate drop. To see the true per-call cost you would have to probe with `imu_hz=400` as the
`encode_xyz` work did. The LED still never blocks (no sleeps), never touches the I2C bus the IMU
shares, writes the pixel only when the rendered color actually changes — the ramp is quantized to
16 steps so that guard keeps catching — and never raises, since a cosmetic LED must not stall
sampling and on the BLE build an escaping exception would trip the re-advertise handler.

The first cut instead called `led.tick(now)` from `code.py` every iteration, which **measured**
(15 s captures, 3+ runs per build, board at rest):

| Build | accel | vs no-LED |
|-------|-------|-----------|
| no LED at all | 49.14 Hz | — |
| `tick()` per iteration, body dormant | 48.56 Hz | **-0.59** |
| `tick()` per iteration, live @ 1 Hz | 48.22 Hz | **-0.93** (median 49 → 48) |
| **`on_battery` (shipped)** | **48.94 Hz** | **-0.21** (median 49, no glitches) |

Two-thirds of that cost was the *bare guard check*, not the ADC read: a method call here is
~150 µs (cf. `cobs_encode`'s 708 µs) and the loop turns ~50-60×/s. Hence `on_battery`. The
tradeoff is a 5 s refresh instead of 1 s — irrelevant for a battery indicator.

`StatusLED.tick(now)` still exists for the **BLE advertising wait**, where `pump` isn't running
so nothing drives `on_battery`/`on_pulse` — exactly when the board is on battery with nobody
reading the stream. That loop has nothing else to do, so the call cost is free there; `tick`
rate-limits the ADC read to `interval_s` but pulses on every spin. Verified on the board:
from `color=None` at boot, the pixel lights on the **first spin** of the advertising loop with no
central connected, and stays lit across connect → disconnect → re-advertise. (`tick` is called
~8800×/s in that busy-wait and rate-limits itself to one ADC read/s.)

**The band is the reading, plugged in or not.** USB changes *how* the band is shown (pulsing
rather than solid), never *which* band. Measured on the board while charging, two packs:

| Pack | Reading on the charger | Band |
|------|------------------------|------|
| partly charged | 4.00 V → 80.3 % | 🟢 green |
| almost full | 4.09 V → 89.5 % | 🟢 green |

Those differ, which is the point: `VOLTAGE_MONITOR` reads the **battery terminal**, not the
charger's output, so the estimate still tracks the pack while USB is attached. An earlier cut of
this feature assumed the opposite (see the correction below) and capped the band while charging,
which pinned the LED to amber and made it report nothing.

Charging *does* elevate the terminal voltage somewhat, so a reading taken on the charger runs a
little optimistic. **No offset corrects for it**, because no offset has been measured — an
uncalibrated fudge factor would be guesswork dressed as precision. If a genuinely low pack is ever
seen reading too high on the charger, that is where to add one, with the number written down.
(That case is untested: both measurements above are of healthy packs.)

The `battery` stream on the wire is unaffected — it still carries the raw estimate plus the
`usb_connected` flag, and the host can interpret them however it likes.

> **Correction (2026-07-18).** This README previously stated that the pixel "reads green on USB
> regardless of the pack's true state", as a property of `read_battery()`. That was never
> measured, and the readings above contradict it. A feature was built on top of that claim before
> anyone checked it; the cost was an LED that displayed a constant amber. Measure the sensor
> before designing around its failure mode.

If the `neopixel` library is missing from the board, `StatusLED` disables itself and the board
streams on with no LED — the indicator is never worth a crash loop.

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
  IMU at 100 Hz (~4.4 KB/s); the link is nowhere near a constraint.
- **BLE** — frames written to the **Nordic UART Service**; host reads with `bleak`. Lower
  bandwidth, so the board build samples the IMU slower (50 Hz, ~2.2 KB/s). The link is
  **measured** to carry ~100 Hz IMU (~4.4 KB/s) before it saturates — see
  [Sample rates](#sample-rates). An earlier "~1–2 KB/s" figure here was never measured and
  understated the link by ~2×.

The sampling loop (`telemetry.py`) is transport-agnostic; each board build is a thin `code.py`
that only supplies the emit sink. On the host, both transports expose the **same stream
interface** (`poll()` / `errors` / `close()` / `.port` / `open_if_available`), so the apps select
one with `--feather-transport {serial,ble}` and nothing else changes.

## Files

| File | Runs on | Purpose |
|------|---------|---------|
| [`feather_protocol.py`](feather_protocol.py) | board **+** host | TLV encode/decode, COBS, `FrameDecoder` |
| [`sensors.py`](sensors.py) | board | `SensorHub` — inits the raw sensors, one read method per stream |
| [`telemetry.py`](telemetry.py) | board | `Telemetry.pump(now_ms, emit)` — the shared sample/schedule/encode loop (rates configurable). Host-importable and tested: it defers its `sensors` import |
| [`status_led.py`](status_led.py) | board | `StatusLED.update/pulse/tick` — onboard NeoPixel battery indicator, pulsing while charging (`band_for`/`pulse_level` are pure and host-testable) |
| [`motion.py`](motion.py) | host | `GravityFilter` / `derive_motion` — gravity + linear_accel from raw accel |
| [`board/serial/code.py`](board/serial/code.py) | board | USB serial entry: `emit = sys.stdout.buffer.write`, full rates (IMU 100 Hz) |
| [`board/ble/code.py`](board/ble/code.py) | board | BLE entry: Nordic UART peripheral (`FeatherSense`), `emit = uart.write`, reduced rates (IMU 50 Hz) |
| [`stream.py`](stream.py) | host | serial `FeatherSenseStream` + shared `FrameRecordDecoder` (bytes→SI records) |
| [`ble_stream.py`](ble_stream.py) | host | BLE `FeatherSenseBLEStream` (bleak on a background thread; same interface) |
| [`read_stream.py`](read_stream.py) | host | serial reader CLI: decode, int→SI (`to_si`), pretty-print / `--stats` / `--raw` |
| [`read_ble.py`](read_ble.py) | host | BLE reader CLI (bleak): `--address` / `--name` / `--stats` / `--only` |
| [`__init__.py`](__init__.py) | host | `open_feather(transport, ...)` — transport-selecting factory used by the apps |

## Sample rates

Rates are `Telemetry(...)` constructor args, so each board build picks its own. Nominal vs.
**measured** (board at rest, CircuitPython 10.2.1). Quote the **device-clock** rate: it is derived
from the timestamps the board stamps at read, so it measures the board rather than the host's
polling (see [Measuring](#measuring)).

| Stream | Nominal (serial) | Measured | % of nominal | Nominal (BLE) | Measured |
|--------|------------------|----------|--------------|---------------|----------|
| accel | 100 Hz (`imu_hz`) | **100.0 Hz** | 100 % | 50 Hz (`imu_hz`) | **50.0 Hz** |
| gyro | 100 Hz (`imu_hz`) | **100.0 Hz** | 100 % | 50 Hz (`imu_hz`) | **50.0 Hz** |
| mag | 20 Hz (`mag_hz`) | **20.0 Hz** | 100 % | 10 Hz (`mag_hz`) | **10.0 Hz** |
| battery | 0.2 Hz (`battery_hz`) | 0.2 Hz | 100 % | 0.2 Hz | 0.2 Hz |

### The rate used to decay with uptime

**Every measurement in this file before 2026-07-16 was taken moments after a reflash, and that
is the only reason the old numbers looked good.** The rate was a function of *board uptime*:

| Build | uptime | accel | mag |
|-------|--------|-------|-----|
| pre-fix (`imu_hz=50`) | ~0 (just reflashed) | 48.9 Hz | 20.7 Hz |
| pre-fix (`imu_hz=50`) | **5.6 h** | **43.5 Hz** | 18.7 Hz |
| post-fix (`imu_hz=50`) | 5.6 h | **50.0 Hz** | 20.1 Hz |

Two bugs compounded, and neither was visible:

1. **CircuitPython builds single-precision floats.** `time.monotonic()` returns seconds since
   boot, so its ULP grows with uptime: **~2 ms at 5.6 h**, ~4 ms at 10 h. A 20 ms schedule was
   being quantized by a clock that got coarser the longer the board ran.
2. **The schedule absorbed lateness instead of correcting it.** `pump` rescheduled each slot as
   `due = now + interval` — from the *observed* time, which is always ≥ the deadline. So every
   cycle's quantization and work time was baked permanently into the next period.

Together: at 5.6 h uptime the effective period inflated 20 ms → ~23 ms, i.e. 43.5 Hz against a
50 Hz nominal — while **timestamps stayed correct**, so nothing looked wrong. For a project that
records therapy sessions, that is a data bug wearing a disguise.

The fix is to run the schedule on **integer milliseconds** from `time.monotonic_ns()` (never a
float; ms is also the wire timestamp unit) and to advance `due += interval` from the deadline,
with a clamp that drops the backlog after a long stall rather than emitting a burst of stale
samples. The result holds **100.0 % of nominal at 5.6 h uptime** — better than the pre-fix build
ever managed even at zero uptime.

This also retires a claim: the old "residual ~2.5 % is the cooperative `pump()` loop" was wrong.
It was the scheduler's arithmetic, and it is gone.

### How it got there

Each step measured on the board:

| Change | accel | mag |
|--------|-------|-----|
| original (env/altitude on board, gravity+linear encoded on board, 100 kHz I2C) | 41.0 Hz | 16.0 Hz |
| + environmental sensing removed | 48.3 Hz | 19.2 Hz |
| + I2C at 400 kHz | 48.8 Hz | 19.8 Hz |
| *(the three rows above were measured at ~0 uptime; at 5.6 h the build did 43.5 Hz)* | | |
| + integer-ms schedule, drift fix (`imu_hz` still 50) | **50.0 Hz** | 20.1 Hz |
| + IMU burst read, ODR 208 Hz, **`imu_hz=100`** | **100.0 Hz** | 20.0 Hz |

The original 41 Hz was **not** loop saturation (it still slept 17 % of the time) — it was
*blackout*. The 1 Hz `env`/`altitude` reads blocked for ~152 ms/s, and 152 ms of dead time removes
~7.6 of the 50 accel slots per second: `848 ms / 20 ms = 42.4 Hz`, against 41 measured.

**Why 400 kHz bought only +0.5 Hz.** It halves read time, but by then the loop was no longer the
constraint — the rate is capped by `imu_hz`, and the saved time became *sleep*, not throughput.
Its real value was **headroom**, and this change is what finally spent it: 100 Hz is exactly the
budget that 400 kHz banked. The lesson generalises, and the sections below lean on it — **while
the loop has idle, a loop optimisation shows up as more idle, not as a bigger number here.** To
see one at all you have to remove the `imu_hz` cap first (see [Measuring the
ceiling](#measuring-the-ceiling)).

### What limits BLE

The BLE build ships at `imu_hz=50, mag_hz=10` (~2.2 KB/s). Measured, not assumed:

| Nominal `imu_hz` | measured accel | verdict |
|------------------|----------------|---------|
| 50 (shipped) | **50.0 Hz** | link has margin; errors=0 |
| 150 | ~100.6 Hz | **link saturated** at ~100 Hz IMU (~4.4 KB/s) |

So the link carries ~2× the shipped rate, and the old "~1–2 KB/s" figure understated it by ~2×.
It was never measured; `imu_hz=20` was a guess that happened to be safe.

**Telling "link saturated" from "loop too slow", without instrumenting the board.** BLE
notifications are acknowledged and never dropped, so host arrival rate == board emit rate. The
device-timestamp spacing then decides it: evenly spaced *at nominal* is working; evenly spaced
*slower* than nominal (the 150 Hz row above) means the loop is stalling inside `uart.write`, i.e.
the link; bursts at nominal separated by long gaps mean the 512 B `StreamOut` buffer filled and
`write` hit its 1 s timeout. The loop is independently ruled out by the serial build, which
sustains 100 Hz on the same code.

**Connection interval: measured to be a red herring.** Requesting the 7.5 ms minimum via
`connection.connection_interval` measured **identical** to leaving the negotiated default alone
(50.0 Hz either way), so that code was removed rather than kept as a plausible-looking no-op. MTU
is likewise not the lever: bleak reports the 23-byte default on BlueZ (one 20-byte frame per
notification) while the board still sustains ~110 notifications/s.

### Measuring the ceiling

A loop optimisation is invisible in the table above, because at 100 Hz the loop still idles. So
measure the **ceiling**: set `imu_hz` far past reach (400) and the achieved rate *is* the loop's
throughput. Each row is one build, `--stats` over ~8 s:

| Build (all with the burst read) | ceiling | Δ |
|---------------------------------|---------|---|
| original `encode` + per-byte `cobs_encode` | 191.5 Hz | — |
| **+ fused `encode_xyz`** | 255.5 Hz | **+33 %** |
| **+ `split`-based `cobs_encode`** | **290.5 Hz** | **+14 %** |

So the shipped 100 Hz build runs at ~34 % of ceiling — **~66 % idle**, more headroom than the
58 % it started with, at double the rate.

**This corrects the advice this file used to give.** It named `cobs_encode` "the obvious next
target"; measured, it is worth less than half of the fused encode. The reason is visible in the
data: a real accel payload is 18 bytes of which **8 are zero** (fixed-point int32s of small
numbers are mostly high-order zeros), so COBS emits 9 chunks from 18 bytes and there is little
for a faster scan to chew on. A `bytes.find`-based rewrite is actually **0.63× — a regression**;
`split` wins because it turns the interpreted loop once per *chunk* instead of once per *byte*.
The bigger prize was `to_raw` + `encode`'s list-comp, per-int `struct.pack` and bytearray concats
collapsing into **one `struct.pack`** — ~15 interpreted operations to one C call.

Both rewrites are **byte-identical** to what they replaced, which is not an aspiration: it is
enforced by ~8000 fuzz cases plus the 253/254/255-byte block-split boundaries against the
original encoder kept verbatim as an oracle in `tests/test_feather.py`, cross-checked against the
independent `cobs` PyPI package.

### Where the loop time goes

Measured per phase on the board — useful before optimising the wrong thing. **These are the
pre-2026-07-16 numbers**, i.e. the cost structure that motivated the encode work above; the
`encode` rows no longer describe the shipped code (see the ceiling table for its effect).

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
at 400 kHz) is read + encode, not the bare sensor read. The two separate IMU reads in this table
are now a single 12-byte burst (`read_imu`), so a cycle pays one transaction, not two.

**A caveat this table cost us once:** a per-phase cost sums to less than the loop, and it cannot
tell you what a change is *worth* while the loop still idles. The ceiling probe can. Use this
table to find the expensive phase, and the ceiling to decide whether fixing it mattered.

Four conclusions worth keeping:

- **The transport is not the bottleneck.** Batching writes buys ~1.6 % of the loop. Don't bother.
- **`encode` is** — and it stayed the answer. Moving the derived `gravity`/`linear_accel` frames
  to the host deleted two of every four frames per cycle; fusing what remained into one
  `struct.pack` was worth another +33 % of loop throughput.
- **A faster read only helps while the loop is the constraint.** 400 kHz was worth ~1.8× per read
  but only +0.5 Hz, because by then `imu_hz` was the cap. It bought idle, not rate.
- **Profile the payload, not just the code.** `cobs_encode` looked like the hot spot by share of
  `encode`, but its input is half zeros, which is what made the intuitive `find`-based fix a
  regression. The byte loop was slow because it was *interpreted per byte*, not because it was
  scanning far.

Reproduce any of this by timing phases on the board with a temporary `code.py` (loop each phase a
few hundred times — CircuitPython's monotonic clock ticks at only ~1024 Hz, so a single read is
unmeasurable), or instrument the real loop by wrapping the `Telemetry._schedule` reader slots and
the emit sink, then subtracting: `wall = reads + emit + pump overhead + sleep`. For whole-loop
throughput prefer the [ceiling probe](#measuring-the-ceiling) — no instrumentation, and it cannot
lie to you about idle.

### Measuring

`read_stream.py --stats` / `read_ble.py --stats` print, per stream, per second:

```
--- 1.001 s ---  errors=1
  accel     n= 100  host=  99.9/s  dev= 100.0/s  gap max=  11.0 ms
```

- **`dev`** — from the **device** timestamps (stamped at read on the board), so it measures the
  board independently of host scheduling, USB buffering or BLE batching. **This is the number to
  quote.**
- **`host`** — arrivals over true elapsed wall time. It only tells you the link kept up; it
  swings ±5 % on BLE from the host's own poll jitter while `dev` sits still.
- **`gap max`** — largest interval between consecutive device timestamps. A rate on target with an
  outsized gap means the stream stalled and caught up in a burst, which neither average shows.

**Divide by measured elapsed, not by the nominal interval.** These tools used to print a raw count
labelled `/s` over a window gated on `>= 1.0 s` — which a blocking read always overshoots, so a
true 91 Hz stream reported "100/s". Every board number in this file was read off that tool, and
the error ran in the flattering direction.

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
pixi run circup install adafruit_lsm6ds adafruit_lis3mdl neopixel
```

`adafruit_bus_device`, `adafruit_register` and `adafruit_pixelbuf` are pulled in automatically.
`neopixel` backs the [status LED](#status-led) only — omit it and the board still streams, just
without the indicator.

Resulting `pixi run circup freeze` on the board:

```
adafruit_lis3mdl==1.2.8
adafruit_bus_device==5.2.17
adafruit_lsm6ds==4.6.3
adafruit_register==1.11.3
adafruit_pixelbuf==2.0.12
neopixel==6.4.2
```


`adafruit_bmp280` / `adafruit_sht31d` are **no longer needed** (environmental sensing was
removed); delete them from the board's `lib/` if present.

Then copy the shared modules plus the **chosen transport's** `code.py` to the CIRCUITPY drive
root (the board auto-reloads and starts streaming):

```bash
# USB serial build (default)
cp feather_protocol.py sensors.py telemetry.py status_led.py board/serial/code.py /media/bob/CIRCUITPY/ && sync

# BLE build — install the BLE library once, then deploy the ble entry
pixi run circup install adafruit_ble
cp feather_protocol.py sensors.py telemetry.py status_led.py board/ble/code.py /media/bob/CIRCUITPY/ && sync
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

