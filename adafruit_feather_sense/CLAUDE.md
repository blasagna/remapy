# `adafruit_feather_sense/`

CircuitPython app for the **Adafruit Feather Bluefruit Sense (nRF52840)** that streams its
onboard sensors over **USB serial or BLE** (interchangeable transports, same wire protocol), plus
host-side readers. Mixed runtime: some files run on the board, some on the host, one is shared.
See `adafruit_feather_sense/README.md` for the full protocol spec, `circup` list, and deploy steps.

- **On the board** (deploy = copy the four shared modules + the chosen entry to the CIRCUITPY
  root as `code.py`; libs via `circup install adafruit_lsm6ds adafruit_lis3mdl neopixel`):
  `sensors.py`
  (`SensorHub(imu_hz=...)` — **raw signals only**: IMU with LSM6DS33/TR-C fallback, LIS3MDL,
  battery via `board.VOLTAGE_MONITOR`; the I2C bus is opened at **400 kHz**, not the 100 kHz
  `busio` default — measured ~1.8× per read). Environmental sensing (BMP280 temp/pressure/altitude,
  SHT31-D humidity) was **removed**: forced-mode conversions blocked the loop ~152 ms/s (~15 % of
  wall, ~7.6 lost IMU samples/s) for unused 1 Hz data. Re-add only with the BMP280 in
  `MODE_NORMAL`. `read_imu()` returns accel **and** gyro from **one 12-byte burst** at reg `0x22`
  (they are contiguous on the chip; driver sets `_bdu`), so they share **one sample instant and one
  timestamp** — the point is simultaneity, not just the halved I2C cost; don't split it back into
  two property reads. The IMU **ODR is set, never inherited**: `_odr_for` picks the slowest rate
  that oversamples the poll 2× (100 Hz → 208). The driver's 104 Hz default is a fine 2.08× at
  `imu_hz=50` and a broken 1.04× at 100. Don't raise it further — ODR above 2× buys noise (~√2 RMS
  per doubling), not information.
  Then `telemetry.py` (`Telemetry.pump(now_ms, emit) -> next_due_ms`
  — the transport-agnostic sample/schedule/encode loop, rates are ctor args; it defers its
  `sensors` import so the host can import it, which is what makes the schedule testable); and
  **two entry points, a literal `code.py` each**: `board/serial/code.py` (USB, `emit =
  sys.stdout.buffer.write`, IMU **100 Hz**) and `board/ble/code.py` (Nordic UART peripheral named
  `FeatherSense` via `adafruit_ble` `UARTService`, `emit = uart.write`, IMU **50 Hz** — the link
  measures ~100 Hz IMU before saturating, so 50 has ~2× margin; the old "~1–2 KB/s" figure was
  never measured and understated it ~2×). Each entry builds one `SensorHub` and injects it
  (`Telemetry(hub=hub)`), passing the same `imu_hz` to both so the ODR and the poll agree.
  `Telemetry` also takes `on_battery(percent, usb_connected)` — an optional sink fired from the
  existing battery slot, so a display can piggyback that read instead of polling (used by the
  status LED; must not raise, or `pump`'s handler eats the battery frame) — and
  `on_pulse(now_ms)` / `pulse_hz` (default 15), a second sink on its **own** slot for a consumer
  that must animate rather than observe (the LED's charging ramp; 0.2 Hz can't drive one). It
  reads no sensor and puts **nothing on the wire**, and is left out of the schedule entirely when
  unset.
- **The schedule runs on integer milliseconds — never `time.monotonic()`.** CircuitPython builds
  **single-precision** floats, so `monotonic()`'s ULP grows with uptime (~2 ms at 5.6 h, ~4 ms at
  10 h). Combined with the old `due = now + interval` (which rescheduled from the *observed* time
  and so baked each cycle's lateness in permanently), the sample rate **silently decayed with
  uptime**: 48.9 Hz just after a reflash but **43.5 Hz at 5.6 h**, against a 50 Hz nominal, with
  timestamps still correct so nothing looked wrong. Every pre-2026-07-16 benchmark in the README
  was taken right after a flash, which is why nobody saw it. `pump` now takes integer `now_ms`
  (from `monotonic_ns`) and advances `due += interval` from the deadline, with a clamp that drops
  the backlog after a long stall instead of bursting stale samples: **100.0 % of nominal at 5.6 h
  uptime**. Don't reintroduce float seconds here, and don't "simplify" the reschedule back to
  `now + interval`.
- **Measured rate chain (accel):** 41 → 48.3 (env removal) → 48.8 (400 kHz) *[all at ~0 uptime;
  43.5 at 5.6 h]* → **50.0** (integer-ms schedule + drift fix, at 5.6 h) → **100.0**
  (burst read + ODR 208 + `imu_hz=100`). Loop **ceiling** (probe with `imu_hz=400`, the only way
  to see a loop win while `imu_hz` still caps the rate): 191.5 → 255.5 (fused `encode_xyz`) →
  **290.5 Hz** (`split`-based `cobs_encode`). So the shipped 100 Hz build is ~34 % of ceiling,
  **~66 % idle**.
- **Status LED:** `status_led.py` (board) — lights the onboard NeoPixel red/yellow/green by
  battery level (<25 / 25–60 / >60 %, ±3 % hysteresis per edge, edges exclusive-below, so a
  reading resting on a threshold can't flicker), and **pulses that same color while charging**
  (12→100→12 % over 2 s, quantized to 16 steps so the "write only on change" guard still catches).
  Color means level and only level; charging is carried by the animation. Nothing latches — the
  band moves both ways. **Display only:** never on the wire, no protocol change.
  **USB picks the animation, never the band.** `VOLTAGE_MONITOR` reads the **battery terminal**,
  not the charger's output: measured while charging, two packs gave 4.00 V/80.3 % and
  4.09 V/89.5 %, so the estimate still tracks the pack when plugged in. The README's old "reads
  green on USB regardless of the pack" gotcha was never measured and is **wrong** — a first cut of
  the charging feature believed it, capped the band while on USB (ceiling = last band seen
  unplugged, yellow when none) and, since that ceiling lives in RAM and the board almost always
  boots plugged in, pinned the LED to a permanent amber. Don't reintroduce a cap. Charging does
  elevate the reading slightly; no offset corrects it because none has been measured — add one
  only with a number attached, and only if a low pack is seen reading high on the charger.
  Deliberately subordinate to streaming, and the shape is **measurement-driven**: it is driven by
  `Telemetry(on_battery=led.update)`, adding **no per-iteration call** to the sampling loop. The
  first cut called `StatusLED.tick(now)` from `code.py` each iteration and measured **-0.93 accel
  samples/s** (49.14 → 48.22, median 49 → 48) — of which **-0.59 was the bare guard check**, not
  the ADC read: a method call here is ~150 µs and the loop turned ~50-60×/s. Riding the battery
  slot costs **-0.21** (median 49). Don't reintroduce a per-iteration LED call — **the rule got
  stricter**, since the loop now turns ~100-120×/s, so any per-iteration cost roughly doubles.
  The charging ramp can't ride a 0.2 Hz slot, so it takes its **own** `Telemetry` slot
  (`on_pulse=led.pulse` at 15 Hz) rather than a call in `code.py` — same rule, one more scheduled
  slot; off USB `pulse` returns on a single attribute test. Measured on the board while charging
  (so the ramp was live): **accel 100.0/s device-side, exactly nominal**, no error frames. That
  **bounds** the cost, it doesn't measure it — at ~34 % of loop ceiling, ~2 ms/s of extra work
  can't move the rate. Probe with `imu_hz=400` if the real per-call cost ever matters.
  `tick(now)` survives **only** for the BLE advertising wait, where `pump`
  isn't running so nothing drives `on_battery`/`on_pulse`, and the idle loop makes the call free. `update`
  never raises (an escaping error would trip the BLE re-advertise handler) and writes the pixel
  only when the band changes. `board`/`neopixel` are imported in `__init__`, not at module scope,
  so a board missing the lib degrades to no LED instead of a crash loop — and `band_for` plus the
  write path (via an injected `pixel`) stay host-importable and unit-tested.
- **Shared** (board + host, pure `struct`): `feather_protocol.py` — a **TLV-over-COBS** wire
  protocol. Each sample is one COBS-framed record `[type][len][timestamp_u32][int32…]`
  terminated by `0x00`; **no floats on the wire** — values are scaled fixed-point int32 (shared
  `SCALES`), converted SI↔int by `to_raw`/`to_si`. Live types `0x01`–`0x05`: accel/gyro/mag/
  battery + `error` (streamed on any caught sampling/encode failure). `0x06` gravity / `0x07`
  linear_accel are **host-derived pseudo-types, never on the wire** (no `SCALES` entry — built in
  SI). Codes are dense and carry **no compatibility guarantee** — board and host ship from this one
  file, so renumbering is resolved by reflashing; recordings are unaffected (`/feather` groups are
  keyed by stream *name*). Add a stream = append the next free code.
  **`encode_xyz` is the board's fast path** for the three-axis types: it fuses `to_raw` + pack +
  frame into a single `struct.pack` (~15 interpreted ops → one C call), worth **+33 % of loop
  ceiling** — `encode` is the loop's most expensive step, so this is where board wins live.
  `cobs_encode` uses `bytes.split` (turns the interpreted loop once per *chunk*, not per *byte*;
  +14 %). Both are **byte-identical** to the originals and must stay so — `tests/test_feather.py`
  keeps the old encoders verbatim as oracles and fuzzes ~8000 cases plus the 254-byte block-split
  boundary. Note for future optimisers: a `bytes.find`-based COBS is a **0.63× regression**,
  because a real payload is half zero bytes (small fixed-point int32s), so there are no long runs
  to scan — profile the payload, not just the code.
- **Host-derived motion:** `motion.py` (host-only) — `GravityFilter.update(ts_ms, xyz)` /
  `derive_motion(ts, values, tau_s)` reconstruct `gravity` (single-pole low-pass, `GRAVITY_TAU_S`)
  and `linear_accel = accel − gravity` from the raw accel stream. This used to run on the board
  (`SensorHub.read_motion`), which made one accel read cost three encoded frames; `encode` is the
  loop's most expensive step (~1.275 ms/frame), so moving it here removed two of every four
  frames/cycle. It also keys the filter off the *device* timestamp (true sample spacing) and makes
  `tau_s` a read-time choice.
- **On the host** — both transports share one interface (`poll()` → SI `SensorRecord`s, `errors`,
  `close()`, `.port`, `open_if_available`), so apps are transport-agnostic. `stream.py` —
  `FeatherSenseStream` (pyserial, non-blocking) + `FrameRecordDecoder` (the shared bytes→records
  decode). `ble_stream.py` — `FeatherSenseBLEStream` (**bleak**; scan/connect/notify on the Nordic
  UART Service run on a background asyncio thread, a queue bridges to the sync `poll()`).
  `read_stream.py` / `read_ble.py` are the standalone pretty-printer CLIs. `__init__.py` exposes
  `open_feather(transport, *, port=None, address=None)` (lazy-imports the backend) — used by
  `rerun_viewer`/`recording`. Import note: host modules add the package dir to `sys.path` so the
  shared `feather_protocol` resolves as a top-level module; **don't import the board files**
  (`board/*/code.py`, `sensors.py`, `telemetry.py`) on the host. `status_led.py` is the
  deliberate exception — it defers its `board`/`neopixel` imports into `StatusLED.__init__`, so
  the module imports cleanly on the host and its pure `band_for` is unit-tested there.

## Tests (`tests/test_feather.py`)

The Feather Sense host integration: `FeatherSenseStream` decode/poll and `open_if_available` probe
(via `FakeSerial`), the shared `FrameRecordDecoder`, the `open_feather` transport dispatch
(serial/ble backends mocked — no real radio), the host `motion` derivation
(`GravityFilter`/`derive_motion` — seeding, tilt bleed, transients, clock wrap, batch/live
agreement), the recorder's `/feather` datasets + `Recording.feather` (including that derived streams
are *not* stored and that `motion(tau_s=…)` re-derives), the viewer's `log_sensors` + its derived
plots (with `rr` mocked), and the status LED (`band_for` bands + hysteresis in both directions, plus
`StatusLED.update`/`pulse`/`tick` against an injected `FakePixel` — pinning that it writes only on a
change, never raises, no-ops without a pixel, and that the charging rules hold: USB selects the
animation and never the band (the capping regression), an almost-full pack reads green on the charger
while a low one still reads red, the ramp keeps the hue and never goes dark, and the 0.2 Hz battery
slot doesn't stamp full brightness over a ramp in progress). No board, serial port, or BLE adapter
needed.

Also **`TelemetryScheduleTests`** — the board's sampling schedule, driven by a fake hub and a fake
clock (reachable because `telemetry.py` defers its `sensors` import; `status_led.py` is the same
trick). It pins the two bugs that cost the rate: no erosion under **sustained lateness** (the
`due = now + interval` regression) and no erosion at **high uptime** (the float32 `monotonic()`
decay), plus the stall clamp, one-read-serves-both-streams, and that accel/gyro share a timestamp.
Plus **`CobsEncodeTests`/`EncodeXyzTests`** (byte-identity fuzz vs. the original encoders) and
**`RateTrackerTests`** (the `--stats` arithmetic — it is the acceptance instrument, and it used to
over-report ~10 %).
