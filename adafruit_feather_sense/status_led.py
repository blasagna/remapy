"""Onboard NeoPixel battery indicator for the Feather Sense (board side).

`StatusLED` lights the board's single RGB pixel red / yellow / green by battery
level, and **pulses that same color while charging**, so the board reports both
its charge level and its power state while untethered instead of only over the
wire. It drives `board.NEOPIXEL` and holds no protocol/transport knowledge —
nothing about the LED reaches the wire.

Three properties this module owes the rest of the system:

- *The stream comes first.* The level is driven from `Telemetry`'s existing
  0.2 Hz battery slot (`Telemetry(on_battery=led.update)`), so the sampling loop
  gains **no per-iteration call**. This is measured, not assumed: an earlier
  design called `tick()` from `code.py` every iteration and cost ~1 accel
  sample/s (49.1 → 48.2 Hz), of which ~0.6 was the bare guard check rather than
  the ADC read — a method call here is ~150 µs, and the loop turns ~50-60×/s.
  Riding the battery slot restores the median rate to the LED-free 49 Hz.
  Animating the pulse needs a faster driver than 0.2 Hz, so `pulse(now_ms)` gets
  its own low-rate `Telemetry` slot (`on_pulse=led.pulse`, `PULSE_HZ`) — still
  scheduled, still never per-iteration. See `telemetry.py`.
- *It never raises.* `update`/`pulse` swallow their own failures: a cosmetic LED
  must not stall sampling, and on the BLE build an escaping exception would trip
  the outer re-advertise handler.
- *The LED is optional.* `board`/`neopixel` are imported in `__init__` rather
  than at module scope, so a board without the `neopixel` library streams on with
  the LED simply absent, and the host can import this module for tests.

**The band is the reading, on USB as much as off it** — measured, after a wrong
turn worth recording. This module briefly capped the displayed band while
charging (ceiling = the last band seen unplugged, `YELLOW` when none), on the
belief — inherited from the README, never checked — that `VOLTAGE_MONITOR` reads
the charger's output and so pins to ~100 % whenever USB is attached. It does not:
it reads the battery terminal. Two packs measured while charging came back
**4.00 V / 80.3 %** and **4.09 V / 89.5 %**, which is a reading that plainly still
tracks the pack. The cap also could not work in practice — its ceiling lived in
RAM and the board almost always boots plugged in, so it sat at yellow forever and
the LED reported nothing at all.

What *is* true: charging elevates the terminal voltage somewhat, so a charging
reading runs a little optimistic. No offset is applied to correct it, because no
offset has been measured — an uncalibrated fudge factor here would be the same
mistake in a smaller costume. If a genuinely low pack ever reads too high on the
charger, this is the place to put one, with the measurement written down.

Level is not monotonic and nothing latches: the band moves in both directions.
"""

RED = (255, 0, 0)
YELLOW = (255, 160, 0)
GREEN = (0, 255, 0)

# Pulse shape: a linear ramp up and back down over PULSE_PERIOD_MS, spanning
# _PULSE_MIN..1.0 of the band color. It never reaches 0 — a pulse that blinks
# fully off reads as "fault" rather than "charging", and loses the color for
# half its cycle.
PULSE_PERIOD_MS = 2000
PULSE_HZ = 15  # ~30 steps/cycle, enough for a smooth ramp at 16 brightness levels
_PULSE_MIN = 0.12
_PULSE_STEPS = 16  # quantized, so a repainted step is skipped rather than re-shown

# Band edges on read_battery()'s linear 0-100 % estimate. _HYST widens whichever
# edge would take us *out* of the current band, so a reading resting on an edge
# settles instead of oscillating.
_RED_MAX = 25.0
_YELLOW_MAX = 60.0
_HYST = 3.0


def band_for(percent, current=None):
    """Color for `percent`, with hysteresis around the `current` color.

    Pass the color presently displayed as `current` (`None` when nothing is lit
    yet) to get the sticky edges; the bare thresholds apply otherwise.
    """
    red_max = _RED_MAX
    yellow_max = _YELLOW_MAX
    if current == RED:
        red_max += _HYST  # need 28 % to climb out of red
    elif current == YELLOW:
        red_max -= _HYST  # need below 22 % to fall back to red
        yellow_max += _HYST  # need 63 % to climb to green
    elif current == GREEN:
        yellow_max -= _HYST  # need below 57 % to fall back to yellow

    if percent < red_max:
        return RED
    if percent < yellow_max:
        return YELLOW
    return GREEN


def pulse_level(now_ms, period_ms=PULSE_PERIOD_MS):
    """Brightness fraction in `_PULSE_MIN`..1.0 for the ramp at `now_ms`.

    Quantized to `_PULSE_STEPS` so a caller polling faster than the ramp changes
    can compare against what it last wrote and skip the repaint. Pure and
    clock-free: the phase comes from the caller's integer ms, the same clock the
    schedule runs on (never `time.monotonic()` — see `telemetry.py`).
    """
    phase = now_ms % period_ms
    half = period_ms // 2
    frac = phase / half if phase < half else (period_ms - phase) / (period_ms - half)
    level = _PULSE_MIN + (1.0 - _PULSE_MIN) * frac
    return int(level * _PULSE_STEPS + 0.5) / _PULSE_STEPS


def scale(color, level):
    """`color` dimmed to `level` (0..1)."""
    return (int(color[0] * level), int(color[1] * level), int(color[2] * level))


class StatusLED:
    """Drives the onboard pixel from a battery level and USB power state.

    Three ways in, because the two board builds differ in what drives them:

    - `update(percent, usb)` — **the level.** Wired to
      `Telemetry(on_battery=...)`, so it rides the battery slot's existing read
      and adds no call to the sampling loop at all.
    - `pulse(now_ms)` — **the animation.** Wired to `Telemetry(on_pulse=...)`,
      its own `PULSE_HZ` slot, because a 0.2 Hz battery slot cannot animate
      anything. A no-op (one attribute test) when not charging.
    - `tick(now)` — self-driving: reads the battery itself (rate-limited) *and*
      pulses. Only for loops with no `Telemetry` at all, i.e. the BLE build while
      advertising, where nothing is being sampled and the cost is irrelevant.

    `usb` selects *how* the band is shown (pulsing vs solid), never *which* band
    — the color is the reading either way. See the module docstring for the
    capping scheme that used to live here and why it was wrong.

    `pixel` is injectable so the write logic is testable off-board; leave it
    `None` on the board and the NeoPixel is constructed here.
    """

    def __init__(self, hub, brightness=0.15, interval_s=1.0, pixel=None):
        self._hub = hub
        self._interval_s = interval_s
        self._next_due = 0.0
        self._color = None
        self._charging = False
        self._written = None  # last RGB actually pushed to the pixel
        if pixel is not None:
            self._pixel = pixel
            return
        try:
            import board
            import neopixel

            self._pixel = neopixel.NeoPixel(
                board.NEOPIXEL, 1, brightness=brightness, auto_write=False
            )
        except Exception:  # noqa: BLE001 - no neopixel lib / no pixel -> just don't indicate
            self._pixel = None

    def _write(self, color):
        """Push `color` to the pixel, skipping an unchanged repaint.

        `show()` bit-bangs the pixel with interrupts off, so a redundant write
        spends loop time to display nothing new. The pulse quantizes its
        brightness (see `pulse_level`) precisely so this guard keeps catching.
        """
        if color != self._written:
            self._pixel.fill(color)
            self._pixel.show()
            self._written = color

    def update(self, percent, usb=0):
        """Show the band for `percent`, pulsing it when `usb` power is present.

        Solid when running on battery; charging hands the pixel to `pulse`, so
        this only seeds the color (at full brightness, so the LED is still right
        for a build that wires `on_battery` but not `on_pulse`). Never raises.
        """
        if self._pixel is None:
            return
        try:
            charging = bool(usb)
            previous = self._color
            color = band_for(percent, self._color)
            self._color = color
            self._charging = charging
            # Charging: only seed the color (on a change, or the very first
            # write) and let `pulse` own the pixel from there, so a slow battery
            # slot can't stamp full brightness over the middle of a ramp.
            if not charging or color != previous or self._written is None:
                self._write(color)
        except Exception:  # noqa: BLE001 - a cosmetic LED must not stop the stream
            pass

    def pulse(self, now_ms):
        """Advance the charging ramp. No-op unless charging. Never raises.

        Cheap on the common path: one attribute test and a return when the board
        is on battery, so wiring this to a `Telemetry` slot costs the loop a
        scheduled call rather than an animation.
        """
        if not self._charging or self._pixel is None or self._color is None:
            return
        try:
            self._write(scale(self._color, pulse_level(now_ms)))
        except Exception:  # noqa: BLE001 - a cosmetic LED must not stop the stream
            pass

    def tick(self, now):
        """Self-driving `update` + `pulse` for loops with no `Telemetry`.

        Don't call this per-iteration from a sampling loop — the guard alone
        measured ~0.6 accel samples/s. Use `Telemetry(on_battery=led.update,
        on_pulse=led.pulse)`. The battery read is rate-limited to `interval_s`;
        the pulse runs every call, which is what makes the ramp smooth in the
        BLE advertising busy-wait.
        """
        if self._pixel is None:
            return
        if now >= self._next_due:
            self._next_due = now + self._interval_s
            try:
                _, percent, usb = self._hub.read_battery()
            except Exception:  # noqa: BLE001 - a bad read must not stop the stream
                usb = None
            if usb is not None:
                self.update(percent, usb)
        self.pulse(int(now * 1000))
