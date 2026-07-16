"""Onboard NeoPixel battery indicator for the Feather Sense (board side).

`StatusLED` lights the board's single RGB pixel red / yellow / green by battery
level, so the board reports its own state while untethered instead of only over
the wire. It drives `board.NEOPIXEL` and holds no protocol/transport knowledge —
nothing about the LED reaches the wire.

Three properties this module owes the rest of the system:

- *The stream comes first.* The pixel is driven from `Telemetry`'s existing
  0.2 Hz battery slot (`Telemetry(on_battery=led.update)`), so the sampling loop
  gains **no per-iteration call**. This is measured, not assumed: an earlier
  design called `tick()` from `code.py` every iteration and cost ~1 accel
  sample/s (49.1 → 48.2 Hz), of which ~0.6 was the bare guard check rather than
  the ADC read — a method call here is ~150 µs, and the loop turns ~50-60×/s.
  Riding the battery slot restores the median rate to the LED-free 49 Hz.
- *It never raises.* `update` swallows its own failures: a cosmetic LED must not
  stall sampling, and on the BLE build an escaping exception would trip the
  outer re-advertise handler.
- *The LED is optional.* `board`/`neopixel` are imported in `__init__` rather
  than at module scope, so a board without the `neopixel` library streams on with
  the LED simply absent, and the host can import this module for tests.

Level is not monotonic: the board charges over USB, so the band moves in both
directions and nothing latches. Note that `VOLTAGE_MONITOR` reads the *charge*
voltage while USB is attached, so the pixel generally sits green when plugged in
regardless of the pack's true state (a property of `read_battery()`'s estimate).
"""

RED = (255, 0, 0)
YELLOW = (255, 160, 0)
GREEN = (0, 255, 0)

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


class StatusLED:
    """Drives the onboard pixel from a battery level.

    Two ways in, because the two board builds differ in what drives them:

    - `update(percent)` — **the hot path.** Wired to `Telemetry(on_battery=...)`,
      so it rides the battery slot's existing read and adds no call to the
      sampling loop at all.
    - `tick(now)` — self-driving (reads the battery itself, rate-limited). Only
      for loops with no battery slot, i.e. the BLE build while advertising,
      where nothing is being sampled and the cost is irrelevant.

    `pixel` is injectable so the write logic is testable off-board; leave it
    `None` on the board and the NeoPixel is constructed here.
    """

    def __init__(self, hub, brightness=0.15, interval_s=1.0, pixel=None):
        self._hub = hub
        self._interval_s = interval_s
        self._next_due = 0.0
        self._color = None
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

    def update(self, percent):
        """Show the band for `percent`. Never raises.

        Writes the pixel only when the band actually changes — `show()` bit-bangs
        the pixel with interrupts off, and repainting an unchanged color would
        spend loop time to display nothing new.
        """
        if self._pixel is None:
            return
        try:
            color = band_for(percent, self._color)
            if color != self._color:
                self._pixel.fill(color)
                self._pixel.show()
                self._color = color
        except Exception:  # noqa: BLE001 - a cosmetic LED must not stop the stream
            pass

    def tick(self, now):
        """Self-driving `update` for loops with no battery slot; rate-limited.

        Don't call this per-iteration from a sampling loop — the guard alone
        measured ~0.6 accel samples/s. Use `Telemetry(on_battery=led.update)`.
        """
        if self._pixel is None or now < self._next_due:
            return
        self._next_due = now + self._interval_s
        try:
            _, percent, _ = self._hub.read_battery()
        except Exception:  # noqa: BLE001 - a bad read must not stop the stream
            return
        self.update(percent)
