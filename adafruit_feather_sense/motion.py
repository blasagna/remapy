"""Host-side derivation of gravity / linear acceleration from raw acceleration.

The LSM6DS does no on-chip fusion, so the board can only report *total*
acceleration (which includes gravity, ~9.8 m/s^2 on one axis at rest). It
streams exactly that and nothing more; this module reconstructs the two derived
streams on the host:

- ``gravity``      — a single-pole low-pass estimate of the gravity vector,
- ``linear_accel`` — ``accel - gravity``, the motion-only component (~0 at rest).

Deriving here rather than on the board buys two things: the device spends its
loop on sampling (three encoded frames per accel read became one), and
:data:`GRAVITY_TAU_S` becomes a **read-time** choice — an existing recording can
be re-derived with a different time constant, which is impossible once a filter
has been baked into the capture.

Accuracy note (unchanged from the board implementation): this is a lightweight
estimate. Sustained (non-transient) linear acceleration slowly bleeds into the
gravity estimate; a gyro-fused orientation filter (e.g. Madgwick/Mahony) would
remove that at more compute cost — cheap to reconsider now that it runs on a
host CPU rather than an nRF52840.

Host-only, but dependency-free (no numpy) so the live stream path and the
offline reader can share one implementation.
"""

# Time constant (seconds) of the low-pass filter that estimates the gravity
# vector. Larger = steadier gravity, slower to follow a reorientation; smaller =
# follows tilt faster but leaks more motion into it.
GRAVITY_TAU_S = 0.5


class GravityFilter:
    """Single-pole low-pass estimate of the gravity vector.

    Feed raw accelerometer samples in device-timestamp order; each
    :meth:`update` returns ``(gravity, linear)``. The filter coefficient adapts
    to the real elapsed time between samples, so it is robust to the jittery
    sample spacing the board's cooperative loop produces.

    Timing comes from the *device* timestamp rather than a host clock, so the
    filter sees the true sample spacing — unaffected by host scheduling or, for
    a recording, by playback speed. A non-positive ``dt`` (duplicate timestamps,
    or the u32 device clock wrapping at ~49.7 days) holds the estimate rather
    than corrupting it.
    """

    def __init__(self, tau_s=GRAVITY_TAU_S):
        self.tau_s = float(tau_s)
        self._gravity = None
        self._t_s = None

    def update(self, timestamp_ms, accel_xyz):
        """Fold one raw (x, y, z) accel sample in; return ``(gravity, linear)``."""
        t_s = timestamp_ms / 1000.0
        total = tuple(float(v) for v in accel_xyz)
        if self._gravity is None:
            # Seed on the first sample: assume the board starts at rest, so all
            # of the first reading is gravity (linear then starts at ~0).
            self._gravity = list(total)
        else:
            dt = t_s - self._t_s
            alpha = dt / (self.tau_s + dt) if dt > 0 else 0.0
            self._gravity = [g + alpha * (v - g) for g, v in zip(self._gravity, total)]
        self._t_s = t_s
        gravity = tuple(self._gravity)
        linear = tuple(v - g for v, g in zip(total, gravity))
        return gravity, linear

    def reset(self):
        """Forget the estimate (next update re-seeds)."""
        self._gravity = None
        self._t_s = None


def derive_motion(timestamps_ms, accel_values, tau_s=GRAVITY_TAU_S):
    """Derive a whole recorded accel stream at once.

    ``accel_values`` is an ``(M, 3)`` sequence of raw (x, y, z) samples aligned
    with ``timestamps_ms``. Returns ``(gravity, linear)``, each a list of M
    (x, y, z) tuples.

    The filter is a sequential IIR — each output depends on the previous one —
    so this is a loop by construction and does not vectorise.
    """
    filt = GravityFilter(tau_s)
    gravity = []
    linear = []
    for ts, xyz in zip(timestamps_ms, accel_values):
        g, lin = filt.update(ts, xyz)
        gravity.append(g)
        linear.append(lin)
    return gravity, linear
