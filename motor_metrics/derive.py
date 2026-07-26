"""Resampling, smoothing and differentiation for landmark trajectories.

Sway velocity and movement smoothness both need a *derivative* of a landmark path, and
landmark paths are noisy: differentiating them raw amplifies the noise more than the
signal. So everything routes through one filter chain — resample to a uniform grid,
then Savitzky-Golay.

**Why the uniform grid.** Camera frames land at roughly :data:`FS` but not exactly:
``Recording.fps()`` is instantaneous ``1000/diff(timestamps_ms)`` and wobbles. A
Savitzky-Golay derivative takes a single scalar ``delta`` (one sample spacing) and
SPARC's FFT assumes uniform spacing outright; both quietly produce wrong numbers on a
jittery timebase rather than complaining.

The grid is 15 Hz because that is what the capture hardware actually sustains — the phone
measured 15-20 fps and the desktop CLIs now request 15 to match. Resampling *up* to a
faster grid than the camera delivers does not add information: it manufactures samples
that are exact linear combinations of their neighbours, and the filter below then averages
over samples that are partly each other, at which point the gain table stops describing
the chain.

**Why the constants are constants.** :data:`FS`, :data:`WINDOW_S` and :data:`POLY` are
module-level and not meant to be passed per call. A SPARC score or a sway velocity is
only comparable against another one computed through an identical filter chain, and
these numbers get compared across months of sessions. Exposing them as tweakable
per-call defaults invites two baselines that differ because of a filter setting rather
than because of Remy. If a constant must change, change it once, here — and treat every
number computed before the change as belonging to a different scale.

**:data:`FS` and :data:`WINDOW_S` are a pair.** The window is specified in seconds but
applied in samples, so halving the rate halves the sample count: at ``WINDOW_S = 0.25`` a
15 Hz grid gives ``int(0.25 * 15) = 3`` samples, and three samples fit a quadratic
*exactly*. The filter would become the identity and stop filtering, without raising, and
every derivative would silently become a raw central difference. Only two non-degenerate
windows are reachable at 15 Hz — 5 samples (0.333 s) and 7 (0.467 s) — and 0.35 selects
the 5, deliberately sitting mid-bracket rather than at the 0.3333 boundary where one
float rounding would collapse it. See ``WindowLengthTests``, which pins both directions.

**What the chain costs.** At the shipped constants (15 Hz, 0.333 s window, quadratic)
the first-derivative gain rolls off with movement frequency — measured against an
analytic sine, and pinned by ``test_derivative_gain_rolls_off_with_frequency``::

    0.25 Hz  0.994     1.0 Hz  0.904     2.0 Hz  0.652
    0.50 Hz  0.975     1.5 Hz  0.793     3.0 Hz  0.339

So postural sway (well under 1 Hz) is measured essentially unattenuated, while the fast
content of a transition is damped by tens of percent. This is a *bias*, not noise: it is
identical for every trial through the same chain, so it cancels in the within-child
comparisons this package is for, and it is one more reason those constants must not
drift between sessions. It also means these magnitudes are not comparable to figures
produced by any other filter chain, published or otherwise.

The window spans 0.333 s rather than the 0.233 s it spanned at 30 Hz, so this table rolls
off sooner than its predecessor: what was 0.95 at 1 Hz is 0.90, and 3 Hz is damped to a
third rather than to two thirds. Sway is unaffected in practice; fast transition content
is measured more conservatively than it was. The alternative 7-sample window was rejected
on measurement — it reproduces the old table compressed *exactly* 2x in frequency, which
would have halved the package's usable bandwidth to buy noise rejection this chain does
not need.

Nothing here interpolates across a dropout. A gap in tracking is a gap; bridging it
would invent the movement that happened inside it. Callers pick a contiguous good run
first (:func:`motor_metrics.quality.longest_run`) and resample within it.
"""

import numpy as np
from scipy.signal import savgol_filter

FS = 15.0  # Hz, the uniform grid every derived quantity is computed on
WINDOW_S = 0.35  # Savitzky-Golay window, in seconds
POLY = 2  # Savitzky-Golay polynomial order


def window_length(fs: float = FS, window_s: float = WINDOW_S, poly: int = POLY) -> int:
    """Savitzky-Golay window in samples: odd, and greater than ``poly``, as scipy requires.

    **The degenerate case is real and silent.** ``poly + 1`` samples fit a polynomial of
    order ``poly`` *exactly*, so a window that small is an interpolation, not a fit:
    ``deriv=0`` returns the input untouched and ``deriv=1`` collapses to a plain central
    difference. Nothing raises — the chain simply stops filtering. At ``POLY=2`` that is
    ``w == 3``, which ``window_s=0.25`` produces at any ``fs`` in [12, 20), i.e. exactly the
    range a 15 Hz grid sits in. :data:`WINDOW_S` is therefore paired with :data:`FS` and the
    two must be changed together; the pairing is pinned by ``WindowLengthTests``.
    """
    w = max(int(window_s * fs), poly + 1)
    return w + 1 if w % 2 == 0 else w


def resample_uniform(t_ms, x, fs: float = FS):
    """Linearly resample ``x`` from timestamps ``t_ms`` onto a uniform ``fs`` grid.

    ``x`` is ``(N,)`` or ``(N, K)``; returns ``(t_s, x_uniform)`` with ``t_s`` in
    seconds from the first sample. Returns empty arrays when fewer than two samples are
    given — there is nothing to interpolate between, and a caller asking for a
    derivative of one point should get an empty answer, not an exception.

    NaN inputs propagate to their neighbouring intervals rather than being dropped:
    silently interpolating across missing frames would fabricate the very movement the
    tracker failed to see.
    """
    t = np.asarray(t_ms, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    single = x.ndim == 1
    cols = x.reshape(-1, 1) if single else x

    if t.size < 2:
        return np.empty(0), np.empty(0) if single else np.empty((0, cols.shape[1]))

    t_s = (t - t[0]) / 1000.0
    span = float(t_s[-1])
    count = int(np.floor(span * fs)) + 1
    if count < 2:
        return np.empty(0), np.empty(0) if single else np.empty((0, cols.shape[1]))

    grid = np.arange(count) / fs
    out = np.stack([np.interp(grid, t_s, col) for col in cols.T], axis=1)
    return grid, (out[:, 0] if single else out)


def smooth(x, fs: float = FS, *, window_s: float = WINDOW_S, poly: int = POLY, deriv: int = 0):
    """Savitzky-Golay filter (or its ``deriv``-th derivative) along axis 0.

    Assumes ``x`` is already on a uniform ``fs`` grid — pass it through
    :func:`resample_uniform` first. With ``deriv=1`` the result is in units per second.

    Returns an all-NaN array of the input's shape when the input is shorter than the
    window, rather than raising as ``savgol_filter`` does. Short segments are ordinary
    here — a two-second sit is a real trial, and a two-frame one is a mis-marked
    annotation. Neither should take down a report table that has 40 other rows in it.
    """
    x = np.asarray(x, dtype=np.float64)
    w = window_length(fs, window_s, poly)
    if x.shape[0] < w:
        return np.full(x.shape, np.nan)
    return savgol_filter(x, w, poly, deriv=deriv, delta=1.0 / fs, axis=0)


def velocity(t_ms, p, fs: float = FS):
    """Velocity of a path: ``(t_s, components, speed)``.

    ``p`` is ``(N, K)`` positions at ``t_ms``. Returns the uniform time grid, the
    ``(M, K)`` per-axis velocity, and the ``(M,)`` scalar speed (the norm), all in
    units per second. Speed is NaN throughout when the segment is shorter than the
    smoothing window.
    """
    t_s, uniform = resample_uniform(t_ms, p, fs=fs)
    if t_s.size == 0:
        k = np.asarray(p).reshape(len(np.asarray(p)), -1).shape[1]
        return np.empty(0), np.empty((0, k)), np.empty(0)
    v = smooth(uniform, fs=fs, deriv=1)
    return t_s, v, np.linalg.norm(v, axis=1)
