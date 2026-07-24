"""Draw a :class:`motor_metrics.live.LiveMetrics` readout onto a BGR frame.

The OpenCV half of the live surface; :meth:`rerun_viewer.viewer.PoseRerunLogger.
log_live_metrics` is the other. Both render the same computed readout — the metrics are
never recomputed per surface — so a number on the video and the same number on a Rerun
plot cannot disagree.

Kept out of :mod:`motor_metrics.live` so that module stays pure computation and imports
no ``cv2``, and out of :mod:`pose_estimation.draw` because that is the generic skeleton
drawer and this knows the metric vocabulary. Text is drawn plain (a single
``cv2.putText``, cyan), matching ``pose_estimation.main.draw_angles`` exactly rather than
using the outlined :func:`~pose_estimation.draw.put_text` — the black outline reads as a
drop shadow the joint-angle overlay does not have.

**Blanking is the point of the layout.** A metric whose value is NaN renders as ``--``,
never as a stale number: coverage below the gate means the tracker has lost the child,
and a figure left on screen then reads as a measurement of him. ``coverage`` is drawn
first, and in a color keyed to whether it clears the gate, so the reason for the dashes
is always on screen next to them.

**The sit-hold steadiness meter** (:func:`sit_steadiness` + :func:`_draw_meter`) is the
one non-numeric element: a red→green fill bar giving good-vs-bad feedback on a continuum,
the first piece of the child-facing display. Its honesty lives entirely in what it reads —
the trunk's deviation from its *own* rolling baseline, never an absolute upright angle or a
"good posture" threshold. See :func:`sit_steadiness`.

**ASCII only.** ``cv2.putText`` renders the Hershey fonts, which have no glyphs beyond
ASCII — a ``Δ`` or ``°`` comes out as garbage rather than as nothing, so it is not a
cosmetic issue. Hence ``trunk dev`` and ``deg``. The Rerun view names are free to use
Unicode; that path draws with real fonts.
"""

import cv2
import numpy as np

from pose_estimation.draw import FONT

from .live import MIN_COVERAGE


def _text(frame, s, org, scale, color) -> None:
    """Plain, un-outlined text — the same single ``cv2.putText`` call as
    ``pose_estimation.main.draw_angles``, so the two overlays look identical.

    Deliberately *not* :func:`pose_estimation.draw.put_text`: that helper lays a thick
    black copy under every glyph for legibility over arbitrary footage, which reads as a
    drop shadow the joint-angle overlay does not have.
    """
    cv2.putText(frame, s, org, FONT, scale, color, 1, cv2.LINE_AA)


# Cyan, matching pose_estimation.main.draw_angles so the two overlays read as one
# family. The coverage/warn colors stay distinct on purpose (see below).
_TEXT_COLOR = (255, 255, 0)
_OK_COLOR = (120, 255, 120)
_WARN_COLOR = (80, 200, 255)
_BAD_COLOR = (80, 80, 255)

_MARGIN = 12
_LINE_H = 22
_SCALE = 0.5
# Wider than the longest label in _rows, so a value never butts up against its own
# label. `cycles L/R` is exactly 10 characters and did.
_LABEL_W = 12

_BAR_W = 200
_BAR_H = 14
_BAR_TRACK = (50, 50, 50)
_BAR_BORDER = (200, 200, 200)

#: Trunk deviation, in degrees, at which the sit-hold steadiness meter reads empty. This
#: is a **display/game tolerance the caregiver tunes, not a validated clinical threshold**
#: — the same distinction ``live.py`` draws about ``MIN_COVERAGE``. It sets how twitchy the
#: bar is, and changing it cannot change any recorded number.
STEADINESS_TOL_DEG = 10.0


def sit_steadiness(metrics, tol_deg: float = STEADINESS_TOL_DEG) -> float | None:
    """How close the trunk is to its *own* rolling baseline, in ``[0, 1]``, or ``None``.

    ``1.0`` = on the window's median lean (steadiest); ``0.0`` = ``tol_deg`` or more away
    from it. ``None`` when this is not a valid hold readout, so the caller draws nothing
    rather than a stale bar.

    This is the honest sit-hold continuum, and the honesty is entirely in *what it reads*.
    It is built on ``live_trunk_angle_delta_deg`` — the current lean minus the window's own
    median — **not** on an absolute upright angle (which would inherit ``WORLD_UP``'s
    level-camera assumption, exactly what a propped-up phone breaks) and **not** on a "good
    posture" threshold (which would be the loss-of-posture criterion ``hold.py`` refuses to
    invent). It measures *steadiness relative to the child's own baseline*, which is the
    one thing a single tilted camera can honestly report — see the README's child-facing
    display note.
    """
    if metrics is None or metrics.live_mode != "hold" or not metrics.live_valid:
        return None
    delta = metrics.live_trunk_angle_delta_deg
    if delta is None or delta != delta:  # NaN
        return None
    return float(max(0.0, 1.0 - abs(float(delta)) / tol_deg))


def _quality_color(q: float) -> tuple[int, int, int]:
    """BGR on a red (``q=0``) → yellow (``0.5``) → green (``q=1``) continuum."""
    q = min(1.0, max(0.0, float(q)))
    if q < 0.5:
        return (0, int(round(510 * q)), 255)  # red -> yellow
    return (0, 255, int(round(510 * (1.0 - q))))  # yellow -> green


def _draw_meter(frame, label: str, q: float, org: tuple[int, int]) -> None:
    """A labeled fill bar for ``q`` in ``[0, 1]``: length *and* color both encode it.

    Double-encoding is deliberate — the child watches the bar fill and greens as they
    steady, and the color survives if the fill is hard to judge at a glance.
    """
    x, y = org
    color = _quality_color(q)
    _text(frame, label, (x, y), _SCALE, color)
    top, bot = y + 6, y + 6 + _BAR_H
    cv2.rectangle(frame, (x, top), (x + _BAR_W, bot), _BAR_TRACK, -1)
    fill = int(round(_BAR_W * min(1.0, max(0.0, q))))
    if fill > 0:
        cv2.rectangle(frame, (x, top), (x + fill, bot), color, -1)
    cv2.rectangle(frame, (x, top), (x + _BAR_W, bot), _BAR_BORDER, 1)


def _fmt(value, digits: int = 3, suffix: str = "") -> str:
    """A number, or ``--`` when it is NaN. See the module docstring on why."""
    if value is None or (isinstance(value, float) and value != value):
        return "--"
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}{suffix}"
    return f"{float(value):.{digits}f}{suffix}"


def _rows(metrics) -> list[tuple[str, str]]:
    """The ``(label, value)`` lines for this readout's mode."""
    if metrics.live_mode == "crawl":
        return [
            ("cadence", _fmt(metrics.live_cadence_cpm, 1, " cpm")),
            (
                "  L / R",
                f"{_fmt(metrics.live_cadence_cpm_left, 1)} / "
                f"{_fmt(metrics.live_cadence_cpm_right, 1)}",
            ),
            (
                "cycles L/R",
                f"{_fmt(metrics.live_n_cycles_left)} / {_fmt(metrics.live_n_cycles_right)}",
            ),
            ("period CV", _fmt(metrics.live_cycle_period_cv, 3)),
        ]
    return [
        ("sway RMS", _fmt(metrics.live_sway_rms_m, 4, " m")),
        (
            "  ML / AP",
            f"{_fmt(metrics.live_sway_ml_rms_m, 4)} / {_fmt(metrics.live_sway_ap_rms_m, 4)}",
        ),
        ("sway vel", _fmt(metrics.live_sway_velocity_mps, 4, " m/s")),
        ("trunk dev", _fmt(metrics.live_trunk_angle_delta_deg, 1, " deg")),
    ]


def draw_live_metrics(frame, metrics, origin: tuple[int, int] | None = None) -> None:
    """Draw ``metrics`` onto ``frame`` in place, top-left by default.

    Never raises on a blanked or partially-NaN readout — it is called from inside the
    capture loop, where an exception costs the session.
    """
    if frame is None or metrics is None:
        return
    x, y = origin if origin is not None else (_MARGIN, _MARGIN + _LINE_H)

    cov = metrics.live_coverage
    cov_color = _OK_COLOR if cov >= MIN_COVERAGE else _BAD_COLOR
    header = f"{metrics.live_mode}  {metrics.live_window_s:g}s window"
    _text(frame, header, (x, y), _SCALE, _TEXT_COLOR)
    y += _LINE_H

    _text(frame, f"coverage  {_fmt(cov, 2)}", (x, y), _SCALE, cov_color)
    y += _LINE_H
    _text(frame, f"tracked   {_fmt(metrics.live_tracked_s, 1, ' s')}", (x, y), _SCALE,
             _TEXT_COLOR)
    y += _LINE_H

    for label, value in _rows(metrics):
        _text(frame, f"{label:<{_LABEL_W}s}{value}", (x, y), _SCALE, _TEXT_COLOR)
        y += _LINE_H

    # Sit-hold steadiness meter: the continuum feedback. None for crawl or a blanked
    # readout, so it simply does not draw then. See sit_steadiness on why this is honest.
    q = sit_steadiness(metrics)
    if q is not None:
        _draw_meter(frame, f"steady {int(round(q * 100))}%", q, (x, y))
        y += _LINE_H + _BAR_H

    # The vertical reference is an assumption, not a measurement, and "less controlled"
    # is exactly where a propped-up camera breaks it. Say which one is in force rather
    # than letting a tilted session look identical to a level one.
    if metrics.live_up_source not in ("n/a", ""):
        _text(frame, f"up: {metrics.live_up_source}", (x, y), 0.42, _WARN_COLOR)
