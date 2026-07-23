"""Draw a :class:`motor_metrics.live.LiveMetrics` readout onto a BGR frame.

The OpenCV half of the live surface; :meth:`rerun_viewer.viewer.PoseRerunLogger.
log_live_metrics` is the other. Both render the same computed readout — the metrics are
never recomputed per surface — so a number on the video and the same number on a Rerun
plot cannot disagree.

Kept out of :mod:`motor_metrics.live` so that module stays pure computation and imports
no ``cv2``, and out of :mod:`pose_estimation.draw` because that is the generic skeleton
drawer and this knows the metric vocabulary. It reuses that module's
:func:`~pose_estimation.draw.put_text` rather than carrying its own outlined-text
helper.

**Blanking is the point of the layout.** A metric whose value is NaN renders as ``--``,
never as a stale number: coverage below the gate means the tracker has lost the child,
and a figure left on screen then reads as a measurement of him. ``coverage`` is drawn
first, and in a color keyed to whether it clears the gate, so the reason for the dashes
is always on screen next to them.

**ASCII only.** ``cv2.putText`` renders the Hershey fonts, which have no glyphs beyond
ASCII — a ``Δ`` or ``°`` comes out as garbage rather than as nothing, so it is not a
cosmetic issue. Hence ``trunk dev`` and ``deg``. The Rerun view names are free to use
Unicode; that path draws with real fonts.
"""

import numpy as np

from pose_estimation.draw import put_text

from .live import MIN_COVERAGE

_OK_COLOR = (120, 255, 120)
_WARN_COLOR = (80, 200, 255)
_BAD_COLOR = (80, 80, 255)
_LABEL_COLOR = (220, 220, 220)
_VALUE_COLOR = (255, 255, 255)

_MARGIN = 12
_LINE_H = 22
_SCALE = 0.5
# Wider than the longest label in _rows, so a value never butts up against its own
# label. `cycles L/R` is exactly 10 characters and did.
_LABEL_W = 12


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
    put_text(frame, header, (x, y), _SCALE, _LABEL_COLOR)
    y += _LINE_H

    put_text(frame, f"coverage  {_fmt(cov, 2)}", (x, y), _SCALE, cov_color)
    y += _LINE_H
    put_text(frame, f"tracked   {_fmt(metrics.live_tracked_s, 1, ' s')}", (x, y), _SCALE,
             _LABEL_COLOR)
    y += _LINE_H

    for label, value in _rows(metrics):
        put_text(frame, f"{label:<{_LABEL_W}s}{value}", (x, y), _SCALE, _VALUE_COLOR)
        y += _LINE_H

    # The vertical reference is an assumption, not a measurement, and "less controlled"
    # is exactly where a propped-up camera breaks it. Say which one is in force rather
    # than letting a tilted session look identical to a level one.
    if metrics.live_up_source not in ("n/a", ""):
        put_text(frame, f"up: {metrics.live_up_source}", (x, y), 0.42, _WARN_COLOR)
