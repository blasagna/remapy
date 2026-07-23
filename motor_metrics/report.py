"""One row per trial: the table everything else in this package feeds.

Metrics are **recomputed on read and never written back into the recording**. That is the
same rule the recording format itself is built on (``recording/recorder.py`` stores only
minimal raw signals; ``Recording.angles()`` and ``Recording.motion()`` re-derive the rest),
and it matters more here than elsewhere: every number in this table is a function of the
filter constants in :mod:`motor_metrics.derive`, and freezing derived values into the
``.h5`` would strand them at whatever those constants happened to be that week. The raw
landmarks are the record; a metric is an opinion about them.
"""

from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from recording.reader import Recording

from .crawl import crawl_metrics
from .hold import hold_metrics
from .labels import label_warnings
from .quality import Gate
from .segments import segments
from .signals import WORLD_UP
from .transition import transition_metrics

# Adapters giving every metric one dispatch signature. They exist to make each "this
# argument does not apply here" an explicit, explained decision: forwarding **kwargs
# blindly instead is what let `--window-s` reach transition_metrics() and raise TypeError
# on any session containing a transition.


def _hold(rec, seg, *, up, gate, window_s):
    return hold_metrics(rec, seg, up=up, gate=gate, window_s=window_s)


def _transition(rec, seg, *, up, gate, window_s):
    """Drops ``window_s``.

    Windowing exists to make holds of unequal length comparable on duration-confounded
    path length. A transition is a discrete event, not a duration to be sampled:
    truncating one to a common prefix would cut the movement in half and make its
    smoothness a measurement of where the cut landed.
    """
    return transition_metrics(rec, seg, up=up, gate=gate)


def _crawl(rec, seg, *, up, gate, window_s):
    """Drops ``up`` and ``window_s``.

    ``up``: in prone there is no useful vertical, so a belly crawl is measured against
    the body's own long axis (see :mod:`motor_metrics.crawl`); accepting one would imply
    it meant something. ``window_s``: crawl metrics are already duration-free — cadence
    is per-minute and ``cycle_period_cv`` is dimensionless — so truncating buys nothing
    but a smaller sample.
    """
    return crawl_metrics(rec, seg, gate=gate)


# Trials that produce a metric row, and what computes them. `calib` and `exclude` are
# housekeeping and never appear.
_DISPATCH = {
    "sit_hold": _hold,
    "stand_hold": _hold,
    "transition": _transition,
    "crawl": _crawl,
}

#: The exercises that yield a metric row — a strict subset of ``labels.EXERCISES``.
TRIAL_EXERCISES = tuple(_DISPATCH)

# Identity and quality first, so a reader meets `coverage` before any number it qualifies.
_LEAD_COLUMNS = [
    "session",
    "exercise",
    "dimension",
    "label",
    "start_ms",
    "end_ms",
    "n_frames",
    "coverage",
    "duration_s",
    "tracked_s",
]


def metrics_table(
    rec,
    *,
    up: Optional[np.ndarray] = None,
    gate: Gate = Gate(),
    session: Optional[str] = None,
    window_s: Optional[float] = None,
):
    """One row per trial in ``rec``, as a pandas DataFrame.

    Columns are the union across exercise types, so a sitting row has NaN in the crawl
    columns and vice versa — the alternative is four tables that cannot be concatenated
    for a trend. Label params appear prefixed ``p_`` (``p_arms``, ``p_side``), keeping
    them clear of metric fields of the same name.

    ``window_s`` truncates **holds** to a common prefix so their path lengths compare
    fairly; the other exercises document why they ignore it. The arguments are spelled
    out rather than forwarded as ``**kwargs`` on purpose — blind forwarding is what let
    ``window_s`` reach a metric that does not take one and raise at the call site.

    ``up`` defaults to :data:`motor_metrics.signals.WORLD_UP`, which assumes a level
    camera. It is **not** auto-calibrated from a ``calib`` segment: doing so silently
    trades a known assumption for an unknown one, since a calibration pose cannot
    separate a tilted camera from a child who does not sit vertically. Check the calib
    diagnostic, then pass ``up=estimate_up(rec, calib_seg)`` if it warrants it.

    The ``warnings`` column carries :func:`motor_metrics.labels.label_warnings` — typos
    that would otherwise split a baseline into two silently-different groups.
    """
    import pandas as pd

    up = WORLD_UP if up is None else up
    rows = []
    for seg in segments(rec):
        fn = _DISPATCH.get(seg.exercise)
        if fn is None:
            continue  # `calib`: a reference pose, not a trial
        metrics = fn(rec, seg, up=up, gate=gate, window_s=window_s)
        rows.append(
            {
                "session": session,
                "exercise": seg.exercise,
                "dimension": seg.parsed.dimension,
                "label": seg.ann.label,
                "start_ms": seg.ann.start_ms,
                "end_ms": seg.ann.end_ms,
                **{f"p_{k}": v for k, v in seg.parsed.params.items()},
                **vars(metrics),
                "warnings": "; ".join(label_warnings(seg.ann.label)),
            }
        )
    return _ordered(pd.DataFrame(rows))


def session_table(
    paths: Iterable[Path | str],
    *,
    up: Optional[np.ndarray] = None,
    gate: Gate = Gate(),
    window_s: Optional[float] = None,
):
    """:func:`metrics_table` over several recordings, concatenated for a trend.

    The ``session`` column is each file's stem and is set from the path, so it is not
    accepted here. This is the cross-session view the whole package is for: with no
    external GMFM score to calibrate against, a number is only meaningful next to the
    same number from Remy's other sessions.
    """
    import pandas as pd

    frames = []
    for path in paths:
        path = Path(path)
        with Recording(path) as rec:
            frames.append(
                metrics_table(rec, up=up, gate=gate, window_s=window_s, session=path.stem)
            )
    if not frames:
        return _ordered(pd.DataFrame())
    return _ordered(pd.concat(frames, ignore_index=True))


def _ordered(df):
    """Lead columns first, params next, metrics after, warnings last."""
    if df.empty:
        return df
    lead = [c for c in _LEAD_COLUMNS if c in df.columns]
    params = sorted(c for c in df.columns if c.startswith("p_"))
    tail = ["warnings"] if "warnings" in df.columns else []
    rest = [c for c in df.columns if c not in lead + params + tail]
    return df[lead + params + rest + tail]
