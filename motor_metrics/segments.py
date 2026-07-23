"""Turn a recording's labeled annotations into frame spans a metric can run on.

An annotation is a labeled interval in **milliseconds** on the recording timeline; a
metric needs **frame indices**. Both live on the same clock — ``Annotation.start_ms``
and ``Recording.timestamps_ms`` are written from the same source — so this is a plain
``searchsorted`` with no clock alignment anywhere. (That is exactly the property the
Feather Sense IMU does *not* have: its timestamps are a device clock counting from board
boot, and the offset between the two is not stored in the recording. It is why a
camera-only v1 needs no alignment work, and why fusing the IMU later starts by fixing
that.)

Trials are marked by hand in the ``annotate`` tool using the
:mod:`motor_metrics.labels` vocabulary.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from recording.annotations import Annotation

from .labels import ParsedLabel, parse_label


@dataclass(frozen=True)
class Span:
    """A bare ``[start, stop)`` frame span, for metrics that read no label.

    :class:`Segment` carries an annotation and a parsed label because a trial *is* a
    labeled thing. But ``hold_metrics`` and ``crawl_metrics`` only ever read ``start``
    and ``stop`` — the label reaches them through their own arguments, and only
    ``transition_metrics`` reaches into ``seg.parsed``. So a caller that has a frame
    range but no annotation (an internal sub-range, or a live rolling window with no
    annotator at all) has something honest to pass, instead of fabricating an
    ``Annotation`` that no human ever marked.
    """

    start: int
    stop: int

    @property
    def n_frames(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class Segment:
    """One trial: its annotation, its parsed label, and its ``[start, stop)`` frame span."""

    ann: Annotation
    parsed: ParsedLabel
    start: int
    stop: int

    @property
    def exercise(self) -> str:
        return self.parsed.exercise

    @property
    def n_frames(self) -> int:
        return self.stop - self.start


def frame_span(timestamps_ms, start_ms: int, end_ms: int) -> tuple[int, int]:
    """Frame indices ``[start, stop)`` covering the closed time interval ``[start_ms, end_ms]``.

    The interval is closed because the annotator marks in- and out-points *on frames*
    they are looking at, so the frame under the out-point is part of the trial.

    A span that covers no frames (an annotation outside the recording) returns
    ``start == stop`` rather than raising; the consuming metric reports it as zero
    coverage, which is the honest reading of a mis-marked annotation.
    """
    ts = np.asarray(timestamps_ms)
    start = int(np.searchsorted(ts, start_ms, side="left"))
    stop = int(np.searchsorted(ts, end_ms, side="right"))
    return start, max(start, stop)


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start <= b_end and b_start <= a_end


def segments(rec, exercise: Optional[str] = None) -> list[Segment]:
    """Parseable trials in ``rec``, optionally filtered to one ``exercise``.

    Dropped, in both cases silently and by design:

    - **Unparseable labels.** Recordings predate this vocabulary and are full of free
      text; they are not trials and must not become one by accident.
    - **Anything overlapping an ``exclude`` segment.** ``exclude`` means "do not trust
      this stretch of video", and a hold whose middle is untrustworthy is not a shorter
      valid hold — it is an invalid one. The trial is dropped whole rather than trimmed:
      trimming would silently change what a trial *was*, and a trial interrupted in the
      middle would have to be split into two, inventing a boundary the annotator never
      marked. To keep the usable parts, mark them as two trials around the excluded
      stretch — which is a few keystrokes in the ``annotate`` tool.

    ``calib`` segments are returned (:func:`motor_metrics.signals.estimate_up` needs
    them); ``exclude`` segments are not, being an instruction rather than a trial.
    """
    parsed: list[tuple[Annotation, ParsedLabel]] = []
    for ann in rec.annotations:
        p = parse_label(ann.label)
        if p is not None:
            parsed.append((ann, p))

    cuts = [a for a, p in parsed if p.exercise == "exclude"]

    out: list[Segment] = []
    for ann, p in parsed:
        if p.exercise == "exclude":
            continue
        if exercise is not None and p.exercise != exercise:
            continue
        if any(_overlaps(ann.start_ms, ann.end_ms, c.start_ms, c.end_ms) for c in cuts):
            continue
        start, stop = frame_span(rec.timestamps_ms, ann.start_ms, ann.end_ms)
        out.append(Segment(ann=ann, parsed=p, start=start, stop=stop))
    return out
