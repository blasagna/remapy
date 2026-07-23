"""Annotation label vocabulary tying recorded time segments to standardized exercises.

The instruments this project draws on (GMFM-88, AIMS) score *ordinally* — a GMFM item
is 0-3, AIMS is observed/not-observed. That coarseness is the problem: a child can sit
at a "2" for a year while genuinely improving. So the item is not reimplemented here.
Instead each item defines a reproducible **trial**, and the modules alongside this one
measure the *continuous* variable underneath it.

This module is the vocabulary that says which trial a segment is. The grammar is::

    exercise[;key=value]*

    sit_hold;arms=free;support=none;gmfm=23
    transition;from=prone;to=sit;side=left
    crawl;style=belly;dir=away

It is typed by hand at ``annotate``'s terminal prompt (``annotate/main.py`` stores free
text and is deliberately **unchanged** — the vocabulary is a read-side convention only).

Two rules hold this together:

- :func:`parse_label` is **total**: it returns ``None`` for anything it does not
  recognize and never raises. Recordings predate this vocabulary and are full of free
  text like ``"walking"``; they must keep loading.
- Value checking is **separate** (:func:`label_warnings`) and advisory. A typo'd value
  must not raise mid-report, but it also must not pass silently — ``arms=freee`` would
  otherwise become its own group in a ``groupby`` and quietly split a baseline in half.

``gmfm=<item>`` is a free field the annotator copies off the score sheet. GMFM-88 item
*numbers* are deliberately not hard-coded anywhere in this package: the numbering is
behind the manual, and a wrong constant compared across months is worse than none. The
dimension mapping (:data:`DIMENSIONS`) is all the code actually needs.
"""

from typing import Mapping, NamedTuple, Optional

EXERCISES = ("sit_hold", "stand_hold", "transition", "crawl", "calib", "exclude")

# GMFM-88 dimension each exercise's trials sit in. B=sitting, C=crawling, D=standing.
# `calib` / `exclude` are housekeeping, not trials, so they map to nothing.
DIMENSIONS: dict[str, str] = {
    "sit_hold": "B",
    "transition": "B",
    "crawl": "C",
    "stand_hold": "D",
}

# Allowed params per exercise: key -> allowed values. An EMPTY tuple means the value is
# free text (`gmfm` off the score sheet, `reason` for an exclusion).
PARAMS: dict[str, dict[str, tuple[str, ...]]] = {
    "sit_hold": {
        "arms": ("free", "prop", "held"),
        "support": ("none", "trunk", "pelvis"),
        "gmfm": (),
    },
    "stand_hold": {
        "support": ("hands_held", "trunk", "furniture"),
        "gmfm": (),
    },
    "transition": {
        "from": ("prone", "sit"),
        "to": ("prone", "sit"),
        "side": ("left", "right"),
        "gmfm": (),
    },
    "crawl": {
        "style": ("belly",),
        "dir": ("left", "right", "toward", "away"),
        "gmfm": (),
    },
    "calib": {"pose": ("upright",)},
    "exclude": {"reason": ()},
}


class ParsedLabel(NamedTuple):
    """One parsed label. ``raw`` is the annotator's original text, kept verbatim."""

    exercise: str
    params: dict[str, str]
    raw: str

    @property
    def dimension(self) -> Optional[str]:
        """GMFM-88 dimension (``"B"``/``"C"``/``"D"``), or ``None`` for housekeeping labels."""
        return DIMENSIONS.get(self.exercise)


def parse_label(label: str) -> Optional[ParsedLabel]:
    """Parse ``exercise[;key=value]*``; return ``None`` if ``exercise`` is unknown.

    Never raises. The exercise and param *keys* are lower-cased (a terminal prompt
    invites fumbled case); param *values* are kept verbatim, since ``reason=`` and
    ``gmfm=`` are free text where case may carry meaning. Malformed params are dropped
    here and reported by :func:`label_warnings` rather than failing the parse — a
    half-typed label should still tell you which exercise it was.
    """
    if not label:
        return None
    parts = [p.strip() for p in str(label).split(";")]
    exercise = parts[0].lower()
    if exercise not in EXERCISES:
        return None

    params: dict[str, str] = {}
    for part in parts[1:]:
        if not part or "=" not in part:
            continue  # bare token / empty segment -> see label_warnings
        key, _, value = part.partition("=")  # split on the FIRST '=' only
        key = key.strip().lower()
        value = value.strip()
        if key:
            params[key] = value  # duplicate key: last wins -> see label_warnings
    return ParsedLabel(exercise=exercise, params=params, raw=str(label))


def format_label(exercise: str, params: Optional[Mapping[str, object]] = None, **kw) -> str:
    """Build a label string; round-trips through :func:`parse_label`.

    Params may be passed as a mapping, as keywords, or both (keywords win). The mapping
    form exists because ``transition`` uses ``from=``, which cannot be a Python keyword
    argument.

    Known keys are emitted in :data:`PARAMS` declaration order and extras alphabetically
    after them, so the same trial always formats to the same string — labels are compared
    by eye across months of sessions.
    """
    if exercise not in EXERCISES:
        raise ValueError(f"Unknown exercise {exercise!r}; expected one of {EXERCISES}.")
    merged: dict[str, object] = {**(params or {}), **kw}

    order = list(PARAMS.get(exercise, {}))
    known = [k for k in order if k in merged]
    extra = sorted(k for k in merged if k not in order)
    return ";".join([exercise] + [f"{k}={merged[k]}" for k in known + extra])


def label_warnings(label: str) -> list[str]:
    """Advisory complaints about ``label`` — typos that would quietly split a baseline.

    Returns human-readable strings, never raises. Intended for the notebook's label-QC
    pass and :mod:`motor_metrics.report`, not for the metric functions: an unrecognized
    value is a data-entry problem for a human to fix, not a reason to fail a report.

    An unparseable label yields a single warning; free-text legacy labels are the common
    case and are expected to show up here.
    """
    parsed = parse_label(label)
    if parsed is None:
        return [f"unrecognized label {str(label)!r} (not one of {', '.join(EXERCISES)})"]

    out: list[str] = []
    parts = [p.strip() for p in str(label).split(";")][1:]
    seen: set[str] = set()
    for part in parts:
        if not part:
            continue
        if "=" not in part:
            out.append(f"{parsed.exercise}: bare token {part!r} is not key=value; ignored")
            continue
        key = part.partition("=")[0].strip().lower()
        if key in seen:
            out.append(f"{parsed.exercise}: duplicate key {key!r}; last value wins")
        seen.add(key)

    allowed = PARAMS.get(parsed.exercise, {})
    for key, value in parsed.params.items():
        if key not in allowed:
            out.append(
                f"{parsed.exercise}: unknown param {key!r} "
                f"(expected {', '.join(allowed) or 'none'})"
            )
            continue
        values = allowed[key]
        if values and value.lower() not in values:
            out.append(
                f"{parsed.exercise}: {key}={value!r} not one of {', '.join(values)}"
            )
    return out
