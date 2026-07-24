---
name: collect-session
description: The data-collection runbook — capture a recording, eyeball pose-quality in annotate, mark standardized trials with the label vocabulary, then run metrics. Use when recording a therapy session, labeling trials, or preparing a session for the motor-metrics pipeline.
---

# Collect and label a therapy session

The end-to-end pipeline that turns a live capture into labeled trials the `motor_metrics` package can
measure. Each step is an existing CLI; the detail behind each lives in that package's `CLAUDE.md`.

> **Review note:** this runbook is synthesized from the existing `recording` → `annotate` →
> `motor_metrics` CLIs, not from an external clinical protocol. Correct the trial-marking guidance to
> match how sessions are actually administered.

## 1. Record

Capture face-blurred video + pose landmarks to an HDF5 file:

```bash
pixi run python -m recording.main --output session.hdf5   # add --video session.mp4 for a parallel mp4
```

Or record while watching the live Rerun view (optionally alongside the Feather Sense IMU):

```bash
pixi run rerun --record session.hdf5 [--feather]
```

Recording always redacts faces before persisting. Only the **minimal raw** signals are stored;
everything else is derived on read.

## 2. Eyeball pose quality, then mark trials in `annotate`

```bash
pixi run annotate session.hdf5
```

- Keep the pose overlay on (`p`, default on). **Dimmed limbs = low-visibility / extrapolated
  landmarks** — the QC signal to watch: a trial spent mostly dimmed will floor its `coverage` and
  the metrics will (correctly) return NaN.
- Scrub with `,`/`.` (frame), `<`/`>` (~1 s), `space` (play/pause).
- Mark a trial: `i` in, `o` out, then type the label at the terminal prompt. `x` deletes the nearest
  segment. Edits save immediately.
- If the middle of a hold is untrustworthy, mark **two** trials around it (or an `exclude` span) —
  `segments.py` drops anything overlapping an `exclude` whole rather than inventing a trimmed
  boundary.

### Label vocabulary

Labels are free text following `exercise[;key=value]*` (parsed read-side only, in
`motor_metrics.labels`). Examples:

- `sit_hold;arms=free;gmfm=23`
- `sit_hold;arms=prop`
- `crawl;side=left`
- `supported_stand`

`arms=free`, `side=`, `gmfm=` etc. are **assertions in the label**, not detections. Watch spelling —
a typo like `arms=freee` becomes its own `groupby` bucket and silently splits a baseline (it surfaces
as a `warnings` entry, not an error).

## 3. Run metrics

```bash
pixi run metrics session.hdf5 [--csv out.csv] [--exercise sit_hold] [--window-s N]
```

Use `--window-s N` to compare holds of unequal length fairly (`path_length_m` is
duration-confounded). See the `metrics-report` skill for the cross-session trend, which is the point
given there is no external GMFM score to calibrate against.
