---
name: metrics-report
description: Produce motor-metrics output — per-trial tables for one recording, the cross-session trend across many recordings, and the notebook for interactive exploration. Use when reporting metrics, comparing sessions over time, or QC-ing the vertical reference before trusting hold numbers.
---

# Produce a metrics report

Turn labeled recordings into the numbers that show change over time. Metrics are computed **offline**
and **derived on read** — never frozen into the `.h5`. Detail in `motor_metrics/CLAUDE.md`.

## Single recording — per-trial table

```bash
pixi run metrics session.hdf5 [--csv out.csv] [--exercise sit_hold] [--window-s N]
```

- One row per labeled trial; columns are the **union** across exercises, so different exercises
  concatenate into one table with no per-exercise holes.
- Label params are prefixed `p_` (e.g. `p_arms`); `warnings` carries the label QC.
- `--window-s N` makes holds of unequal length comparable (`path_length_m` is duration-confounded —
  prefer `mean_velocity_mps` or a common window).

## Across sessions — the trend (the whole point)

`session_table(paths)` builds the cross-session view; there is no external GMFM score to calibrate
against, so the within-child trend is the deliverable. Point it at the recordings in time order,
e.g. via the notebook (below) or:

```bash
pixi run python -c "from motor_metrics.report import session_table; print(session_table(['s1.hdf5','s2.hdf5','s3.hdf5']).to_string())"
```

## Interactive exploration — notebook

```bash
pixi run notebook
```

Open `notebooks/motor_metrics.ipynb`. Its order is deliberate: label inventory + coverage QC → **the
vertical-reference diagnostic (check it before trusting any hold number)** → per-exercise tables →
sway/limb plots → cross-session trend.

## Reading caveats (don't report a number blind)

- Metric magnitudes carry a fixed pipeline **bias** (the `derive.py` filter's derivative gain) — they
  cancel within-child but are **not comparable to any other pipeline**.
- Ellipse area reads ~0 for one-axis rocking — read it next to the ML/AP RMS.
- SPARC's absolute value is a pipeline artifact — read it only against itself, and prefer the median
  of several transitions.
- `crawl` speed is in **image widths/s, never metres**; the deliverable is the *pattern* (cadence,
  reciprocity, leg favoring), not distance.
