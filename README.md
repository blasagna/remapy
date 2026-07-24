# remapy
Playing with tools inspired by Remy's therapies

**NOTE: This is a work in progress**

Remy + therapy + python = remapy

Much of this is implemented using Claude code and other coding LLMs. I build similar tools to sample sensors from embedded systems and process them in my day job. 

## Motivation

My son Remy has a rare genetic syndrome that causes global developmental
delays, in both movement and cognitive function (more at
[rareremy.org](https://www.rareremy.org)). While we pursue therapeutic development
and medical research, most of our day-to-day energy goes into physical and
occupational therapy with him.

remapy is a place to build tools that motivate Remy in those therapies and that can
track changes in his abilities before they're visible in his gross actions. If
therapeutic development advances to a clinical trial, the same tools can help
establish a baseline of his abilities before treatment and measure progress or
change afterward.

![Pose estimation on a session with Remy](img/remy_pose.png)

## References

Standard, clinically validated exercises and scoring systems this project draws on
for motor-development metrics:

- **GMFM-88 — Gross Motor Function Measure.** 88 items across five dimensions
  (lying/rolling, sitting, crawling/kneeling, standing, walking/running/jumping),
  each scored 0–3.
  [CanChild overview](https://canchild.ca/en/resources/44-gross-motor-function-measure-gmfm) ·
  [Physiopedia](https://www.physio-pedia.com/Gross_Motor_Function_Measure) ·
  [User's Manual (Mac Keith Press)](https://www.mackeith.co.uk/book/gross-motor-function-measure-gmfm-66-gmfm-88-users-manual-revised-3rd-edition/)

Not currently used:

- **PDMS-3 — Peabody Developmental Motor Scales, Third Edition.** Gross-motor
  subtests (Body Control, Body Transport, Object Control) plus fine-motor and a
  supplemental physical-fitness subtest.
  [Pearson](https://www.pearsonassessments.com/en-us/Store/Professional-Assessments/Motor-Sensory/Peabody-Developmental-Motor-Scales,-Third-Edition/p/P100049000) ·
  [WPS](https://www.wpspublish.com/peabody-developmental-motor-scales-third-edition.html) ·
  [PAR](https://www.parinc.com/products/PDMS-3)
- **AIMS — Alberta Infant Motor Scale.** 58 observational items across prone,
  supine, sitting, and standing positions, norm-referenced from birth to 18 months.
  [Physiopedia](https://www.physio-pedia.com/Alberta_Infant_Motor_Scale_(AIMS)) ·
  [Score sheets (Elsevier)](https://www.us.elsevierhealth.com/alberta-infant-motor-scale-score-sheets-aims-9780323798426.html)

## TODO

- [x] increase IMU sampling rate to at least 50, ideally 100 Hz — **100 Hz over USB serial, 50 Hz
      over BLE**, both measured at 100 % of nominal
- [x] measure streaming throughput compared to nominal sampling rates — `--stats` now reports a
      device-clock rate + max gap alongside host arrival rate; this is how the above was verified,
      and it caught the rate silently decaying with board uptime
- [x] baseline IMU signal stats at rest — see
      [Sensors](adafruit_feather_sense/README.md#sensors) (accel RMS σ ≈ 0.0102 m/s², gyro
      ≈ 0.0021 rad/s at the shipped ODR 208)
- [x] implement metrics from standard exercises and scoring defined in GMFM-88 — **four trials
      (sitting hold, sit↔prone transition, belly crawl, supported standing), offline, camera-only**.
      The instruments score *ordinally* (GMFM items are 0–3), which is too coarse to show change:
      Remy can sit at a "2" for a year while genuinely improving. So the items are not
      reimplemented — each one defines a reproducible *trial*, and `motor_metrics/` measures the
      **continuous variable underneath it** (hold duration and postural sway; transition smoothness
      via SPARC; crawl cadence and left–right reciprocity). Label trials with `pixi run annotate`,
      then `pixi run metrics` / `notebooks/motor_metrics.ipynb`
- [x] compute metrics in real time (GUI overlay) as well as offline — **the tilt-robust,
      trigger-free subset**: `pixi run live` (sway + trunk lean) / `pixi run live-crawl` (cadence +
      cycle variability, for **both arms and legs** — the crawl overlay leads with the legs and a
      "favors one leg" readout, since that is where Remy's signal is), or `--live-metrics` on `pose`
      and `rerun`. `motor_metrics/live.py` feeds a
      rolling window to the *same* `hold_metrics`/`crawl_metrics` the offline table uses. Cost was
      never the obstacle (~3 ms per recompute against a 33 ms frame); the constraints are that the
      Savitzky-Golay fit leaves the last 3 samples extrapolated — so the readout is deliberately
      100 ms old, where it equals the offline value *exactly* — and that without an annotator there
      is no honest trial boundary, so the window is fixed and **nothing infers movement onset**.
      SPARC, submovement counts and any duration metric are therefore still offline-only: they need
      an onset/offset the code is not entitled to invent. Live values never enter the `.h5` or the
      offline table (different window, different measurement)
- [ ] child-facing live display — the same readout driving something motivating rather than a
      numeric overlay. Note "held for N seconds" is *not* available (that is the loss-of-posture
      inference `motor_metrics` refuses); "coverage green and trunk within X of its own baseline" is.
      **First piece landed:** `pixi run live` now draws a sit-hold *steadiness meter* — a
      red→green fill bar reading good-vs-bad on a continuum, built on the trunk's deviation from its
      *own* rolling baseline (so a tilted camera shifts the baseline, not the score) rather than an
      absolute upright angle. Still a numeric overlay around it; the game/animation layer is the rest
- [ ] refactor parts of CLAUDE.md into distributed rules, skills, hooks, commands, etc. All context is not needed for every prompt.
- [ ] fuse the Feather Sense IMU into the metrics — blocked on camera↔IMU clock alignment: the
      recording stores the device clock (ms since board boot) but not its offset to the host
      timeline. Would give a true gravity vector (no level-camera assumption) and 100 Hz sway
- [ ] program feather sense in C++ with Zephyr RTOS, or embedded Rust (embassy-nrf and nrf-hal)
- [ ] program feather sense in embedded Rust (embassy-nrf and nrf-hal). Use schematics from adafruit to build BSP.
- [ ] port to Android 
- [ ] consider multiple camera views
- [ ] consider adding a depth camera
