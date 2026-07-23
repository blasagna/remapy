# Data Collection Runbook — motor_metrics

*remapy / motor_metrics · operator's guide*

A step-by-step, follow-at-the-mat companion to
[`data-collection-protocol.md`](data-collection-protocol.md). The protocol explains *why*;
this runbook is the *what to do, in order*. Trial labels below are the exact vocabulary
from `motor_metrics/labels.py` — type them verbatim (structured, not free text).

---

## 0. One-time setup — redo only if the camera moves

1. **Place the mat and camera on floor-tape marks.** Camera **2.2–2.6 m** from the center
   of the mat, centered on it. Tape both positions so the geometry repeats every session.
   Assign this geometry a `setupid` (e.g. `livingroom1`) for filenames.
2. **Mount the camera at the tripod's minimum height and make it *level*.** Level (0° pitch)
   is the only load-bearing requirement — height is not. Do **not** tilt the camera down to
   re-center Remy; a tilt biases every trunk-angle and sway number. If he sits too low in the
   frame, either accept it or move the camera *farther* back — never tilt.
3. **Framing check:** confirm both shoulders, both hips, and **both wrists** stay in frame
   with margin, even with Remy sitting lower than center. Those landmarks (11, 12, 23, 24 +
   15, 16) drive every metric.

---

## 1. Every session — fixed pre-flight (~5 min)

Do these **in order, every time**, before Remy is on the mat:

1. **Check camera level** with a bubble level or phone level app resting on the camera body.
   This is the single assumption every trunk-angle and sway number depends on.
2. **Confirm tape positions** — camera and mat on their marks; same mat, same room, similar
   time of day, similar lighting.
3. **Scale reference:** hold a rod or two markers of **known length (e.g. 30.0 cm)** where
   Remy will sit, in frame, for **2–3 seconds**. This is the raw material for the scale check
   — without it, every meter-denominated sway number that session is unverifiable.
4. **Start recording with a live view:**
   `pixi run rerun --record YYYY-MM-DD_HHMM_<setupid>.h5`. This spawns the Rerun viewer so you
   can **see the camera feed and skeleton overlay live** — use it to confirm framing, camera
   level, and that all the load-bearing landmarks (shoulders, hips, both wrists) are tracked
   before Remy starts a trial. The `--record` flag writes the HDF5 recording alongside; with no
   `--save`/`--record-video` it's **HDF5 only** — no `.rrd`, no mp4. Faces are blurred by
   default. Keep the recording rolling continuously through the whole session; you segment it
   afterward in `annotate`, not during capture.

   > `pixi run record` also produces the same HDF5 but has **no on-screen view** to check
   > framing against — prefer the `rerun` form above whenever you have a display.

---

## 2. Calibration segment — always first (~10 s)

Have Remy sit or stand **as upright as he comfortably can, still, facing the camera** (front
view, same as the holds), for about 10 seconds. This is the vertical-reference diagnostic the
notebook checks before trusting any hold number.

> **Skipping calibration is the #1 way to silently invalidate a whole day.** Do it every
> session. Later you'll label this segment **`calib;pose=upright`**.

---

## 3. The trials — what Remy does, per exercise type

Run **four exercise types**. For **each type**: warm-up already done, then **3 trials** with
unhurried rest between them (this is not a timed clinical exam). If he tires before 3, record
what you get and note it — a 2-trial day is still usable, just with a wider error estimate.

**Rotate the order of the four types across sessions** (sit → stand → transition → crawl one
day; crawl → sit → transition → stand the next) so fatigue doesn't always penalize whichever
is last.

| Type | Camera view | What to have Remy do | Trial length | Label to use afterward |
|---|---|---|---|---|
| **Sitting hold** | **Front** (facing camera) | Sit on the mat and hold. Note honestly whether arms are free, propping on the floor, or held, and whether anyone supports his trunk/pelvis. | Until he loses interest or posture | `sit_hold;arms={free\|prop\|held};support={none\|trunk\|pelvis};gmfm=<item#>` |
| **Standing hold** | **Back** (facing away, toward the support) | Stand at the support and hold, noting the support used. Back view because he stands against a support that a front view can't see past. | Until he loses interest or posture | `stand_hold;support={hands_held\|trunk\|furniture};gmfm=<item#>` |
| **Transition** | **Front / three-quarter** | A single posture change, e.g. prone→sit or sit→prone. Note the side he leads/pushes from. Keep the view so both shoulders and both hips stay visible through the whole move. | Naturally short | `transition;from={prone\|sit};to={prone\|sit};side={left\|right};gmfm=<item#>` |
| **Crawl** | **Broadside** (travels across frame) | One crawl/belly-crawl bout across the mat, moving left↔right across the frame. Keep him slightly oblique so **both** wrists stay visible. Note direction of travel. | Naturally short (a few cycles) | `crawl;style=belly;dir={left\|right\|toward\|away};gmfm=<item#>` |

### Why these views (from the signal definitions)

- **Holds and transitions → front or back, never profile.** Every hold/transition metric is
  built on `trunk_vector` (`signals.py`) = `mid_shoulder − mid_hip`, so it needs **both**
  shoulders (11, 12) and **both** hips (23, 24). Front *and* back keep those left/right pairs
  separated in the image, so the midpoints come from real, visible points — the two views are
  metrically equivalent for everything this package computes (MediaPipe may swap the left/right
  labels from behind, but the midpoints are symmetric, so `trunk_vector` is unaffected). A
  **profile view self-occludes** the far-side landmark, and MediaPipe *extrapolates* the hidden
  one rather than dropping it — those frames still pass `pose_present` carrying invented
  coordinates, silently degrading the trunk vector without failing `coverage`.
- **Standing hold → back**, because Remy stands against a support a front camera can't see
  past. This is a fine substitute: back is metrically equivalent to front here (previous bullet),
  and face-blur still works (the pose/hybrid backend redacts the head region from pose keypoints,
  which are still detected from behind). Sitting hold and calibration stay front.
- **Front keeps the reliable sway axis reliable.** `project_horizontal` returns `(ML, AP)`
  where **ML (side-to-side) is in the image plane and measured well; AP (forward/back) is
  inferred depth and much noisier.** The sway framework assumes ML is the trustworthy axis, and
  a front view honors that. You *cannot* make both axes reliable with one camera — a side view
  would only trade a reliable ML for a noisy-landmark trunk vector, no net win. `trunk_angle` is
  **unsigned** anyway, so a side view buys nothing for lean magnitude.
- **Crawl is the exception → broadside.** The crawl signal is the wrist projected on the trunk
  axis plus pelvis translation across the frame (`com_norm`), so the body must *travel across*
  the image. Keep him slightly oblique rather than a dead profile so both wrists stay
  unoccluded.
- **Calibration → front,** matching the holds it's the vertical reference for.
- **Pick one view per exercise and use it identically every session** — trend comparability
  depends on the fixed geometry, not just the metrics.

Notes that matter for the numbers:

- **`arms=free` / `support=none` are *assertions in the label*, not detections.** The code
  trusts what you type. Be honest — a mislabeled support level splits your baseline.
- `gmfm=<item#>` is optional and free-text: copy the item number straight off the GMFM-88
  score sheet if you're tracking one. It's never validated, so any value is accepted.

### GMFM-88 item lookup — fill in from the score sheet, once

The code deliberately does **not** hard-code GMFM item numbers (`labels.py`: the numbering lives
behind the manual, and a wrong constant compared across months is worse than none). So fill this
table in **once**, against Remy's actual GMFM-88 score sheet (ideally with his PT), and reuse it
every session instead of re-deriving the number at each `annotate` prompt. The dimension each
exercise falls in (B = sitting, C = crawling, D = standing) is fixed by `labels.DIMENSIONS`; only
the item number is yours to confirm.

> **Verify the number *and its wording* before trusting it.** The "candidate item" column is the
> commonly-published numbering as a starting point, **not** verified to the manual — confirm each
> against the score sheet. Enter the item that matches what Remy *actually did*; the label's
> `arms=`/`support=` fields and the `gmfm=` number must describe the **same** trial.

> **`gmfm=` is optional and never affects a metric.** No quality gate or continuous-variable
> analysis reads it (it appears nowhere in `motor_metrics/*.py` outside `labels.py`) — it rides
> along only as a `p_gmfm` cross-reference to the score sheet. So a missing or unconfirmed number
> must **never block a session**: record the exercise plus its `arms=`/`support=`/`side=`/`dir=`
> fields (those *do* drive grouping), and add `gmfm=` later — `annotate` edits save in place, so
> it can be filled in after the fact once confirmed.

| Label you'll record | Dim | Candidate item (verify!) | Item wording (from your sheet) | **Confirmed item #** |
|---|---|---|---|---|
| `sit_hold;arms=free;support=none` | B | B24 (or B34 on a bench) | Sit on mat, arms free, maintains 3 s | `____` |
| `sit_hold;arms=prop` | B | B23 | Sit on mat, arm-propping, maintains 5 s | `____` |
| `sit_hold;support=trunk` | B | B21 / B22 | Sit, supported at thorax, head upright / midline, 3 s | `____` |
| `transition;from=sit;to=prone` | B | B30 | Sit on mat, lowers self to prone with control | `____` |
| `transition;from=prone;to=sit` | B | *depends on the movement* | Prone→sit; pick the item matching what he does | `____` |
| `crawl;style=belly` | C | C39 | Prone, creeps/commando-crawls forward 1.8 m | `____` |
| `stand_hold;support=furniture` | D | D48 | Standing, holding onto large bench with both hands | `____` |
| `stand_hold;support=hands_held` | D | *held-standing item in D* | Standing, held by an adult | `____` |

---

## 4. After the session — annotate and label

1. `pixi run annotate YYYY-MM-DD_HHMM_<setupid>.h5`
2. Scrub the recording; use `i`/`o` to mark in/out points of each segment, then type the
   label at the terminal prompt **in the vocabulary above — not free text.**
3. Label the calibration segment `calib;pose=upright` and each trial with its exact label.
   Mark any ruined stretch (a hand crossing the torso, an interruption) as
   **`exclude;reason=<what happened>`** — the analysis drops any trial overlapping an
   exclusion whole rather than trimming around it, so mark two good trials *around* a bad
   middle if needed.
4. Edits save immediately.

---

## 5. Quality gates — check before trusting the day

Run `pixi run metrics YYYY-MM-DD_HHMM_<setupid>.h5 --csv out.csv` and check each trial row:

- [ ] **`coverage ≥ 0.8`** per trial. Below 0.5 → mark `exclude` and re-shoot, don't salvage.
- [ ] **`tracked_s` close to `duration_s`** — a big gap means an occlusion mid-trial.
- [ ] **`warnings` column empty** — anything there is a label typo; fix the annotation.
- [ ] **Calibration diagnostic near 0°** (the notebook's vertical-reference check,
      `pixi run notebook`). If it's off, the camera was tilted all session — re-level before
      next time and treat that session's trunk/sway numbers as biased.

---

## 6. Number of trials / sessions — the bigger picture

- **Per session:** 1 calibration + **3 trials × 4 exercise types = 12 trials** (fewer if he
  tires; that's fine).
- **Reliability sub-study (do this first, before trusting any trend):** **2 sessions/day**
  ≥ 2 hours apart, **3 days/week for 3 weeks** = **nine same-day pairs**. Same-day pairing
  holds real development constant, so morning-vs-afternoon differences estimate measurement
  noise. From these, per metric, compute `SEM = SD_within` and `MDC95 = 1.96 × √2 × SEM` —
  the smallest change later that counts as real.
- **Do the scale + trunk-angle concurrent checks on at least the first session of every
  reliability block** (protractor on a frozen `calib` frame for trunk angle; measured
  known-length object for scale).

---

## 7. File / data hygiene

- **Filename:** `YYYY-MM-DD_HHMM_<setupid>.h5` — bump `setupid` only when the camera
  physically moves.
- **Keep every raw `.h5` forever** — metrics are recomputed on read, so future `derive.py`
  changes can re-run history.
- **Record the git commit hash** with each exported metrics batch — SPARC and sway velocity
  are only comparable across an identical filter chain.
- **Store reliability-block repeats separately** from ongoing trend sessions.

---

## Keep front-of-mind while collecting

- **Sway magnitudes have no home reference standard** — a second phone at another angle only
  confirms bigger/smaller, not the absolute meters.
- **SPARC has no external ground truth** — its value is only meaningful against itself via the
  trial-1-vs-trial-3 fatigue contrast.
- Everything here is **n = 1**, about Remy against his own baseline.
