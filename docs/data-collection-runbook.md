# Data Collection Runbook — motor_metrics

*remapy / motor_metrics · operator's guide*

Use this runbook at the mat. It gives the steps in their correct sequence. The
[`data-collection-protocol.md`](data-collection-protocol.md) file gives the reasons for the steps.

The trial labels in this runbook are the vocabulary from `motor_metrics/labels.py`. Type each
label exactly as this runbook shows it. Do not type free text.

---

## 0. Setup — do one time, and again only if the camera moves

1. **Put the mat and the camera on floor-tape marks.** Put the camera 2.2–2.6 m from the center
   of the mat. Align the camera with the center of the mat. Put tape at the two positions, so that
   each session uses the same geometry. Give this geometry a `setupid` (for example,
   `livingroom1`). Use the `setupid` in the file names.
2. **Attach the camera at the minimum height of the tripod.** Make the camera *level* (0° pitch).
   The level is the only important requirement. The height is not important.

   > **CAUTION:** Do not tilt the camera down to put Remy at the center of the frame. A tilt
   > adds an error to each trunk-angle number and each sway number. If Remy is too low in the
   > frame, accept it, or move the camera farther back.

3. **Do a check of the frame.** Make sure that the two shoulders, the two hips, and the two wrists
   stay in the frame with a margin. Do this check with Remy at a low position in the frame. All
   metrics use these landmarks (11, 12, 23, 24, 15, 16).

---

## 1. Checks before each session (approximately 5 min)

Do these steps in this sequence before Remy goes on the mat. Do them in each session.

1. **Make sure that the camera is level.** Put a bubble level or a phone level app on the camera
   body. All trunk-angle numbers and sway numbers are correct only when the camera is level.
2. **Make sure that the camera and the mat are on their tape marks.** Use the same mat and the
   same room. Use an equivalent time of day and equivalent light.
3. **Measure the body dimensions one time each month.** Do this measurement away from the camera.
   1. Use a tape measure to measure the shoulder width of Remy, from acromion to acromion.
   2. Measure his hip width.
   3. Write the two dimensions in the session notes.

   Do not hold an object in the frame for this check. The software does the scale check later.
   It compares your dimensions with the distance between landmarks 11 and 12 in
   `landmarks_world` during the `calib` segment.
4. **Start the recording with a live view.** Type this command:

   `pixi run rerun --record YYYY-MM-DD_HHMM_<setupid>.h5`

   1. Look at the camera image and the skeleton in the Rerun viewer.
   2. Make sure that the framing is correct and that the camera is level.
   3. Make sure that the viewer shows the shoulders, the hips, and the two wrists.
   4. Let the recording continue for the full session. Do not stop it between trials.

   The `--record` flag writes only an HDF5 file. It does not write an `.rrd` file or an mp4 file,
   unless you also use `--save` or `--record-video`. The software blurs the faces by default. You
   divide the recording into segments later, in `annotate`.

   > **NOTE:** The `pixi run record` command also makes the same HDF5 file. But it does not show
   > the camera image, so you cannot see the framing. When a display is available, use the
   > `rerun` command.

---

## 2. Calibration segment — always first (approximately 10 s)

> **WARNING:** Record the calibration segment at the start of each session. If you do not record
> it, you cannot use the hold numbers from that day. The notebook cannot find the error.

Tell Remy to sit or stand straight and to stay still for approximately 10 seconds. He must face
the camera (front view, the same view as for the holds). Later, give this segment the label
**`calib;pose=upright`**. The notebook uses this segment to find the vertical reference before you
use a hold number.

---

## 3. The trials — what Remy does for each exercise type

Do **four exercise types**. Do the warm-up first. Then, for each type, do **3 trials**. Let Remy
rest between the trials. This is not a timed clinical test, so do not hurry.

If Remy is tired before the third trial, record the trials that you have. Write a note about it.
A day with 2 trials is satisfactory, but the error estimate is larger.

**Change the sequence of the four types in each session.** For example, use sit → stand →
transition → crawl on one day. Then use crawl → sit → transition → stand on the next day. Thus,
fatigue does not always have an effect on the same type.

| Type | Camera view | What Remy does | Trial length | Label (type it later) |
|---|---|---|---|---|
| **Sitting hold** | **Front** (Remy faces the camera) | Remy sits on the mat and holds the position. Write if his arms are free, on the floor, or held. Write if a person holds his trunk or pelvis. | Until he stops or loses the posture | `sit_hold;arms={free\|prop\|held};support={none\|trunk\|pelvis};gmfm=<item#>` |
| **Standing hold** | **Back** (Remy faces away, toward the support) | Remy stands at the support and holds the position. Write which support he uses. Use the back view, because the support blocks a front view. | Until he stops or loses the posture | `stand_hold;support={hands_held\|trunk\|furniture};gmfm=<item#>` |
| **Transition** | **Front or three-quarter** | Remy does one posture change, for example prone→sit or sit→prone. Write the side that he pushes from. Keep the two shoulders and the two hips in the view for the full movement. | Short | `transition;from={prone\|sit};to={prone\|sit};side={left\|right};gmfm=<item#>` |
| **Crawl** | **Broadside** (Remy moves across the frame) | Remy does one crawl or belly-crawl across the mat, from left to right or right to left. Put him at a small angle to the camera, so that the two wrists stay in view. Write the direction of movement. | Short (a few cycles) | `crawl;style=belly;dir={left\|right\|toward\|away};gmfm=<item#>` |

### Why the runbook uses these views

- **Use the front or back view for holds and transitions. Do not use a side view.** All hold
  metrics and transition metrics use `trunk_vector` (`signals.py`), which is
  `mid_shoulder − mid_hip`. Thus, these metrics need the two shoulders (11, 12) and the two hips
  (23, 24).
- In the front view and in the back view, the left and right points are apart in the image.
  Thus, the midpoints come from real points that the camera can see. The two views give the same
  metrics. MediaPipe can change the left and right labels in the back view. The midpoints do not
  change, so `trunk_vector` stays correct.
- In a side view, the near side of the body hides the far-side landmarks. MediaPipe does not
  remove a hidden landmark. It calculates an estimated position. These frames pass
  `pose_present` with incorrect coordinates. The trunk vector becomes less accurate, but
  `coverage` does not show the problem.
- **Use the back view for the standing hold.** Remy stands against a support, and the support
  blocks a front view. The back view gives the same metrics as the front view (see the previous
  item). The face blur also works. The pose and hybrid backends find the head from the pose
  keypoints, and MediaPipe finds these keypoints from behind. Use the front view for the sitting
  hold and for the calibration.
- **The front view keeps the reliable sway axis reliable.** `project_horizontal` gives `(ML, AP)`.
  ML (side to side) is in the image plane, so its measurement is good. AP (forward and back) comes
  from an estimated depth, so it has much more noise.
- The sway metrics use ML as the reliable axis, and a front view agrees with this. One camera
  cannot make the two axes reliable. A side view gives a reliable AP axis, but it makes the trunk
  vector less accurate. Also, `trunk_angle` has no sign, so a side view does not improve the lean
  value.
- **Use a broadside view for the crawl.** The crawl signal is the wrist position along the trunk
  axis and the pelvis movement across the frame (`com_norm`). Thus, the body must move across the
  image. Put Remy at a small angle to the camera, so that the two wrists stay in view. Do not use
  a full side view.
- **Use the front view for the calibration.** The calibration is the vertical reference for the
  holds, so it uses the same view as the holds.
- **Use the same view for each exercise in each session.** You can compare trends only when the
  geometry stays the same.

Notes about the numbers:

- **The values `arms=free` and `support=none` are what you state in the label.** The software does
  not measure them. The software accepts the value that you type. Type the correct value. An
  incorrect support level divides your baseline into two groups.
- The `gmfm=<item#>` field is optional, and you can type any text in it. If you use the GMFM-88
  score sheet, copy the item number from the sheet. The software does not do a check of this
  value.

### GMFM-88 item numbers — complete this table one time

The software does not contain GMFM item numbers. The item numbers come from the manual, and an
incorrect number in the software would cause errors in comparisons across months. See
`labels.py` for this decision.

Complete this table one time, with Remy's GMFM-88 score sheet. If possible, do this with his
physical therapist. Then use the table in each session at the `annotate` prompt.

`labels.DIMENSIONS` sets the GMFM-88 dimension of each exercise (B = sitting,
C = crawling, D = standing). You must find only the item number.

> **CAUTION:** Make sure that each item number and its text agree with the score sheet. The
> "Candidate item" column shows numbers from published sources. These numbers are only a start
> point. They are not checked against the manual. Use the item that agrees with the movement that
> Remy did. The `arms=` field, the `support=` field, and the `gmfm=` number must all be for the
> same trial.

> **NOTE:** The `gmfm=` field is optional, and it has no effect on the metrics. No quality gate
> and no analysis uses it. In `motor_metrics/`, only `labels.py` refers to it. It is in the output only
> as the `p_gmfm` column, so that you can find the item on the score sheet.
>
> Do not stop a session because you do not have a number. Record the exercise and its `arms=`,
> `support=`, `side=`, and `dir=` fields, because these fields control the groups. You can add
> `gmfm=` later. The `annotate` tool saves edits in the file.

| Label to record | Dim | Candidate item (make sure) | Item text (from your sheet) | **Confirmed item #** |
|---|---|---|---|---|
| `sit_hold;arms=free;support=none` | B | B24 (or B34 on a bench) | Sit on mat, arms free, maintains 3 s | `____` |
| `sit_hold;arms=prop` | B | B23 | Sit on mat, arm-propping, maintains 5 s | `____` |
| `sit_hold;support=trunk` | B | B21 / B22 | Sit, supported at thorax, head upright / midline, 3 s | `____` |
| `transition;from=sit;to=prone` | B | B30 | Sit on mat, lowers self to prone with control | `____` |
| `transition;from=prone;to=sit` | B | *Use the item for the movement that he does* | Prone→sit; select the item for the movement | `____` |
| `crawl;style=belly` | C | C39 | Prone, creeps/commando-crawls forward 1.8 m | `____` |
| `stand_hold;support=furniture` | D | D48 | Standing, holding onto large bench with both hands | `____` |
| `stand_hold;support=hands_held` | D | *The held-standing item in D* | Standing, held by an adult | `____` |

The "Item text" column copies the text of the score sheet, so it does not use STE.

---

## 4. After the session — annotate and label

1. Type `pixi run annotate YYYY-MM-DD_HHMM_<setupid>.h5`.
2. Move through the recording. Push `i` at the start of a segment. Push `o` at the end of the
   segment.
3. At the terminal prompt, type the label. Use the vocabulary in this runbook. Do not type free
   text.
4. Give the calibration segment the label `calib;pose=upright`. Give each trial its correct label.
5. If a part of the recording is bad, give it the label **`exclude;reason=<what happened>`**. For
   example, a hand goes across the torso, or a person stops the trial.

   The analysis removes all of each trial that touches an `exclude` segment. It does not cut the
   bad part from the trial. Thus, if the middle of a trial is bad, mark two good trials, one
   before and one after the bad part.

The `annotate` tool saves each edit immediately.

---

## 5. Quality gates — do these checks before you use the data for the day

Type `pixi run metrics YYYY-MM-DD_HHMM_<setupid>.h5 --csv out.csv`. Then do a check of each trial
row:

- [ ] **`coverage ≥ 0.8`** for each trial. If the value is less than 0.5, mark the trial `exclude`
      and record it again. Do not try to repair it.
- [ ] **`tracked_s` is near `duration_s`.** A large difference shows that something blocked the
      camera during the trial.
- [ ] **The `warnings` column is empty.** A value in this column shows an error in a label. Correct
      the annotation.
- [ ] **The calibration result is near 0°.** Use the vertical-reference check in the notebook
      (`pixi run notebook`). If the result is not near 0°, the camera had a tilt for the full
      session. Make the camera level before the next session. The trunk numbers and sway numbers
      from that session have an error.

---

## 6. Number of trials and sessions

- **For each session:** 1 calibration + **3 trials × 4 exercise types = 12 trials**. If Remy is
  tired, fewer trials are satisfactory.
- **Reliability sub-study:** Do this sub-study first, before you use a trend.
  1. Do **2 sessions each day**, with a minimum of 2 hours between them.
  2. Do this on **3 days each week for 3 weeks**. This gives **nine same-day pairs**.

  Remy does not develop much in one day. Thus, a difference between the morning and the
  afternoon shows the measurement noise. For each metric, calculate `SEM = SD_within` and
  `MDC95 = 1.96 × √2 × SEM`. `MDC95` is the smallest change that is real.
- **Do the scale check and the trunk-angle check in the first session of each reliability block.**
  For the trunk angle, use a protractor on a frozen `calib` frame. For the scale, compare the
  measured shoulder width with `landmarks_world` during the `calib` segment.

---

## 7. Files and data

- **File name:** `YYYY-MM-DD_HHMM_<setupid>.h5`. Change the `setupid` only when you move the
  camera to a different position.
- **Keep all raw `.h5` files.** Do not delete them. The software calculates the metrics again each
  time that it reads a file. Thus, after a change to `derive.py`, you can calculate all old
  sessions again.
- **Write the git commit hash with each set of exported metrics.** You can compare SPARC and sway
  velocity only when the same filter chain calculated them.
- **Keep the repeat sessions of the reliability block apart from the usual trend sessions.**

---

## Remember during the collection

- **Sway values have no reference standard at home.** A second phone at a different angle can
  show only if the sway is larger or smaller. It cannot show the value in meters.
- The meter values come from the MediaPipe estimate of the body size of Remy. They are not a
  calibrated measurement. The monthly body-scale check can find only a change in that estimate.
- **SPARC has no external reference.** Compare its value only with other SPARC values. For
  example, compare trial 1 with trial 3 to find the effect of fatigue.
- All of this data is for **n = 1**. It compares Remy only with his own baseline.
