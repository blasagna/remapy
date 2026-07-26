# Reading the overlay

What every row on the phone screen means, for the person holding the phone. For the code behind
these numbers see `motor_metrics/CLAUDE.md` and the `live.py` docstrings; this file assumes you are
mid-session and want to know whether to trust what you are looking at.

The panel is top-left. Three buttons are top-right: **exercise** (`hold` / `crawl` / `off`),
**lens** (`rear` / `front`) and **framing** (`landscape` / `portrait`). Each shows the state that
is live right now, not the one it will switch to.

## Read these first, in every mode

| Row | Reads |
| --- | --- |
| `coverage 84%  (3.2s)` | How much of the rolling window had **trusted torso landmarks** (84 %), and the longest unbroken stretch of trusted tracking in it (3.2 s). |
| `up world_y` | Which vertical the maths is using. |
| `fps 15  gpu` | Delivered frame rate, and which inference delegate. |

**`coverage` is the row that licenses all the others.** Green at 50 % or better, red below. When it
is red every metric below it reads `--`, and that is the design, not a fault: a number left on
screen while tracking has failed reads as a measurement of the child. Persistent red is a framing
or lighting problem — get the whole torso in frame and better lit — not a bug to report.

**`up world_y` is a caveat, which is why it is orange.** `world_y` means the vertical is the
phone's own down-axis, so it assumes the phone is level. Prop it at an angle and the absolute
lean is wrong. This matters for `hold` and **not at all for `crawl`**, which reads no vertical
anywhere — its axis is the child's own trunk vector. It is also why the hold readout leads with
`trunk dev` (a difference from its own baseline) rather than an absolute upright angle.

**`fps` is green at 13.5 and above**, which is 90 % of the 15 Hz grid the whole filter chain is
built on. A healthy reading sits at **15 and never above** — the app deliberately throttles to the
grid. Below 13.5 the chain is resampling *up* and inventing samples that correlate with each
other, which lands on the sway numbers without announcing itself.

The `gpu` / `cpu` label has a limitation worth knowing: it reports which delegate was successfully
**built at startup**, not which is performing now. A hot, throttled phone still says `gpu`. If fps
is low, that is the first thing to rule out, not the last.

**Check the face redaction with your own eyes**, every session, after every lens or framing change.
A black box over the face is the system working. A **fully black frame** means nothing located a
face and the app blanked the whole thing rather than risk showing one — also working, but if it
happens repeatedly with someone clearly in view, that is worth reporting.

## `hold` — sitting and supported standing

A 5-second rolling window. All of it is the sway of the trunk over the pelvis.

| Row | Reads | Better |
| --- | --- | --- |
| `sway RMS 0.0123 m` | Typical distance of the trunk from its own average position over the window. A distance, not a speed. | lower |
| `  ML / AP 0.0081 / 0.0093` | The same sway split into **ML** side-to-side and **AP** front-to-back, in metres. | lower |
| `sway vel 0.0142 m/s` | Mean speed the trunk travels while holding. Catches fast small corrections that `sway RMS` averages away. | lower |
| `trunk dev 2.4 deg` | Current trunk lean **minus the window's own median lean**. | nearer 0 |
| `steady` bar | The same `trunk dev` as a red→green fill. Full green = on the window's median lean; empty = 10° or more off it. | fuller |

Two things to hold onto here.

**`trunk dev` is not "how upright he is".** It is how far he has moved from whatever he was doing
over the last few seconds. Zero means steady, in whatever position that is. This is deliberate: an
absolute upright angle would inherit the level-camera assumption, and a phone propped on a cushion
would score a perfectly steady child badly. Referencing his own baseline puts a tilted phone into
the baseline instead of into the number.

**`ML` and `AP` are not equally trustworthy.** Side-to-side sits in the image plane and is measured
well. Front-to-back is inferred depth and is markedly noisier. If the two disagree, believe `ML`.

The `steady` bar's 10° is a display tolerance chosen to make the bar move usefully. It is not a
clinical threshold and nothing is scored against it.

## `crawl` — belly crawl

A 6-second window, because a period statistic needs several pull cycles before it means anything.
`cpm` is **cycles per minute**. The legs lead the readout: that is where the signal is.

| Row | Reads |
| --- | --- |
| `arm cad 42.0 cpm` | Arm pull cycles per minute, pooled across both arms (from the wrists). |
| `leg cad 38.0 cpm` | Leg cycles per minute, pooled across both legs (from the knees). |
| `  L / R 44.0 / 32.0` | The same leg cadence split by side. A gap here is favouring, showing up in the rate. |
| `leg favor 0.85 L` | **The headline.** How much further one leg travels than the other, and which. `even` below 0.05. |
| `leg cyc 4 / 3` | Raw leg cycle counts, left / right, inside the window. The small integers behind the cadences. |
| `period CV 0.180` | How irregular the **leg** cycle timing is: standard deviation over mean period. Dimensionless, so it compares across cadences. Lower is more rhythmic. |

**`leg favor` is on its own scale — it is not a percentage.** It is the difference between the two
legs' travel divided by their average, which runs from `0` to `2`:

| Reading | Means |
| --- | --- |
| `even` | under 0.05 apart |
| `0.67 L` | the left leg travels **twice** as far as the right |
| `1.20 L` | roughly **three times** as far |
| `2.00 L` | the right leg is not moving at all |

The letter is the side that travels further. This row reads no vertical at all, so it survives a
tilted or hand-held phone intact — of everything on screen it is the least sensitive to how the
phone is propped.

`arm cad` has no left/right split on screen; the arms are shown pooled because the developmental
question here is about the legs. The offline table has the per-arm figures.

Not on screen, and not an oversight: **reciprocity** (whether the limbs alternate or move together
— "bunny haul" versus a mature crawl) needs a transform whose error is worst at exactly the edges
of a short trailing window, so it stays offline. So does forward **speed**, which is only
comparable at fixed framing.

## `off`

Hides the whole panel, `coverage` and `fps` included. The camera, the pose tracking and the face
redaction all keep running, and so do the metrics underneath — nothing is switched off but the
drawing.

Turning it off also switches the measurement back to `hold`, so the few seconds a mode change costs
are spent while there is nothing on screen to watch. Leave it off for longer than that and the tap
back shows a populated `hold` readout immediately.

The skeleton stays drawn in `off`. It is the check that tracking is still working, and it costs
nothing to read.

## Things that apply everywhere

**`--` always means "no number", never zero and never a stale one.** It appears when the window is
too short, when coverage is below the gate, or when the maths could not produce a value. A dash is
the readout refusing to guess.

**Every number is about 1/7 of a second old**, and that is on purpose rather than a lag to
apologise for. Reading two samples back from the edge of the window is the point at which the live
value equals *exactly* what the offline analysis would report for that instant. Reading the very
latest sample would be fresher and meaningfully wrong.

**Switching between `hold` and `crawl` throws the window away.** The two use different window
lengths, so there is nothing to carry across. The readout blanks and re-warms over the next five to
six seconds — expected, and worth doing between trials rather than during one. Flipping the lens or
the framing discards it too, for a stronger reason: both change what the camera sees, and averaging
across that would measure the change rather than the child. Coming *out* of `off` does not — that
cost was already paid on the way in.

**Nothing is recorded.** No files, no network, no storage permission. Nothing on this screen is
saved anywhere, so a number worth keeping has to be written down or re-derived from a laptop
recording.

**Phone numbers and laptop numbers are not yet interchangeable.** Whether the phone's pose model
produces the same landmarks as the desktop one has not been measured — see *Open risk: landmark
parity* in `CLAUDE.md`. Until it is, treat a session recorded on the phone as its own baseline
rather than a continuation of the laptop trend.
