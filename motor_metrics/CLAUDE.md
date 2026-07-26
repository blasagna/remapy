# `motor_metrics/`

Continuous motor metrics for standardized therapy exercises, computed **offline** from
recordings (see the README references for GMFM-88 / PDMS-3 / AIMS).

- **The premise, which drives every design choice here:** the instruments score **ordinally**
  (GMFM items are 0–3; AIMS is observed/not-observed) and that is too coarse to show change —
  Remy can sit at a "2" for a year while genuinely improving. So the scoring is **not**
  reimplemented. Each item defines a reproducible **trial**; this package measures the
  *continuous variable underneath it*. **GMFM-88 is the spine** (criterion-referenced, so no age
  ceiling — AIMS norms stop at 18 months and he would floor out; dims B/C/D bracket him exactly).
  PDMS-3 is deferred (needs the kit + examiner administration).
- **Scope: camera-only, offline.** The Feather Sense IMU is out, which is what lets
  `segments.py` be a plain `searchsorted` — annotations and `Recording.timestamps_ms` share a
  clock, while the IMU's is a device clock whose offset is never stored. See the README TODO.
- **Division of labour: the annotator judges, the code measures.** `duration_s` is the *marked*
  trial length, because the in/out points are a human's call on when sitting began and ended.
  Nothing infers loss-of-posture from a trunk-angle threshold, and `arms=free` is an assertion in
  the label, not a detection (`hands_low_frac` is a QC hint — weight-bearing is a force question
  and there is no force sensor). `tracked_s` is a *data-quality* figure, not a claim about sitting.
- **Derive-on-read, never written back** — the same rule as `recording/recorder.py`, and it binds
  harder here: every number is a function of the `derive.py` constants, so freezing metrics into
  the `.h5` would strand them at whatever those were that week. `Recording.angles()` is the pattern.
- `labels.py` — the annotation vocabulary, `exercise[;key=value]*` (`sit_hold;arms=free;gmfm=23`).
  Typed by hand at `annotate`'s prompt; **`annotate` never parses the vocabulary** — it stores and
  displays labels as free text, so this stays a purely read-side convention (the GUI's on-screen
  label readout shows them verbatim, it does not validate them). `parse_label` is **total**:
  returns `None` on legacy free text,
  never raises. Value checking is separate and advisory (`label_warnings`) because a typo'd
  `arms=freee` must not raise mid-report but must not pass silently either — it would become its
  own `groupby` bucket and split a baseline. GMFM item *numbers* are deliberately not hard-coded
  (`gmfm=` is a free field copied off the score sheet); only the B/C/D `DIMENSIONS` map is.
- `quality.py` — `Gate`/`landmarks_ok`/`coverage`/`longest_run`. Load-bearing, because MediaPipe
  **extrapolates** occluded landmarks rather than dropping them: those frames pass `pose_present`
  carrying invented coordinates. (`pose_present` is *not* buggy — `recorder.py` writes a full
  33×3 NaN row, so NOSE-x NaN ⟺ whole row NaN. Don't "fix" it.) Every metric gates on the
  landmarks it reads and returns `coverage` beside its numbers. `longest_run` never bridges a
  dropout — that would invent the movement inside it.
- `signals.py` — **`landmarks_world` is hip-centered: MediaPipe puts the frame's origin *at* the
  mid-hip.** Two consequences run through everything: (1) a mid-hip "COM sway" proxy is
  **identically zero** — pinned by `test_world_mid_hip_com_proxy_is_identically_zero`; the real
  signal is `trunk_vector` (mid-shoulder over the pelvis), which the hip-centered origin makes
  scale- and calibration-free; (2) floor translation is **not recoverable** — only `com_norm`
  (image fractions) sees it. `trunk_from_vertical` reuses `angles.angle_between` and inherits its
  **unsigned** semantics (no forward/backward/lateral); `project_horizontal` carries direction and
  returns (ML, AP) **split on purpose** — ML is in the image plane, AP is inferred depth and much
  noisier. `WORLD_UP` is vertical **only if the camera is level**; `estimate_up` is opt-in and not
  the default (it cannot separate camera tilt from a child who does not sit vertically), and every
  metric records `up_source`.
- `derive.py` — resample to a uniform grid, then Savitzky-Golay. `FS`/`WINDOW_S`/`POLY` are
  **module constants, not per-call knobs**: two numbers are comparable only through an identical
  chain, and these get compared across months. The chain's measured derivative gain is documented
  and pinned (0.994 at 0.25 Hz → 0.90 at 1 Hz → 0.34 at 3 Hz) — a *bias*, identical for every
  trial, so it cancels within-child but makes the magnitudes non-comparable to any other pipeline.
  First and only use of scipy in the repo. `smooth` returns NaN below the window, never raises.
  **`FS` and `WINDOW_S` are a pair, not two independent knobs.** The window is written in seconds
  and applied in samples, so `WINDOW_S=0.25` at `FS=15` gives `int(0.25*15)=3` — and three samples
  fit a quadratic *exactly*, making the filter the identity and every derivative a raw central
  difference, silently and without raising. The grid moved 30 → 15 Hz to match what the capture
  hardware sustains, so `WINDOW_S` moved 0.25 → 0.35 (5 samples, 0.333 s). `WindowLengthTests` pins
  both the shipped pair clearing the floor and the collapse case itself. **Every number computed
  before that change belongs to a different scale** — recoverable, since metrics are derived on
  read, so re-running `pixi run metrics` regenerates past sessions on the new chain; any figure
  already written down elsewhere is not.
- `segments.py` — annotations → frame spans. Drops unparseable labels and anything overlapping an
  `exclude` **whole** (a hold whose middle is untrustworthy is not a shorter valid hold; trimming
  would invent a boundary). Mark two trials around the excluded stretch to keep the good parts.
  Also carries a public `Span` (start/stop with no annotation), used by the live path — replacing
  `crawl.py`'s private `_Span`.
- `hold.py` — sitting **and** supported standing (one function; the label carries the difference).
  Duration + sway (path length, 95 % ellipse, RMS, ML/AP split, mean velocity) + trunk-angle stats.
  **`path_length_m` is duration-confounded** — a worse 20 s hold beats a better 8 s one on it; use
  `mean_velocity_mps` or a common `window_s`. **Ellipse area reads ~0 for one-axis rocking** (a
  line encloses nothing), so read it next to the ML/AP RMS, never alone.
- `transition.py` — `sparc` (spectral arc length) is primary on trunk angular speed. **Read only
  against itself**: the absolute value is a pipeline artifact of the grid it was computed on — the
  integrated band is `min(SPARC_FC, fs/2)`, so `FS` moves it. Its measured noise
  robustness has a **ceiling** — flat to ~2 % of peak speed, erratic past ~5 % (sd 0.24) — so big
  brisk transitions score reliably and small slow ones may not; prefer the median of several.
  It separates *fluid from effortful* but does **not** count corrections (more submovements
  eventually score *better* as they blend); `count_submovements` counts. `symmetry_index` is a
  **between-trial** comparison grouped by the label's `side=`.
- `crawl.py` — **the GMFM item's "1.8 m" is not measurable here** (hip-centered frame, one camera,
  no depth, no IMU); `speed_norm_per_s` is in **image widths/s, never metres**. The deliverable is
  the *pattern*, which the pelvis-centered frame captures perfectly: cadence, `cycle_period_cv`,
  and `phase_offset` (**0.5 = reciprocal/mature, 0.0 = symmetric "bunny" haul**). Limb signal is
  the limb projected on the **trunk axis** — in prone there is no useful vertical.
  `MIN_CYCLE_EXCURSION_M` is **not optional**: the prominence gate alone is relative to the
  signal's own range, so it normalizes pure jitter up into a textbook crawl (measured: 57 cycles
  from noise, a still child reporting a confident fictional cadence).
  **Both limb girdles are measured** — arms (wrists, the unprefixed fields, their long-standing
  names) *and* legs (knees, the `leg_*` fields), computed by one `_girdle` helper called twice.
  This is load-bearing for Remy: his arms often move together (symmetric, not alternating) while he
  drives with the **legs and favors one repeatedly** — so `leg_amplitude_symmetry` ("favors one leg",
  sign = side) and `leg_phase_offset` (alternating vs together) carry the signal the wrists miss.
  Each girdle **gates independently** (`ARM_LANDMARKS`/`LEG_LANDMARKS`, own `coverage`/`tracked_s`):
  legs leave frame in prone more than arms, and one occluded knee must not cost the arm cadence.
  `_prepared`/`limb_signal` take a `marker` (`wrist`/`knee`); `quality.KNEES` mirrors `WRISTS`.
- `live.py` / `live_draw.py` — the **live** path: the same `hold_metrics`/`crawl_metrics`, run over
  a rolling window during capture instead of over an annotator's marks. `LiveWindow` is a ring
  buffer that *is* a `Recording` as far as the metrics care (it exposes exactly the attributes
  `tests.fakes.fake_recording` does), so **no metric math is reimplemented**; it pushes through
  `recording.recorder.landmark_rows` — extracted from `HDF5Recorder.append` — so a live frame and a
  recorded one convert identically, NaN convention included. Two measurements shape it. **Cost is
  not the constraint** (~3 ms per recompute over a 5 s window, ~8 ms over 30 s, against a 67 ms
  frame MediaPipe already dominates), so the window is recomputed whole every `RECOMPUTE_EVERY`
  frames. **`LIVE_LAG` is computed, not written down** (`window_length() // 2`, currently 2): the
  Savitzky-Golay fit leaves the last 2 samples of *any* window one-sided, and measured against the
  offline chain the edge-extrapolated velocity carries ~90 % of the signal's own spread as error
  (RMSE 0.0614 m/s vs a 0.0679 m/s signal sd) while at that lag the live value is **bit-identical**
  to the offline one. It was a literal `3` until `FS` moved and it silently became a stale value
  that still ran — deriving it is what stops that recurring. So instantaneous readouts are 133 ms
  old *on purpose*; aggregates (sway RMS,
  cadence) span the window and dilute the 3 edge samples away, and are not trimmed — trimming only
  moves the edge. A fixed-epoch resample anchor was tried and **rejected on measurement** (rolling
  jitter 0.143 % → 0.124 % of value, with a *worse* max), so `derive.py` keeps its no-per-call-knobs
  property. What is deliberately absent: **no movement detector**, hence no SPARC / submovements /
  `movement_duration_s` — they need an onset the code is not entitled to invent, which is the same
  refusal `hold.py` makes about loss-of-posture; no `duration_s` (the annotator's marks); no
  *between-trial* `symmetry_index` (transition's, grouped by `side=`) — but the crawl **within-window**
  left/right `live_leg_amplitude_symmetry` **is** live (the "favors one leg" readout, distinct from
  the between-trial one); no `speed_norm_per_s` (framing-dependent) or `phase_offset`/`leg_phase_offset`
  reciprocity (Hilbert's edge effects peak at a short trailing window's edges, so alternating-vs-together
  stays offline for both girdles). **Crawl carries both girdles live** (`live_*` arm fields + `live_leg_*`
  fields off the knees); the crawl overlay leads with the legs, since that is Remy's signal. `crawl` is
  the **camera-robust** mode — it reads no `up` at all — while `hold` inherits `WORLD_UP`'s level-camera
  assumption, so
  its headline is `live_trunk_angle_delta_deg` (referenced to the window's own median) rather than
  an absolute lean, and `up_source` is on screen. **The never-mix rule:** live values must never
  reach the `.h5` or `metrics_table` — different window, no marked trial, a 3-sample-old readout —
  and it is enforced *structurally*, since **every `LiveMetrics` field is `live_`-prefixed** and a
  test pins both that and disjointness from the offline columns. Keep the prefix. Below
  `MIN_COVERAGE` the readout **blanks rather than going stale** (a number left on screen during a
  dropout reads as a measurement of the child); `live_draw` renders NaN as `--` and the Rerun logger
  skips NaN so the plots stop advancing.
  **Loop order is load-bearing** in `rerun_viewer/main.py`: the recorder archives the clean blurred
  frame *before* `draw_live_metrics` runs, because the overlay mutates in place and a HUD burned
  into a recording would show numbers the offline metrics legitimately disagree with.
  **Overlay text is plain cyan** (a single `cv2.putText`, matching `pose_estimation.main.draw_angles`)
  — deliberately *not* the outlined `draw.put_text`, whose black under-copy reads as a drop shadow the
  joint-angle overlay lacks; coverage stays green/red and `up:` orange, the two colors that carry
  meaning. Legibility over busy footage instead comes from a **translucent dark panel** shaded behind
  the whole readout via `draw.shade_box` (a clamped `cv2.addWeighted` blend) — the same panel
  `draw_angles` now draws behind the top-left joint angles, sized to the measured text. `_shade_panel`
  walks the same per-element `y` advances the draw code uses, so the box tracks the layout. **`sit_steadiness`/`_draw_meter`** add the first child-facing piece: a red→green fill bar
  giving good-vs-bad sit-hold feedback on a continuum. Its honesty is in *what it reads* — the
  trunk's deviation from its **own** window baseline (`live_trunk_angle_delta_deg`), so a tilted
  camera shifts the baseline not the score, and never an absolute upright angle or a "good posture"
  threshold (the loss-of-posture criterion `hold.py` refuses). `STEADINESS_TOL_DEG` is a display/game
  tolerance, not a validated threshold — it lives in the render layer (`live.py` stays pure
  buffer-and-dispatch and computes no metric of its own). Crawl gets no meter (it reads no vertical).
- `report.py` / `main.py` — `metrics_table(rec)` → one row per trial, columns the **union** across
  exercises (so they concatenate into a trend); `session_table(paths)` for the cross-session view,
  which is the whole point given there is no external GMFM score to calibrate against. Label params
  are prefixed `p_` to stay clear of metric fields of the same name; `warnings` carries the label QC.
  `duration_s`/`tracked_s`/`coverage`/`n_frames` mean the same thing in **every** metric dataclass
  so the union table has no per-exercise holes. CLI: `pixi run metrics session.h5 [--csv out.csv]`.
- `notebooks/motor_metrics.ipynb` (`pixi run notebook`) — label inventory + coverage QC → **the
  vertical-reference diagnostic** (check it before trusting any hold number) → per-exercise tables
  → sway/limb plots → cross-session trend.

## Tests (`tests/test_motor_metrics.py`)

The metrics package. Pure logic runs unmocked against real numpy/scipy and is pinned to **closed
forms, not recorded outputs** (polygon perimeter for path length; `5.991·π·σ²` for the sway ellipse;
constructed lean angles; sine frequency for cadence; anti-phase → 0.5). Where no closed form exists
the tests pin **ordering or invariance**: SPARC's absolute value is a pipeline artifact, so asserting
a number would pin the noise. Three regression pins earn their keep: the world mid-hip COM proxy being
identically zero, the `derive.py` filter's frequency response, and `MIN_CYCLE_EXCURSION_M` (without it
pure jitter reads as 57 crawl cycles). Every metric is exercised for empty / single-frame /
shorter-than-window / fully-untracked segments — all must return NaN, never raise, since one
mis-marked annotation must not take down a 40-row report. Integration tests drive the real
`HDF5Recorder` → `AnnotationStore` → `Recording` → `metrics_table` path against temp files.

## Tests (`tests/test_live.py`)

The live path. Same discipline as the metrics tests: sway RMS against `amplitude/sqrt(2)`, cadence
against the driving frequency. Three pins carry it — the **`LIVE_LAG` identity** (trailing-window
value at `-4` equals the offline value to 12 places, *and* the edge velocity's error rivals the
signal's own sd, so nobody "simplifies" the lag to 0), the **never-mix rule** (every field
`live_`-prefixed and disjoint from the offline columns), and **blanking after a dropout** (a stale
sway figure must not survive into an untracked window). Plus ring wraparound/ordering, `pose_present`
being True for *extrapolated* low-visibility landmarks while `landmarks_ok` is False (the trap,
restated where it is most tempting to conflate them), a still child reporting **no** cadence, and a
per-push budget. `draw_live_metrics` is pinned to mutate in place — that is why the capture loop
archives before drawing.
