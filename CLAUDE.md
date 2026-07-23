# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

Early stage. "Playing with tools inspired by Remy's therapies" (per README). The dependency
set points at computer-vision / motion work: `mediapipe`, `opencv`, `rerun-sdk` (visualization),
plus `numpy`/`scipy`/`pandas` and `h5py` for data.

Current code: `video_capture/` (OpenCV capture), `pose_estimation/` (MediaPipe pose,
consuming `video_capture`), `face_blur/` (MediaPipe face redaction), `rerun_viewer/`
(logs the pipeline to the Rerun viewer), `recording/` (HDF5 recording for offline analysis),
`annotate/` (label time segments in a recording), `motor_metrics/` (continuous metrics for
standardized GMFM-88 trials, computed offline from labeled recordings), and `list_devices/`
(enumerate compatible capture devices).

## Packages

### `video_capture/`

- `capture.py` — `VideoCapture`, a context-manager wrapper around `cv2.VideoCapture`. Defaults
  to camera index `0` (built-in webcam); also accepts a file path/URL. Provides `read()` (single
  BGR frame), `frames()` (iterator), a `resolution` property, and raises `CaptureError` on
  open/read failure. Import both from `video_capture.capture` (the `__init__.py` files are
  intentionally empty — import from submodules, not the package root).
- `main.py` — example CLI (`python -m video_capture.main`) that displays the webcam feed in a
  window (quit with `q`/`Esc`). Flags: `--source`, `--width`, `--height`, `--no-window`
  (headless), `--max-frames` (bounded runs). A bare-integer `--source` is treated as a camera
  index, anything else as a path/URL. All four CLIs request **1280×720** by default
  (`--width`/`--height` override; the device picks the nearest supported mode).

### `pose_estimation/`

Body-pose skeleton + joint angles, using `video_capture` for frames.

- **MediaPipe API note:** this build (0.10.35, Python 3.14) ships only the **Tasks API** —
  `mp.solutions.pose` and `mediapipe.framework` are absent. Use `PoseLandmarker` from
  `mediapipe.tasks.python.vision`. There are no bundled drawing utils, so skeletons are drawn
  manually with OpenCV.
- `model.py` — `ensure_model()` downloads/caches the **lite** `.task` bundle (not shipped with
  pip) under `pose_estimation/models/` (gitignored) on first run.
- `estimator.py` — `PoseEstimator`, a context manager wrapping `PoseLandmarker` in VIDEO mode.
  `detect(frame_bgr, timestamp_ms)` requires monotonically increasing timestamps. `POSE_CONNECTIONS`
  is the skeleton edge list.
- `angles.py` — `joint_angles()` derives elbow/shoulder/knee/hip angles from
  `pose_world_landmarks` (metric coords; MediaPipe does not output angles itself).
- `draw.py` — `draw_skeleton(frame, landmarks_norm, connections, visibility=None,
  min_visibility=0.5)`, the one OpenCV skeleton drawer, shared by the live CLI and `annotate`.
  Takes a plain `(K, 2+)` **numpy** array (not MediaPipe objects) and **imports no mediapipe**,
  which is the point: a recording already stores its edge list in `meta/pose_connections`, so
  `annotate` can draw poses without pulling the whole MediaPipe stack in for a 35-pair constant.
  Adds the two behaviours the live path never needed — it **skips NaN** points/bones (the
  recorder writes full-NaN rows for untracked frames and `annotate` scrubs across them;
  `int(nan)` raises), and **dims** landmarks below `min_visibility`, since MediaPipe
  *extrapolates* occluded points rather than dropping them and an invented coordinate otherwise
  looks identical to a measured one. The threshold matches `motor_metrics.quality.Gate`, so what
  looks solid is what the metrics will accept. `main.py:draw_pose` is a thin adapter over it.
- `main.py` — CLI (`python -m pose_estimation.main`); same flags as `video_capture.main`, plus
  `--model`. Windowed mode overlays skeleton + angles; `--no-window` prints angles.

### `face_blur/`

Privacy redaction of faces in the video stream, used by every entry point.

- **MediaPipe API note:** same Tasks-only build as `pose_estimation`. Uses `FaceDetector`
  (blaze_face short-range) in VIDEO mode. `model.py` mirrors `pose_estimation/model.py` —
  `ensure_model()` downloads/caches the ~230 KB `.tflite` under `face_blur/models/`
  (gitignored). The download helper is duplicated (not imported from `pose_estimation`) to keep
  the package decoupled; unifying them is the "factor out libraries" TODO.
- `redact.py` — style-agnostic primitives shared by both backends: `padded_bounds()`
  (pad+clamp a box, with an optional larger `top_pad` for headroom) and `redact_region()`
  (dispatch `box` = solid fill / `mosaic` = pixelation). `box` (default) is **irreversible**;
  `mosaic` is only weak de-identification (recoverable by ML re-identification).
- `blur.py` — `FaceBlurrer` (**detector** backend), a context manager. `blur(frame_bgr)` runs
  its own `FaceDetector` and redacts every face **in place**, managing its own monotonically
  increasing timestamp. Boxes are padded ~15% before redaction.
- `pose_blur.py` — `PoseFaceBlurrer` (**pose** backend). Instead of a second detector, it
  redacts the head region derived from the pose model's face keypoints (`FACE_LANDMARKS`, pose
  indices 0–10: nose/eyes/ears/mouth). `blur(frame_bgr, pose_result)` reads
  `pose_result.pose_landmarks`; the keypoint box is padded outward and further **above** (to
  cover forehead/hair the keypoints miss). No-op when no pose is present, and no extra model to
  download. More reliable than the detector when a body is tracked (odd angles, profile, small
  faces).
- `hybrid.py` — `HybridFaceBlurrer` (**hybrid** backend). Composes the other two: uses the pose
  keypoints when `pose_result` carries a detected pose, otherwise falls back to the detector.
  Covers both full/upper-body footage and close-up face-only framing where the pose model may
  not fire.
- **Interchangeable backends:** all three expose the same `blur(frame, pose_result) / close()`
  surface (the detector ignores `pose_result`), so call sites are uniform. `factory.py` →
  `build_blurrer(method, ...)` picks the backend from the CLI `--blur-method
  {detector,pose,hybrid}` choice (`BLUR_METHODS`).
- **Redaction is applied to the image sink only** (window / Rerun log / recording), always after
  detection runs on the raw frame, so pose accuracy is unaffected and only redacted frames are
  ever shown/recorded. In `--no-window` headless runs of `video_capture`/`pose_estimation` (no
  image sink) blur is skipped; `recording`/`rerun_viewer` always redact before persisting.

All four CLIs (`video_capture`, `pose_estimation`, `rerun_viewer`, `recording`) expose
`--blur-faces` / `--no-blur-faces` (**default on**), `--blur-style {box,mosaic}` (default
`box`), `--blur-method {detector,pose,hybrid}` (default `hybrid`), and `--face-model`
(detector/hybrid). The pose/hybrid backends reuse the already-computed pose result in
`pose`/`rerun`/`record`; in the `capture` demo (which has no pose loop) `--blur-method
pose|hybrid` lazily spins up a `PoseEstimator` for the displayed frames.

### `rerun_viewer/`

Streams the pipeline to the [Rerun](https://rerun.io) viewer.

- **Package note:** the visualization library is **`rerun-sdk`** (imported as `rerun`). The
  similarly named `rerun` PyPI package is an unrelated file-watcher and was replaced in
  `pixi.toml` — do not re-add it.
- `viewer.py` — `PoseRerunLogger.log_frame()` logs the video with a 2D skeleton overlay
  (`video/image`), a 3D skeleton from world landmarks (`pose3d`), and scalar line plots for
  frame rate + joint angles (`metrics/*`). Missing poses are cleared via `rr.Clear`. World
  coords are remapped (negate y/z) so the figure stands upright. Frames are JPEG-encoded
  (`cv2.imencode` on BGR → `rr.EncodedImage`) to keep the viewer's in-memory store small — unless
  the caller passes already-encoded `jpeg_bytes` (+ `image_size`, since the 2D skeleton needs the
  frame shape), which are logged verbatim; that is the replay path. `log_annotations()` logs
  labeled segments as a text log that turns on at `start_ms` and clears at `end_ms`, placed on
  **both** timelines so they show up whichever one you scrub.
- **Memory note:** when spawning (not `--save`), the viewer holds the whole stream in RAM,
  evicting oldest data past its `memory_limit`. The logger calls `rr.spawn(memory_limit=...)`
  explicitly (the `init(spawn=True)` bool form can't forward the limit).
- `main.py` — CLI (`python -m rerun_viewer.main`); capture/model flags plus `--save PATH`
  (write a `.rrd` instead of spawning), `--no-spawn`, `--memory-limit` (default `75%`; live
  path only — a no-op under `--save`), `--jpeg-quality` (default `75`), and `--record PATH.h5`
  / `--record-video PATH.mp4` (write an HDF5 recording alongside — see `recording/`). Spawns
  the viewer by default.
- `replay.py` — the **offline** direction: replays an existing HDF5 recording into Rerun, either
  as a `.rrd` (`pixi run export-rrd session.h5 out.rrd`) or straight into the viewer
  (`pixi run replay session.h5`) — same module, one optional positional output. Rebuilds the
  duck-typed pose result from the stored landmark rows and feeds the **unmodified**
  `PoseRerunLogger`, logging the archived JPEG blobs verbatim (`Recording.frame_jpeg`), so no
  decode/re-encode round trip and no re-run of MediaPipe. Everything the file carries is replayed:
  `/feather` streams turn on the sensor tab, `/annotations` the annotations tab.
  **Only the raw feather streams are replayed** — the reader adds `gravity`/`linear_accel` on read
  *and* the logger re-derives them from accel, so replaying the reader's copies would log each
  twice (`_DERIVED_STREAMS`). **Sensor alignment is nominal:** the device-clock↔recording-clock
  offset is never stored, so the first sample is anchored at the recording's `t=0`; relative
  sensor timing is exact, video↔sensor alignment is not.
- **Feather Sense (optional):** when the board is streaming (USB serial **or BLE**), the viewer
  also plots its sensors on a **"Feather Sense" tab** (accel/gyro/mag as x/y/z line plots, plus
  battery; `gravity`/`linear_accel` are **derived here** from each raw `accel` record via a
  `GravityFilter` — the board does not send them). The stream is obtained via
  `adafruit_feather_sense.open_feather(...)`: `--feather` requires the device (errors if absent),
  `--no-feather` disables the probe, default is **auto** (used only if detected);
  `--feather-transport {serial,ble}` (default serial), `--feather-port` (serial) /
  `--feather-address` (BLE) override detection. Samples are drained non-blockingly each camera
  frame and placed on the shared `time` timeline using the device clock. See
  `adafruit_feather_sense/`.

### `recording/`

Compact, SciPy-native session recording for offline analysis — an archival alternative to the
Rerun `.rrd`. Stores only the **minimal raw** signals; derived quantities are recomputed on read.

- **Philosophy:** persist the face-blurred video + the pose model's raw landmark outputs, and
  recompute anything derivable (joint angles, fps, pose-present, 2D pixel points) from those.
- **Video-in-HDF5:** frames are stored as per-frame JPEG blobs in a `vlen` uint8 dataset (frame
  `i` aligns with landmark row `i`). Chosen over an mp4 sidecar because **H.264 is unavailable**
  in this OpenCV build (VideoWriter only does `mp4v`/`XVID`/`MJPG`).
- `recorder.py` — `HDF5Recorder`, a context manager. `append(frame_bgr, timestamp_ms, result)`
  writes JPEG + landmark rows (NaN rows when no pose); datasets are resizable/gzip'd and created
  lazily on the first frame; metadata (landmark names, 35-pair connections, image size, model +
  mediapipe version, blur style, coordinate conventions) is written as attrs. Optional
  `video_path` also writes a parallel `mp4v` file.
- **Feather Sense (optional):** `append_sensor(name, timestamp_ms, values, fields)` writes each
  sensor stream to its own lazily-created `/feather/<name>` group (own `timestamps_ms` +
  `(M, K)` float32 `values`, or `source`/`message` strings for `error`), grown independently
  since streams are async/multi-rate. `recording/main.py` gains the same feather flags as the
  rerun CLI (`--feather` / `--no-feather` / `--feather-transport {serial,ble}` / `--feather-port`
  / `--feather-address`). Read back via `Recording.feather` (identical regardless of transport).
  Only the board's **raw** streams are stored (accel/gyro/mag/battery/error) — per the minimal-raw
  rule, `gravity`/`linear_accel` are derived on read, never written.
- `reader.py` — `Recording`, a read-only loader exposing arrays (`landmarks_world`, etc.),
  `pose_present`, `fps()`, `frame(i)` (JPEG-decoded), `angles()` (recomputed via
  `pose_estimation.angles.joint_angles`, returned as a pandas DataFrame), `annotations` (a
  read-only snapshot of any labeled time segments, `[]` on recordings without them), and
  `feather` (a `{stream: arrays}` dict of any recorded Feather Sense data, `{}` if none).
  `feather` additionally carries `gravity`/`linear_accel`, **derived on read** from the stored raw
  accel via `adafruit_feather_sense.motion` (each flagged `derived=True`; same pattern as
  `angles()`). `motion(tau_s=…)` re-derives them with a different filter time constant — the point
  of not baking the filter into the capture.
- `annotations.py` — `AnnotationStore`, a context manager that opens an **existing** recording
  `"r+"` to add/edit labeled **time segments** (an interval table, not per-frame). `add(label,
  start_ms, end_ms)`, `list()`, `delete(index)`; the optional `/annotations/{label,start_ms,
  end_ms,deleted}` group is created lazily on first `add` (absent on older recordings). Deletes
  are **tombstones** (a `deleted` flag), so row indices stay stable; writes flush immediately.
  Overlapping/concurrent labels are supported since each segment is its own row. Written by the
  `annotate/` tool; read back read-only via `Recording.annotations`.
- `export.py` — `export_mp4()` / `python -m recording.export session.h5 out.mp4` reconstructs an
  mp4 (mp4v) from the stored JPEG frames; fps defaults to the median of recorded timestamps.
- `main.py` — standalone CLI (`python -m recording.main`); same capture/pose/blur flags as the
  other CLIs plus `--output` and `--video`.

### `annotate/`

Post-hoc labeling tool: scrub an already-recorded `.h5` in an OpenCV window and attach text
labels to time segments (stored via `recording.annotations.AnnotationStore`).

- `main.py` — CLI (`python -m annotate.main session.h5` / `pixi run annotate session.h5`).
  Displays `Recording.frame(i)` (already face-blurred) with a bottom timeline strip showing the
  playhead and existing segments as colored, lane-stacked spans. Keys: `,`/`.` step a frame,
  `<`/`>` jump ~1s, `space` play/pause, `i`/`o` mark in/out (then type a label at the terminal
  prompt), `x` delete the nearest segment, `p` toggle the pose overlay, `?` help, `q`/`Esc` quit.
  Edits save immediately.
- **Labels are named on screen, not just colored.** `_active_labels()` names the segment(s) under
  the playhead in that label's own strip color, and a swatch→label legend sits above the strip.
  Color alone was never enough: the `motor_metrics.labels` vocabulary makes trials differ only by
  their params (`arms=free` vs `arms=prop`), so spans that matter are near-identical. Text is
  drawn with a black outline (`_put_text`) because it lands over arbitrary footage. Overlapping
  segments are normal, so the readout is a list.
- **Pose overlay (`p`, default on)** — draws the stored landmarks via
  `pose_estimation.draw.draw_skeleton`, using `rec.pose_connections` from the file (no mediapipe
  import). It is **drawn before the strip rectangle** — the strip is painted *over* the bottom
  `_STRIP_H` px of the same image, so drawing after would run limbs across the timeline (pinned
  by an equality check on the strip region with the overlay on vs off). Dimmed limbs mark
  low-visibility/extrapolated landmarks, which is the QC signal the data-collection runbook asks
  the operator to eyeball. `main()` caches `rec.pose_present` **once** — it is a property that
  re-slices the whole `(N,33,3)` world array on every access.
- **HDF5 locking note:** the tool holds two handles on the same file — `AnnotationStore` (`"r+"`)
  and `Recording` (`"r"`). h5py requires the **`"r+"` handle be opened first**; opening `"r"`
  before `"r+"` on one path in one process raises `OSError`. `main()` opens the store before the
  reader for exactly this reason (pinned by `test_rw_then_ro_handle_coexist`).

### `motor_metrics/`

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
  and pinned (0.997 at 0.25 Hz → 0.95 at 1 Hz → 0.61 at 3 Hz) — a *bias*, identical for every
  trial, so it cancels within-child but makes the magnitudes non-comparable to any other pipeline.
  First and only use of scipy in the repo. `smooth` returns NaN below the window, never raises.
- `segments.py` — annotations → frame spans. Drops unparseable labels and anything overlapping an
  `exclude` **whole** (a hold whose middle is untrustworthy is not a shorter valid hold; trimming
  would invent a boundary). Mark two trials around the excluded stretch to keep the good parts.
- `hold.py` — sitting **and** supported standing (one function; the label carries the difference).
  Duration + sway (path length, 95 % ellipse, RMS, ML/AP split, mean velocity) + trunk-angle stats.
  **`path_length_m` is duration-confounded** — a worse 20 s hold beats a better 8 s one on it; use
  `mean_velocity_mps` or a common `window_s`. **Ellipse area reads ~0 for one-axis rocking** (a
  line encloses nothing), so read it next to the ML/AP RMS, never alone.
- `transition.py` — `sparc` (spectral arc length) is primary on trunk angular speed. **Read only
  against itself**: at 30 Hz the absolute value is a pipeline artifact. Its measured noise
  robustness has a **ceiling** — flat to ~2 % of peak speed, erratic past ~5 % (sd 0.24) — so big
  brisk transitions score reliably and small slow ones may not; prefer the median of several.
  It separates *fluid from effortful* but does **not** count corrections (more submovements
  eventually score *better* as they blend); `count_submovements` counts. `symmetry_index` is a
  **between-trial** comparison grouped by the label's `side=`.
- `crawl.py` — **the GMFM item's "1.8 m" is not measurable here** (hip-centered frame, one camera,
  no depth, no IMU); `speed_norm_per_s` is in **image widths/s, never metres**. The deliverable is
  the *pattern*, which the pelvis-centered frame captures perfectly: cadence, `cycle_period_cv`,
  and `phase_offset` (**0.5 = reciprocal/mature, 0.0 = symmetric "bunny" haul**). Limb signal is
  the wrist projected on the **trunk axis** — in prone there is no useful vertical.
  `MIN_CYCLE_EXCURSION_M` is **not optional**: the prominence gate alone is relative to the
  signal's own range, so it normalizes pure jitter up into a textbook crawl (measured: 57 cycles
  from noise, a still child reporting a confident fictional cadence).
- `report.py` / `main.py` — `metrics_table(rec)` → one row per trial, columns the **union** across
  exercises (so they concatenate into a trend); `session_table(paths)` for the cross-session view,
  which is the whole point given there is no external GMFM score to calibrate against. Label params
  are prefixed `p_` to stay clear of metric fields of the same name; `warnings` carries the label QC.
  `duration_s`/`tracked_s`/`coverage`/`n_frames` mean the same thing in **every** metric dataclass
  so the union table has no per-exercise holes. CLI: `pixi run metrics session.h5 [--csv out.csv]`.
- `notebooks/motor_metrics.ipynb` (`pixi run notebook`) — label inventory + coverage QC → **the
  vertical-reference diagnostic** (check it before trusting any hold number) → per-exercise tables
  → sway/limb plots → cross-session trend.

### `list_devices/`

Discovery helper: which capture devices can the other CLIs actually use? Compatibility is
defined exactly as `video_capture` uses it — a camera *index* that `cv2.VideoCapture` can open
**and read a frame from** — so a listed device is one you can pass to any entry point via
`--source <index>`.

- `devices.py` — `DeviceInfo` (index, default + max resolution, fps, backend, plus V4L2
  `name`/`node`) and `enumerate_devices()`, which probes indices `0..max_index` (extended on
  Linux to cover any higher `/dev/videoN` node). `probe_index()` mirrors `VideoCapture.open()` +
  `read()` so phantom/metadata-only nodes that open but never yield a frame are filtered out; it
  also requests an oversized frame and reads the clamped value back to discover each device's
  **max** supported width/height. OpenCV's logger is silenced during the scan so probing empty
  indices is quiet.
- `main.py` — CLI (`python -m list_devices.main`); prints a per-device summary with default and
  max resolution, the `--source <index>` to reuse, and a ready-to-paste
  `--source/--width/--height` line for max-res capture. `--json` for machine-readable output,
  `--max-index` to widen the scan. Exit code `1` (not `0`) when no device is found.

### `adafruit_feather_sense/`

CircuitPython app for the **Adafruit Feather Bluefruit Sense (nRF52840)** that streams its
onboard sensors over **USB serial or BLE** (interchangeable transports, same wire protocol), plus
host-side readers. Mixed runtime: some files run on the board, some on the host, one is shared.
See `adafruit_feather_sense/README.md` for the full protocol spec, `circup` list, and deploy steps.

- **On the board** (deploy = copy the four shared modules + the chosen entry to the CIRCUITPY
  root as `code.py`; libs via `circup install adafruit_lsm6ds adafruit_lis3mdl neopixel`):
  `sensors.py`
  (`SensorHub(imu_hz=...)` — **raw signals only**: IMU with LSM6DS33/TR-C fallback, LIS3MDL,
  battery via `board.VOLTAGE_MONITOR`; the I2C bus is opened at **400 kHz**, not the 100 kHz
  `busio` default — measured ~1.8× per read). Environmental sensing (BMP280 temp/pressure/altitude,
  SHT31-D humidity) was **removed**: forced-mode conversions blocked the loop ~152 ms/s (~15 % of
  wall, ~7.6 lost IMU samples/s) for unused 1 Hz data. Re-add only with the BMP280 in
  `MODE_NORMAL`. `read_imu()` returns accel **and** gyro from **one 12-byte burst** at reg `0x22`
  (they are contiguous on the chip; driver sets `_bdu`), so they share **one sample instant and one
  timestamp** — the point is simultaneity, not just the halved I2C cost; don't split it back into
  two property reads. The IMU **ODR is set, never inherited**: `_odr_for` picks the slowest rate
  that oversamples the poll 2× (100 Hz → 208). The driver's 104 Hz default is a fine 2.08× at
  `imu_hz=50` and a broken 1.04× at 100. Don't raise it further — ODR above 2× buys noise (~√2 RMS
  per doubling), not information.
  Then `telemetry.py` (`Telemetry.pump(now_ms, emit) -> next_due_ms`
  — the transport-agnostic sample/schedule/encode loop, rates are ctor args; it defers its
  `sensors` import so the host can import it, which is what makes the schedule testable); and
  **two entry points, a literal `code.py` each**: `board/serial/code.py` (USB, `emit =
  sys.stdout.buffer.write`, IMU **100 Hz**) and `board/ble/code.py` (Nordic UART peripheral named
  `FeatherSense` via `adafruit_ble` `UARTService`, `emit = uart.write`, IMU **50 Hz** — the link
  measures ~100 Hz IMU before saturating, so 50 has ~2× margin; the old "~1–2 KB/s" figure was
  never measured and understated it ~2×). Each entry builds one `SensorHub` and injects it
  (`Telemetry(hub=hub)`), passing the same `imu_hz` to both so the ODR and the poll agree.
  `Telemetry` also takes `on_battery(percent, usb_connected)` — an optional sink fired from the
  existing battery slot, so a display can piggyback that read instead of polling (used by the
  status LED; must not raise, or `pump`'s handler eats the battery frame) — and
  `on_pulse(now_ms)` / `pulse_hz` (default 15), a second sink on its **own** slot for a consumer
  that must animate rather than observe (the LED's charging ramp; 0.2 Hz can't drive one). It
  reads no sensor and puts **nothing on the wire**, and is left out of the schedule entirely when
  unset.
- **The schedule runs on integer milliseconds — never `time.monotonic()`.** CircuitPython builds
  **single-precision** floats, so `monotonic()`'s ULP grows with uptime (~2 ms at 5.6 h, ~4 ms at
  10 h). Combined with the old `due = now + interval` (which rescheduled from the *observed* time
  and so baked each cycle's lateness in permanently), the sample rate **silently decayed with
  uptime**: 48.9 Hz just after a reflash but **43.5 Hz at 5.6 h**, against a 50 Hz nominal, with
  timestamps still correct so nothing looked wrong. Every pre-2026-07-16 benchmark in the README
  was taken right after a flash, which is why nobody saw it. `pump` now takes integer `now_ms`
  (from `monotonic_ns`) and advances `due += interval` from the deadline, with a clamp that drops
  the backlog after a long stall instead of bursting stale samples: **100.0 % of nominal at 5.6 h
  uptime**. Don't reintroduce float seconds here, and don't "simplify" the reschedule back to
  `now + interval`.
- **Measured rate chain (accel):** 41 → 48.3 (env removal) → 48.8 (400 kHz) *[all at ~0 uptime;
  43.5 at 5.6 h]* → **50.0** (integer-ms schedule + drift fix, at 5.6 h) → **100.0**
  (burst read + ODR 208 + `imu_hz=100`). Loop **ceiling** (probe with `imu_hz=400`, the only way
  to see a loop win while `imu_hz` still caps the rate): 191.5 → 255.5 (fused `encode_xyz`) →
  **290.5 Hz** (`split`-based `cobs_encode`). So the shipped 100 Hz build is ~34 % of ceiling,
  **~66 % idle**.
- **Status LED:** `status_led.py` (board) — lights the onboard NeoPixel red/yellow/green by
  battery level (<25 / 25–60 / >60 %, ±3 % hysteresis per edge, edges exclusive-below, so a
  reading resting on a threshold can't flicker), and **pulses that same color while charging**
  (12→100→12 % over 2 s, quantized to 16 steps so the "write only on change" guard still catches).
  Color means level and only level; charging is carried by the animation. Nothing latches — the
  band moves both ways. **Display only:** never on the wire, no protocol change.
  **USB picks the animation, never the band.** `VOLTAGE_MONITOR` reads the **battery terminal**,
  not the charger's output: measured while charging, two packs gave 4.00 V/80.3 % and
  4.09 V/89.5 %, so the estimate still tracks the pack when plugged in. The README's old "reads
  green on USB regardless of the pack" gotcha was never measured and is **wrong** — a first cut of
  the charging feature believed it, capped the band while on USB (ceiling = last band seen
  unplugged, yellow when none) and, since that ceiling lives in RAM and the board almost always
  boots plugged in, pinned the LED to a permanent amber. Don't reintroduce a cap. Charging does
  elevate the reading slightly; no offset corrects it because none has been measured — add one
  only with a number attached, and only if a low pack is seen reading high on the charger.
  Deliberately subordinate to streaming, and the shape is **measurement-driven**: it is driven by
  `Telemetry(on_battery=led.update)`, adding **no per-iteration call** to the sampling loop. The
  first cut called `StatusLED.tick(now)` from `code.py` each iteration and measured **-0.93 accel
  samples/s** (49.14 → 48.22, median 49 → 48) — of which **-0.59 was the bare guard check**, not
  the ADC read: a method call here is ~150 µs and the loop turned ~50-60×/s. Riding the battery
  slot costs **-0.21** (median 49). Don't reintroduce a per-iteration LED call — **the rule got
  stricter**, since the loop now turns ~100-120×/s, so any per-iteration cost roughly doubles.
  The charging ramp can't ride a 0.2 Hz slot, so it takes its **own** `Telemetry` slot
  (`on_pulse=led.pulse` at 15 Hz) rather than a call in `code.py` — same rule, one more scheduled
  slot; off USB `pulse` returns on a single attribute test. Measured on the board while charging
  (so the ramp was live): **accel 100.0/s device-side, exactly nominal**, no error frames. That
  **bounds** the cost, it doesn't measure it — at ~34 % of loop ceiling, ~2 ms/s of extra work
  can't move the rate. Probe with `imu_hz=400` if the real per-call cost ever matters.
  `tick(now)` survives **only** for the BLE advertising wait, where `pump`
  isn't running so nothing drives `on_battery`/`on_pulse`, and the idle loop makes the call free. `update`
  never raises (an escaping error would trip the BLE re-advertise handler) and writes the pixel
  only when the band changes. `board`/`neopixel` are imported in `__init__`, not at module scope,
  so a board missing the lib degrades to no LED instead of a crash loop — and `band_for` plus the
  write path (via an injected `pixel`) stay host-importable and unit-tested.
- **Shared** (board + host, pure `struct`): `feather_protocol.py` — a **TLV-over-COBS** wire
  protocol. Each sample is one COBS-framed record `[type][len][timestamp_u32][int32…]`
  terminated by `0x00`; **no floats on the wire** — values are scaled fixed-point int32 (shared
  `SCALES`), converted SI↔int by `to_raw`/`to_si`. Live types `0x01`–`0x05`: accel/gyro/mag/
  battery + `error` (streamed on any caught sampling/encode failure). `0x06` gravity / `0x07`
  linear_accel are **host-derived pseudo-types, never on the wire** (no `SCALES` entry — built in
  SI). Codes are dense and carry **no compatibility guarantee** — board and host ship from this one
  file, so renumbering is resolved by reflashing; recordings are unaffected (`/feather` groups are
  keyed by stream *name*). Add a stream = append the next free code.
  **`encode_xyz` is the board's fast path** for the three-axis types: it fuses `to_raw` + pack +
  frame into a single `struct.pack` (~15 interpreted ops → one C call), worth **+33 % of loop
  ceiling** — `encode` is the loop's most expensive step, so this is where board wins live.
  `cobs_encode` uses `bytes.split` (turns the interpreted loop once per *chunk*, not per *byte*;
  +14 %). Both are **byte-identical** to the originals and must stay so — `tests/test_feather.py`
  keeps the old encoders verbatim as oracles and fuzzes ~8000 cases plus the 254-byte block-split
  boundary. Note for future optimisers: a `bytes.find`-based COBS is a **0.63× regression**,
  because a real payload is half zero bytes (small fixed-point int32s), so there are no long runs
  to scan — profile the payload, not just the code.
- **Host-derived motion:** `motion.py` (host-only) — `GravityFilter.update(ts_ms, xyz)` /
  `derive_motion(ts, values, tau_s)` reconstruct `gravity` (single-pole low-pass, `GRAVITY_TAU_S`)
  and `linear_accel = accel − gravity` from the raw accel stream. This used to run on the board
  (`SensorHub.read_motion`), which made one accel read cost three encoded frames; `encode` is the
  loop's most expensive step (~1.275 ms/frame), so moving it here removed two of every four
  frames/cycle. It also keys the filter off the *device* timestamp (true sample spacing) and makes
  `tau_s` a read-time choice.
- **On the host** — both transports share one interface (`poll()` → SI `SensorRecord`s, `errors`,
  `close()`, `.port`, `open_if_available`), so apps are transport-agnostic. `stream.py` —
  `FeatherSenseStream` (pyserial, non-blocking) + `FrameRecordDecoder` (the shared bytes→records
  decode). `ble_stream.py` — `FeatherSenseBLEStream` (**bleak**; scan/connect/notify on the Nordic
  UART Service run on a background asyncio thread, a queue bridges to the sync `poll()`).
  `read_stream.py` / `read_ble.py` are the standalone pretty-printer CLIs. `__init__.py` exposes
  `open_feather(transport, *, port=None, address=None)` (lazy-imports the backend) — used by
  `rerun_viewer`/`recording`. Import note: host modules add the package dir to `sys.path` so the
  shared `feather_protocol` resolves as a top-level module; **don't import the board files**
  (`board/*/code.py`, `sensors.py`, `telemetry.py`) on the host. `status_led.py` is the
  deliberate exception — it defers its `board`/`neopixel` imports into `StatusLED.__init__`, so
  the module imports cleanly on the host and its pure `band_for` is unit-tested there.

## Environment & commands

The project uses [Pixi](https://pixi.sh) (conda-forge + PyPI) for dependency and environment
management. The env is pinned to `linux-64` and Python 3.14.

- `pixi install` — resolve/create the `default` environment from `pixi.toml` / `pixi.lock`.
- `pixi run <cmd>` — run a command inside the environment (e.g. `pixi run python script.py`).
- `pixi shell` — drop into an activated shell.
- `pixi run jupyter lab` — start Jupyter (the `jupyter` package is a dependency).
- `pixi add <pkg>` — add a conda dependency; `pixi add --pypi <pkg>` for a PyPI dependency.

Tasks defined under `[tasks]` in `pixi.toml`:

- `pixi run list-devices` — list capture devices and the `--source` index to use for each.
- `pixi run capture` — run the live webcam demo (`python -m video_capture.main`).
- `pixi run capture-headless` — read 30 frames with no window; useful where there's no display.
- `pixi run pose` — live pose skeleton + joint-angle overlay from the webcam.
- `pixi run pose-headless` — 30 frames, no window; prints joint angles.
- `pixi run rerun` — stream webcam + skeleton + metrics to the Rerun viewer.
- `pixi run rerun-headless` — 30 frames, no viewer; writes `recording.rrd` (open with `rerun recording.rrd`).
- `pixi run record` — record webcam + pose to `recording.hdf5` for offline analysis.
- `pixi run record-headless` — record 30 frames to `recording.hdf5` (bounded run).
- `pixi run export-video <in.hdf5> <out.mp4> [--fps N]` — rebuild an mp4 from a recording's stored frames.
- `pixi run export-rrd <in.hdf5> <out.rrd>` — convert a recording to a Rerun `.rrd` (video +
  skeleton + angles, plus feather/annotations when present); open with `rerun out.rrd`.
- `pixi run replay <in.hdf5>` — same conversion, straight into a spawned Rerun viewer.
- `pixi run annotate <session.hdf5>` — scrub a recording and label time segments (edits saved in place).
- `pixi run metrics <session.hdf5> [--csv out.csv] [--exercise sit_hold]` — motor metrics for the
  labeled trials in a recording (`--window-s N` to compare holds of unequal length fairly).
- `pixi run notebook` — Jupyter Lab in `notebooks/` (offline metric exploration).
- `pixi run test` — run the unit-test suite (verbose). `pixi run test-quiet` for the terse summary.

When adding a build/lint/test workflow, wire it up as a Pixi task so it's captured in the repo
rather than run ad hoc.

## Tests

`tests/` holds `unittest` coverage for the reusable libraries (not the CLIs). Run with `pixi run
test`. Every external boundary is mocked so the suite needs no camera, network, model download,
display, or GPU; ~480 tests run in a few seconds:

- `tests/fakes.py` — shared duck-typed stand-ins: MediaPipe landmark/pose/detection results, an
  opened `cv2.VideoCapture` (`FakeCapture`), `cv2.VideoWriter` (`FakeVideoWriter`), and a
  pyserial handle (`FakeSerial`, fed pre-baked protocol bytes). Plus `fake_recording()` (a
  duck-typed `Recording` — the `motor_metrics` functions read only attributes, so their tests
  need no HDF5) and `body_world()` (synthetic *anatomy*: `make_landmarks` spreads points along a
  diagonal, which is fine for pass-through tests but is not a body, and postural metrics need one).
- `tests/test_motor_metrics.py` — the metrics package. Pure logic runs unmocked against real
  numpy/scipy and is pinned to **closed forms, not recorded outputs** (polygon perimeter for path
  length; `5.991·π·σ²` for the sway ellipse; constructed lean angles; sine frequency for cadence;
  anti-phase → 0.5). Where no closed form exists the tests pin **ordering or invariance**:
  SPARC's absolute value is a pipeline artifact, so asserting a number would pin the noise. Three
  regression pins earn their keep: the world mid-hip COM proxy being identically zero, the
  `derive.py` filter's frequency response, and `MIN_CYCLE_EXCURSION_M` (without it pure jitter
  reads as 57 crawl cycles). Every metric is exercised for empty / single-frame /
  shorter-than-window / fully-untracked segments — all must return NaN, never raise, since one
  mis-marked annotation must not take down a 40-row report. Integration tests drive the real
  `HDF5Recorder` → `AnnotationStore` → `Recording` → `metrics_table` path against temp files.
- `tests/test_annotate.py` — the `annotate` GUI's drawable logic (the loop itself, needing a
  window and keyboard, is not tested). `draw_skeleton` against real numpy/cv2: all-NaN rows
  no-op, partial NaN skips only the affected bones, `(0.5, 0.5)` lands on the frame center, and
  low visibility dims (a bone taking the *weaker* of its two endpoints). Plus `_active_labels`
  (inclusive boundaries, overlapping spans, empty gaps) and `_render` smoke tests over a fake
  `Recording` for pose on/off, untracked frames, and with/without annotations — one mis-marked
  annotation or one untracked frame must not take down the window.
- `tests/test_feather.py` — the Feather Sense host integration: `FeatherSenseStream` decode/poll
  and `open_if_available` probe (via `FakeSerial`), the shared `FrameRecordDecoder`, the
  `open_feather` transport dispatch (serial/ble backends mocked — no real radio), the host
  `motion` derivation (`GravityFilter`/`derive_motion` — seeding, tilt bleed, transients, clock
  wrap, batch/live agreement), the recorder's `/feather` datasets + `Recording.feather` (including
  that derived streams are *not* stored and that `motion(tau_s=…)` re-derives), and the viewer's
  `log_sensors` + its derived plots (with `rr` mocked), and the status LED (`band_for` bands +
  hysteresis in both directions, plus `StatusLED.update`/`pulse`/`tick` against an injected
  `FakePixel` — pinning that it writes only on a change, never raises, no-ops without a pixel, and
  that the charging rules hold: USB selects the animation and never the band (the capping
  regression), an almost-full pack reads green on the charger while a low one still reads red, the
  ramp keeps the hue and never goes dark, and the 0.2 Hz battery slot doesn't stamp full
  brightness over a ramp in progress). No
  board, serial port, or BLE adapter needed.
  Also **`TelemetryScheduleTests`** — the board's sampling schedule, driven by a fake hub and a
  fake clock (reachable because `telemetry.py` defers its `sensors` import; `status_led.py` is the
  same trick). It pins the two bugs that cost the rate: no erosion under **sustained lateness**
  (the `due = now + interval` regression) and no erosion at **high uptime** (the float32
  `monotonic()` decay), plus the stall clamp, one-read-serves-both-streams, and that accel/gyro
  share a timestamp. Plus **`CobsEncodeTests`/`EncodeXyzTests`** (byte-identity fuzz vs. the
  original encoders) and **`RateTrackerTests`** (the `--stats` arithmetic — it is the acceptance
  instrument, and it used to over-report ~10 %).
- `tests/test_viewer.py` — the logger's calls (with `rr` mocked) plus **`ReplayRecordingTests`**,
  which drives the real `HDF5Recorder` → `AnnotationStore` → `Recording` → `replay_recording` path
  against temp files: one `video/image` per frame, `rr.Clear` on untracked frames, feather streams
  logged once (the double-derive trap), annotations logged, and the blueprint reflecting what the
  file actually carries. Blueprints have no useful `repr`, so assertions walk `root_container`
  for view origins (`_origins`) rather than matching on strings.
- `cv2.VideoCapture`/`VideoWriter` are patched; the native MediaPipe `PoseLandmarker`/
  `FaceDetector` are patched at their import site (`ensure_model` + the class) so no model loads;
  `urllib.request.urlretrieve` is patched in the `ensure_model` tests; the `rerun` SDK is patched
  (`rerun_viewer.viewer.rr`). HDF5 recordings are written to real temp files (fast, self-contained).
- Pure logic (`redact`, `angles`, `pose_blur`, `factory`) runs unmocked against real NumPy/OpenCV.
- Discovery is `python -m unittest discover -s tests -t .`; run from the repo root so the packages
  import. `tests/` is a package (`__init__.py`) so `from tests.fakes import …` resolves.

## Notes

- `pixi.lock` is generated — do not hand-edit; it is marked `linguist-generated` and uses a
  binary merge strategy to avoid 3-way merge conflicts.
- `.pixi/` is the local environment install directory and is not source.
