# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

Early stage. "Playing with tools inspired by Remy's therapies" (per README). The dependency
set points at computer-vision / motion work: `mediapipe`, `opencv`, `rerun-sdk` (visualization),
plus `numpy`/`scipy`/`pandas` and `h5py` for data.

Current code: `video_capture/` (OpenCV capture), `pose_estimation/` (MediaPipe pose,
consuming `video_capture`), `face_blur/` (MediaPipe face redaction), `rerun_viewer/`
(logs the pipeline to the Rerun viewer), `recording/` (HDF5 recording for offline analysis),
and `list_devices/` (enumerate compatible capture devices).

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
  (`cv2.imencode` on BGR → `rr.EncodedImage`) to keep the viewer's in-memory store small.
- **Memory note:** when spawning (not `--save`), the viewer holds the whole stream in RAM,
  evicting oldest data past its `memory_limit`. The logger calls `rr.spawn(memory_limit=...)`
  explicitly (the `init(spawn=True)` bool form can't forward the limit).
- `main.py` — CLI (`python -m rerun_viewer.main`); capture/model flags plus `--save PATH`
  (write a `.rrd` instead of spawning), `--no-spawn`, `--memory-limit` (default `75%`; live
  path only — a no-op under `--save`), `--jpeg-quality` (default `75`), and `--record PATH.h5`
  / `--record-video PATH.mp4` (write an HDF5 recording alongside — see `recording/`). Spawns
  the viewer by default.
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
  prompt), `x` delete the nearest segment, `?` help, `q`/`Esc` quit. Edits save immediately.
- **HDF5 locking note:** the tool holds two handles on the same file — `AnnotationStore` (`"r+"`)
  and `Recording` (`"r"`). h5py requires the **`"r+"` handle be opened first**; opening `"r"`
  before `"r+"` on one path in one process raises `OSError`. `main()` opens the store before the
  reader for exactly this reason (pinned by `test_rw_then_ro_handle_coexist`).

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

- **On the board** (deploy = copy the shared trio + the chosen entry to the CIRCUITPY root as
  `code.py`; libs via `circup install adafruit_lsm6ds adafruit_lis3mdl`): `sensors.py`
  (`SensorHub` — **raw signals only**: IMU with LSM6DS33/TR-C fallback, LIS3MDL, battery via
  `board.VOLTAGE_MONITOR`; the I2C bus is opened at **400 kHz**, not the 100 kHz `busio` default —
  measured ~1.8× per read). Environmental sensing (BMP280 temp/pressure/altitude, SHT31-D
  humidity) was **removed**: forced-mode conversions blocked the loop ~152 ms/s (~15 % of wall,
  ~7.6 lost IMU samples/s) for unused 1 Hz data. Re-add only with the BMP280 in `MODE_NORMAL`.
  Measured IMU rate: 41 Hz → 48.3 Hz (env removal) → **48.8 Hz** (400 kHz), 97.5 % of the 50 Hz
  nominal, with the loop now **58 % idle** — headroom to raise `imu_hz`, not just a faster number.
  Then `telemetry.py` (`Telemetry.pump(now,
  emit)` — the transport-agnostic sample/schedule/encode loop, rates are ctor args); and **two
  entry points, a literal `code.py` each**: `board/serial/code.py` (USB, `emit =
  sys.stdout.buffer.write`, full rates) and `board/ble/code.py` (Nordic UART peripheral named
  `FeatherSense` via `adafruit_ble` `UARTService`, `emit = uart.write`, reduced IMU rate for the
  ~1–2 KB/s link).
- **Shared** (board + host, pure `struct`): `feather_protocol.py` — a **TLV-over-COBS** wire
  protocol. Each sample is one COBS-framed record `[type][len][timestamp_u32][int32…]`
  terminated by `0x00`; **no floats on the wire** — values are scaled fixed-point int32 (shared
  `SCALES`), converted SI↔int by `to_raw`/`to_si`. Live types `0x01`–`0x05`: accel/gyro/mag/
  battery + `error` (streamed on any caught sampling/encode failure). `0x06` gravity / `0x07`
  linear_accel are **host-derived pseudo-types, never on the wire** (no `SCALES` entry — built in
  SI). Codes are dense and carry **no compatibility guarantee** — board and host ship from this one
  file, so renumbering is resolved by reflashing; recordings are unaffected (`/feather` groups are
  keyed by stream *name*). Add a stream = append the next free code.
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
  (`board/*/code.py`, `sensors.py`, `telemetry.py`) on the host.

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
- `pixi run annotate <session.hdf5>` — scrub a recording and label time segments (edits saved in place).
- `pixi run test` — run the unit-test suite (verbose). `pixi run test-quiet` for the terse summary.

When adding a build/lint/test workflow, wire it up as a Pixi task so it's captured in the repo
rather than run ad hoc.

## Tests

`tests/` holds `unittest` coverage for the reusable libraries (not the CLIs). Run with `pixi run
test`. Every external boundary is mocked so the suite needs no camera, network, model download,
display, or GPU and runs in well under a second:

- `tests/fakes.py` — shared duck-typed stand-ins: MediaPipe landmark/pose/detection results, an
  opened `cv2.VideoCapture` (`FakeCapture`), `cv2.VideoWriter` (`FakeVideoWriter`), and a
  pyserial handle (`FakeSerial`, fed pre-baked protocol bytes).
- `tests/test_feather.py` — the Feather Sense host integration: `FeatherSenseStream` decode/poll
  and `open_if_available` probe (via `FakeSerial`), the shared `FrameRecordDecoder`, the
  `open_feather` transport dispatch (serial/ble backends mocked — no real radio), the host
  `motion` derivation (`GravityFilter`/`derive_motion` — seeding, tilt bleed, transients, clock
  wrap, batch/live agreement), the recorder's `/feather` datasets + `Recording.feather` (including
  that derived streams are *not* stored and that `motion(tau_s=…)` re-derives), and the viewer's
  `log_sensors` + its derived plots (with `rr` mocked). No board, serial port, or BLE adapter
  needed.
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
