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
  index, anything else as a path/URL.

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
- `reader.py` — `Recording`, a read-only loader exposing arrays (`landmarks_world`, etc.),
  `pose_present`, `fps()`, `frame(i)` (JPEG-decoded), and `angles()` (recomputed via
  `pose_estimation.angles.joint_angles`, returned as a pandas DataFrame).
- `export.py` — `export_mp4()` / `python -m recording.export session.h5 out.mp4` reconstructs an
  mp4 (mp4v) from the stored JPEG frames; fps defaults to the median of recorded timestamps.
- `main.py` — standalone CLI (`python -m recording.main`); same capture/pose/blur flags as the
  other CLIs plus `--output` and `--video`.

### `list_devices/`

Discovery helper: which capture devices can the other CLIs actually use? Compatibility is
defined exactly as `video_capture` uses it — a camera *index* that `cv2.VideoCapture` can open
**and read a frame from** — so a listed device is one you can pass to any entry point via
`--source <index>`.

- `devices.py` — `DeviceInfo` (index, resolution, fps, backend, plus V4L2 `name`/`node`) and
  `enumerate_devices()`, which probes indices `0..max_index` (extended on Linux to cover any
  higher `/dev/videoN` node). `probe_index()` mirrors `VideoCapture.open()` + `read()` so
  phantom/metadata-only nodes that open but never yield a frame are filtered out. OpenCV's
  logger is silenced during the scan so probing empty indices is quiet.
- `main.py` — CLI (`python -m list_devices.main`); prints a per-device summary with the
  `--source <index>` to reuse. `--json` for machine-readable output, `--max-index` to widen the
  scan. Exit code `1` (not `0`) when no device is found.

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

When adding a build/lint/test workflow, wire it up as a Pixi task so it's captured in the repo
rather than run ad hoc.

## Notes

- `pixi.lock` is generated — do not hand-edit; it is marked `linguist-generated` and uses a
  binary merge strategy to avoid 3-way merge conflicts.
- `.pixi/` is the local environment install directory and is not source.
