# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

Early stage. "Playing with tools inspired by Remy's therapies" (per README). The dependency
set points at computer-vision / motion work: `mediapipe`, `opencv`, `rerun-sdk` (visualization),
plus `numpy`/`scipy`/`pandas` and `h5py` for data.

Current code: `video_capture/` (OpenCV capture), `pose_estimation/` (MediaPipe pose,
consuming `video_capture`), and `rerun_viewer/` (logs both to the Rerun viewer).

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
  path only — a no-op under `--save`), and `--jpeg-quality` (default `75`). Spawns the viewer
  by default.

## Environment & commands

The project uses [Pixi](https://pixi.sh) (conda-forge + PyPI) for dependency and environment
management. The env is pinned to `linux-64` and Python 3.14.

- `pixi install` — resolve/create the `default` environment from `pixi.toml` / `pixi.lock`.
- `pixi run <cmd>` — run a command inside the environment (e.g. `pixi run python script.py`).
- `pixi shell` — drop into an activated shell.
- `pixi run jupyter lab` — start Jupyter (the `jupyter` package is a dependency).
- `pixi add <pkg>` — add a conda dependency; `pixi add --pypi <pkg>` for a PyPI dependency.

Tasks defined under `[tasks]` in `pixi.toml`:

- `pixi run capture` — run the live webcam demo (`python -m video_capture.main`).
- `pixi run capture-headless` — read 30 frames with no window; useful where there's no display.
- `pixi run pose` — live pose skeleton + joint-angle overlay from the webcam.
- `pixi run pose-headless` — 30 frames, no window; prints joint angles.
- `pixi run rerun` — stream webcam + skeleton + metrics to the Rerun viewer.
- `pixi run rerun-headless` — 30 frames, no viewer; writes `recording.rrd` (open with `rerun recording.rrd`).

When adding a build/lint/test workflow, wire it up as a Pixi task so it's captured in the repo
rather than run ad hoc.

## Notes

- `pixi.lock` is generated — do not hand-edit; it is marked `linguist-generated` and uses a
  binary merge strategy to avoid 3-way merge conflicts.
- `.pixi/` is the local environment install directory and is not source.
