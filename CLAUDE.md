# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**How this doc is organized:** each package's deep-dive lives in a nested `CLAUDE.md` inside that
package's directory, loaded on demand only when you work with files there. This root file is the
always-loaded index: project overview, the package map, repo-wide conventions, environment/commands,
and the shared test harness. When you touch a package, read its own `CLAUDE.md` for the detail and
the invariants that constrain that code.

## Project status

Early stage. "Playing with tools inspired by Remy's therapies" (per README). The dependency
set points at computer-vision / motion work: `mediapipe`, `opencv`, `rerun-sdk` (visualization),
plus `numpy`/`scipy`/`pandas` and `h5py` for data.

## Packages

Each entry links to the package's own `CLAUDE.md` (loaded on demand when you work in that directory).

- `video_capture/` — OpenCV capture wrapper (`VideoCapture`) + demo CLI → `video_capture/CLAUDE.md`
- `pose_estimation/` — MediaPipe pose skeleton + joint angles → `pose_estimation/CLAUDE.md`
- `face_blur/` — MediaPipe face redaction (detector / pose / hybrid backends) → `face_blur/CLAUDE.md`
- `rerun_viewer/` — streams the pipeline to the Rerun viewer + offline replay → `rerun_viewer/CLAUDE.md`
- `recording/` — compact HDF5 session recording for offline analysis → `recording/CLAUDE.md`
- `annotate/` — scrub a recording and label time segments → `annotate/CLAUDE.md`
- `motor_metrics/` — continuous motor metrics for GMFM-88 trials (offline + live) → `motor_metrics/CLAUDE.md`
- `list_devices/` — enumerate compatible capture devices → `list_devices/CLAUDE.md`
- `adafruit_feather_sense/` — CircuitPython IMU streamer (serial/BLE) + host readers → `adafruit_feather_sense/CLAUDE.md`

## Global conventions

Repo-wide rules that apply across packages (package-specific rules live in each nested `CLAUDE.md`):

- **Derive-on-read, never written back.** Persist only the **minimal raw** signals; recompute
  anything derivable (joint angles, fps, pose-present, 2D points, `gravity`/`linear_accel`) on read.
  Freezing derived numbers into the `.h5` would strand them at whatever the derivation constants were
  that week. `Recording.angles()` and `recording/recorder.py` are the pattern; it binds hardest in
  `motor_metrics` (every number is a function of `derive.py` constants).
- **Face redaction is applied to the image sink only** (window / Rerun log / recording), always
  *after* detection runs on the raw frame, so pose accuracy is unaffected and only redacted frames
  are ever shown or persisted. All four capture CLIs (`video_capture`, `pose_estimation`,
  `rerun_viewer`, `recording`) expose `--blur-faces`/`--no-blur-faces` (**default on**),
  `--blur-style {box,mosaic}` (default `box`), `--blur-method {detector,pose,hybrid}` (default
  `hybrid`), and `--face-model`. Headless `--no-window` runs with no image sink skip blur;
  `recording`/`rerun_viewer` always redact before persisting.
- **Empty `__init__.py` — import from submodules, not the package root** (e.g.
  `from video_capture.capture import VideoCapture`). The `__init__.py` files are intentionally empty.
- **All capture CLIs request 1280×720 by default** (`--width`/`--height` override; the device picks
  the nearest supported mode).
- **MediaPipe is the Tasks-API-only build** (0.10.35, Python 3.14): `mp.solutions.*` and
  `mediapipe.framework` are absent — use the `mediapipe.tasks.python.vision` classes; skeletons are
  drawn manually with OpenCV. Detail in `pose_estimation/` and `face_blur/`.

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
- `pixi run live` / `pixi run live-crawl` — live metrics over a rolling window in the pose window
  (`--live-metrics {hold,crawl}` on `pose` and `rerun`; `--live-window-s` to change the window).
- `pixi run notebook` — Jupyter Lab in `notebooks/` (offline metric exploration).
- `pixi run test` — run the unit-test suite (verbose). `pixi run test-quiet` for the terse summary.

When adding a build/lint/test workflow, wire it up as a Pixi task so it's captured in the repo
rather than run ad hoc.

## Workflows (skills)

Multi-step procedures are captured as skills under `.claude/skills/` (loaded on demand):

- `feather-deploy` — flash/deploy the Feather Sense board (serial or BLE entry).
- `collect-session` — the data-collection runbook: record → pose-QC in `annotate` → mark trials → metrics.
- `metrics-report` — produce per-trial metrics and the cross-session trend.

## Tests

`tests/` holds `unittest` coverage for the reusable libraries (not the CLIs). Run with `pixi run
test`. Every external boundary is mocked so the suite needs no camera, network, model download,
display, or GPU; ~480 tests run in a few seconds. **Per-file test notes live in each package's nested
`CLAUDE.md`** (e.g. `tests/test_motor_metrics.py` and `tests/test_live.py` under `motor_metrics/`,
`tests/test_feather.py` under `adafruit_feather_sense/`, `tests/test_viewer.py` under `rerun_viewer/`,
`tests/test_annotate.py` under `annotate/`). Shared harness conventions:

- `tests/fakes.py` — shared duck-typed stand-ins: MediaPipe landmark/pose/detection results, an
  opened `cv2.VideoCapture` (`FakeCapture`), `cv2.VideoWriter` (`FakeVideoWriter`), and a
  pyserial handle (`FakeSerial`, fed pre-baked protocol bytes). Plus `fake_recording()` (a
  duck-typed `Recording` — the `motor_metrics` functions read only attributes, so their tests
  need no HDF5) and `body_world()` (synthetic *anatomy*: `make_landmarks` spreads points along a
  diagonal, which is fine for pass-through tests but is not a body, and postural metrics need one).
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
