# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

Early stage. "Playing with tools inspired by Remy's therapies" (per README). The dependency
set points at computer-vision / motion work: `mediapipe`, `opencv`, `rerun` (visualization),
plus `numpy`/`scipy`/`pandas` and `h5py` for data.

Current code: `video_capture/`, a small package wrapping OpenCV video capture.

## Packages

### `video_capture/`

- `capture.py` — `VideoCapture`, a context-manager wrapper around `cv2.VideoCapture`. Defaults
  to camera index `0` (built-in webcam); also accepts a file path/URL. Provides `read()` (single
  BGR frame), `frames()` (iterator), a `resolution` property, and raises `CaptureError` on
  open/read failure. Both are re-exported from the package root.
- `main.py` — example CLI (`python -m video_capture.main`) that displays the webcam feed in a
  window (quit with `q`/`Esc`). Flags: `--source`, `--width`, `--height`, `--no-window`
  (headless), `--max-frames` (bounded runs). A bare-integer `--source` is treated as a camera
  index, anything else as a path/URL.

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

When adding a build/lint/test workflow, wire it up as a Pixi task so it's captured in the repo
rather than run ad hoc.

## Notes

- `pixi.lock` is generated — do not hand-edit; it is marked `linguist-generated` and uses a
  binary merge strategy to avoid 3-way merge conflicts.
- `.pixi/` is the local environment install directory and is not source.
