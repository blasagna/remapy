# `video_capture/`

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
