# `list_devices/`

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
