# `recording/`

Compact, SciPy-native session recording for offline analysis — an archival alternative to the
Rerun `.rrd`. Stores only the **minimal raw** signals; derived quantities are recomputed on read.

- **Philosophy:** persist the face-blurred video + the pose model's raw landmark outputs, and
  recompute anything derivable (joint angles, fps, pose-present, 2D pixel points) from those.
  (This derive-on-read / minimal-raw rule is a repo-wide convention — see root `CLAUDE.md`.)
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

## Tests

Integration tests drive the real `HDF5Recorder` → `AnnotationStore` → `Recording` path against
temp files (fast, self-contained). The recorder's `/feather` datasets + `Recording.feather` — including
that derived streams are *not* stored and that `motion(tau_s=…)` re-derives — are covered in
`tests/test_feather.py` (see `adafruit_feather_sense/CLAUDE.md`).
