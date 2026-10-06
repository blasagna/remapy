# `rerun_viewer/`

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

## Tests (`tests/test_viewer.py`)

The logger's calls (with `rr` mocked) plus **`ReplayRecordingTests`**, which drives the real
`HDF5Recorder` → `AnnotationStore` → `Recording` → `replay_recording` path against temp files: one
`video/image` per frame, `rr.Clear` on untracked frames, feather streams logged once (the
double-derive trap), annotations logged, and the blueprint reflecting what the file actually carries.
Blueprints have no useful `repr`, so assertions walk `root_container` for view origins (`_origins`)
rather than matching on strings. The viewer's `log_sensors` + its derived plots live in
`tests/test_feather.py`.
