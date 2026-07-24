# `face_blur/`

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

## Tests

Pure logic (`redact`, `pose_blur`, `factory`) runs unmocked against real NumPy/OpenCV — see the
root `CLAUDE.md` Tests section for the shared harness conventions.
