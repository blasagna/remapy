# `pose_estimation/`

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
- `draw.py` — `draw_skeleton(frame, landmarks_norm, connections, visibility=None,
  min_visibility=0.5)`, the one OpenCV skeleton drawer, shared by the live CLI and `annotate`.
  Takes a plain `(K, 2+)` **numpy** array (not MediaPipe objects) and **imports no mediapipe**,
  which is the point: a recording already stores its edge list in `meta/pose_connections`, so
  `annotate` can draw poses without pulling the whole MediaPipe stack in for a 35-pair constant.
  Also owns `put_text` (black-outlined text, readable over arbitrary footage), shared by `annotate`
  and `motor_metrics.live_draw` — both draw over whatever the camera happened to see.
  Adds the two behaviours the live path never needed — it **skips NaN** points/bones (the
  recorder writes full-NaN rows for untracked frames and `annotate` scrubs across them;
  `int(nan)` raises), and **dims** landmarks below `min_visibility`, since MediaPipe
  *extrapolates* occluded points rather than dropping them and an invented coordinate otherwise
  looks identical to a measured one. The threshold matches `motor_metrics.quality.Gate`, so what
  looks solid is what the metrics will accept. `main.py:draw_pose` is a thin adapter over it.
- `main.py` — CLI (`python -m pose_estimation.main`); same flags as `video_capture.main`, plus
  `--model`. Windowed mode overlays skeleton + angles; `--no-window` prints angles.
