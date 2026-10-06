# `annotate/`

Post-hoc labeling tool: scrub an already-recorded `.h5` in an OpenCV window and attach text
labels to time segments (stored via `recording.annotations.AnnotationStore`).

- `main.py` — CLI (`python -m annotate.main session.h5` / `pixi run annotate session.h5`).
  Displays `Recording.frame(i)` (already face-blurred) with a bottom timeline strip showing the
  playhead and existing segments as colored, lane-stacked spans. Keys: `,`/`.` step a frame,
  `<`/`>` jump ~1s, `space` play/pause, `i`/`o` mark in/out (then type a label at the terminal
  prompt), `x` delete the nearest segment, `p` toggle the pose overlay, `?` help, `q`/`Esc` quit.
  Edits save immediately.
- **Labels are named on screen, not just colored.** `_active_labels()` names the segment(s) under
  the playhead in that label's own strip color, and a swatch→label legend sits above the strip.
  Color alone was never enough: the `motor_metrics.labels` vocabulary makes trials differ only by
  their params (`arms=free` vs `arms=prop`), so spans that matter are near-identical. Text is
  drawn with a black outline (`_put_text`) because it lands over arbitrary footage. Overlapping
  segments are normal, so the readout is a list.
- **Pose overlay (`p`, default on)** — draws the stored landmarks via
  `pose_estimation.draw.draw_skeleton`, using `rec.pose_connections` from the file (no mediapipe
  import). It is **drawn before the strip rectangle** — the strip is painted *over* the bottom
  `_STRIP_H` px of the same image, so drawing after would run limbs across the timeline (pinned
  by an equality check on the strip region with the overlay on vs off). Dimmed limbs mark
  low-visibility/extrapolated landmarks, which is the QC signal the data-collection runbook asks
  the operator to eyeball. `main()` caches `rec.pose_present` **once** — it is a property that
  re-slices the whole `(N,33,3)` world array on every access.
- **HDF5 locking note:** the tool holds two handles on the same file — `AnnotationStore` (`"r+"`)
  and `Recording` (`"r"`). h5py requires the **`"r+"` handle be opened first**; opening `"r"`
  before `"r+"` on one path in one process raises `OSError`. `main()` opens the store before the
  reader for exactly this reason (pinned by `test_rw_then_ro_handle_coexist`).

## Tests (`tests/test_annotate.py`)

The `annotate` GUI's drawable logic (the loop itself, needing a window and keyboard, is not tested).
`draw_skeleton` against real numpy/cv2: all-NaN rows no-op, partial NaN skips only the affected
bones, `(0.5, 0.5)` lands on the frame center, and low visibility dims (a bone taking the *weaker*
of its two endpoints). Plus `_active_labels` (inclusive boundaries, overlapping spans, empty gaps)
and `_render` smoke tests over a fake `Recording` for pose on/off, untracked frames, and
with/without annotations — one mis-marked annotation or one untracked frame must not take down the
window.
