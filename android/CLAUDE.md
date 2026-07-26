# `android/`

The Android port of the live path: camera + pose + live metrics, for an observer watching Remy
(he never looks at the UI). A **separate Gradle build**, not part of the pixi workspace — pixi is
pinned to `linux-64` and Python 3.14, and nothing here is a Python dependency. The two builds share
exactly one artifact: `metrics/src/test/resources/goldens.json`.

## Status

- `metrics/` — **done and verified.** The whole live kernel in Kotlin, 83 tests green against
  Python goldens at 1e-9, on the **15 Hz** chain (`FS = 15.0`, `WINDOW_S = 0.35`, 5-sample window,
  `LIVE_LAG = 2` — derived from the window, not written down).
- `app/` — **works on a real device.** CameraX + MediaPipe Tasks LIVE_STREAM + Compose overlay +
  face redaction. Verified on a **Pixel 10** (Android 17, arm64-v8a): launches, loads both models
  from assets, holds a sustained 15 fps, renders the overlay clear of the punch-hole camera, tracks
  a real person, and both toggles (lens, framing) and both exercise modes behave — crawl included,
  against an actual crawl. The one thing a working app does **not** establish is *landmark parity*
  with the desktop build; that is still open and still needs measuring, see *Open risk: landmark
  parity*.

## Why a Kotlin rewrite and not Chaquopy

Python does not run on Android without embedding a whole interpreter. The live kernel's entire
third-party surface is numpy plus **two** scipy functions, so reimplementing it (~1,400 LOC) costs
less than shipping ~50 MB of numpy+scipy, marshalling landmark arrays across JNI every frame, and
pinning the app to whatever Python version Chaquopy ships (not 3.14). The trade is that agreement
with the reference implementation has to be *demonstrated* rather than assumed — hence the goldens.

## `metrics/` — the kernel

Pure Kotlin/JVM, **no Android dependency**, so it compiles and its whole suite runs on a desktop
JDK with no SDK, emulator or device. That is what makes the equivalence harness cheap enough to run
on every change; an `androidTest`-only kernel would need a device to tell you the maths is right.

Files mirror `motor_metrics/` one-to-one so the two stay diffable — read the Python module's
docstring for the reasoning behind any of them, and `motor_metrics/CLAUDE.md` for the invariants.

| Kotlin | Python |
|---|---|
| `Derive.kt` | `motor_metrics/derive.py` |
| `Signals.kt` | `motor_metrics/signals.py` |
| `Quality.kt` | `motor_metrics/quality.py` |
| `Hold.kt` | `motor_metrics/hold.py` |
| `Crawl.kt` | `motor_metrics/crawl.py` (+ `transition.symmetry_index`) |
| `LiveWindow.kt` / `LiveMetrics.kt` / `LiveMetricsComputer.kt` | `motor_metrics/live.py` |
| `Angles.kt` | `pose_estimation/angles.py` |
| `PeakFind.kt` | the `scipy.signal.find_peaks` this needs |
| `Matrix.kt`, `PoseFrames.kt`, `Landmarks.kt` | numpy / the `rec` duck type / the `PoseLandmark` enum |

Things worth knowing before editing:

- **`Derive.savgolCoefficients` is the riskiest code in the port.** scipy's default
  `mode="interp"` does not pad: it fits one polynomial to the first/last `windowLength` samples and
  evaluates it at each edge position. `LIVE_LAG` — the whole reason a live readout can be presented
  as the *same measurement* the offline table reports — is defined by that behaviour. A port that
  mirrored or zero-padded instead would agree on every interior sample and disagree on exactly the
  three the live instantaneous readout is taken from. Interior and edges go through one function
  here (scipy uses two code paths), which is provably identical and harder to get half-right.
- **Landmarks are stored as `Float` and computed in `Double`.** Not an optimisation — it is how the
  two implementations stay comparable. The `.h5` and `LiveWindow` hold float32 and the Python
  metrics widen to float64 to compute; holding doubles end to end here would be *more* precise and
  therefore *wrong*, drifting from the reference in the last digits of every number.
- **`PoseFrames` is the duck type made explicit.** `LiveWindow` implements it, so
  `Hold.holdMetrics` / `Crawl.crawlMetrics` run over a live ring buffer **unmodified**. Keep it
  that way: a separate live implementation of the maths is how live and offline numbers start
  disagreeing.
- **The kernel imports no MediaPipe class.** `PoseFrame` is the conversion boundary
  (`recording/recorder.py`'s `landmark_rows` in Kotlin), and building one from a
  `PoseLandmarkerResult` is the app module's job. The full-NaN row for "no pose" is load-bearing —
  `posePresent` and all of `Quality` key off it.
- **Every `LiveMetrics` field keeps the `live` prefix**, and `toMap()` emits the Python
  `live_`-prefixed names. That is the structural half of the never-mix rule; `ConstantsTest` pins
  the exact key set against Python's own `live_field_names()`.

### Deliberately not ported

- **`phase_offset` / reciprocity, both girdles.** Needs `scipy.signal.hilbert`. The live path
  excludes it anyway (Hilbert's edge effects peak at exactly a short trailing window's edges), so
  nothing live is lost — but the offline "bunny haul vs mature crawl" axis is not available on the
  phone. The fields stay in `CrawlMetrics` as NaN so adding them later is a fill-in.
- **SPARC / submovements / `transition.py`**, `report.py` tables, `estimate_up` and calib segments,
  annotations, HDF5. Offline concepts; the desktop stays canonical for analysis.

## `app/` — camera, detector, UI

`MainActivity` -> `PosePipeline` (CameraX `ImageAnalysis` + MediaPipe) -> `CameraScreen` (Compose).
`LandmarkRows` is the *only* file that knows both MediaPipe's types and the kernel's.

- **`RunningMode.LIVE_STREAM`, not `VIDEO`.** The desktop CLIs use `detect_for_video`, which blocks,
  because their loop is a synchronous pull. Here the async callback is the right shape, and
  `STRATEGY_KEEP_ONLY_LATEST` drops frames rather than queueing them so the UI stays current.
  Dropped frames are fine *by construction*: `resampleUniform` puts everything on a uniform grid and
  `windowSpan` selects by time, not frame count. Both were written for a jittery webcam.
- **No `PreviewView`, and this is not negotiable.** A CameraX `Preview` use case hands the camera
  stream straight to a `SurfaceView` — the raw, unredacted feed would reach the screen without
  passing through `FaceRedaction` at all. Rendering analysed frames ourselves costs smoothness
  (display rate = analysis rate) and buys a redaction decision that is always taken deliberately,
  in one place. `blankWhenUnlocated` closes the last gap by blanking a frame outright when no face
  was located. The raw-video toggle below does not weaken this — arguably it depends on it, since
  a `PreviewView` would make redaction impossible rather than optional.
- **`PosePipeline.videoOnly` is the one state that shows an unredacted face**, and it is the
  Android equivalent of the desktop CLIs' long-standing `--no-blur-faces`. It bypasses pose, face
  detection and `FaceRedaction` together in an early branch of `analyze`, because the normal path
  is driven entirely by the pose result callback — skip `detectAsync` and no frame reaches the
  screen at all. Three things hold it honest and none are optional: `CameraScreen` draws a standing
  red notice for as long as it is on (a word on a button is not proportionate to revealing a
  child's face); it is **never persisted**, so every launch starts redacted; and nothing is
  recorded here either, so a raw frame is on screen and nowhere else. The bypass emits
  `LiveMetrics.blank` rather than pushing no-pose rows into the window — there is nothing to push —
  and it **recycles the rotated bitmap itself**, which the tracked path gets free from
  `MPImage.close()`. That last one is the whole memory story on this path: ~3.7 MB at 15 Hz is the
  rate `BitmapRing` records the GC losing to, so a missing `recycle()` reads fine for thirty
  seconds and OOM-kills mid-session.
- **The face detector is gated on `FaceRedaction.hasPoseFaceBox`, never on "is a pose present".**
  This was the intermittent-black-frames bug, and the distinction is the whole of it. Under
  `HYBRID` the pose head box is unavailable in *two* cases — no pose at all, and a tracked body
  whose eleven face landmarks are all below the visibility gate (head down, turned away, far off:
  a belly crawl). Gating on `!hasPose` covered only the first, so in the second the detector never
  ran, `redact`'s fallback got an empty detection list, nothing was redacted and
  `blankWhenUnlocated` blanked the frame. The predicate shares `faceBounds` with `poseHeadBox`
  precisely so the two cannot disagree about what "located" means. Note the Python side has the
  same `if pose else detector` branch (`face_blur/hybrid.py`) and is *not* wrong in the same way:
  it has no `blankWhenUnlocated`, so the case degrades to an unredacted face there — worse, and
  the reason Android added the blanking. The cost of the fix is that the detector now runs on more
  frames, synchronously on the callback thread, which pushes on the open fps question.
  `PosePipeline.noteBlanked` logs one line per blank *burst* with `pose=` and `detections=`
  beside a session total: it is kept deliberately, because the screen cannot tell a correct blank
  (nobody in frame) from this bug returning, and both just look like the video flicking to black.
- **`INTERNET` and `ACCESS_NETWORK_STATE` are explicitly removed** in the manifest. Both get merged
  in by dependencies; neither is needed (models ship in `assets/`). An app that points a camera at a
  child should not be *able* to talk to a network.
- **Bitmap lifecycle is the sharp edge here**, and both mistakes have already been made and fixed:
  `MPImage.close()` recycles the bitmap it was built from, so the display bitmap must be a separate
  copy (sharing them crashes the next compose pass); and that copy must come from `BitmapRing`, not
  a fresh allocation. A 1280x720 ARGB bitmap is ~3.7 MB on the *native* heap, and allocating one per
  frame outruns the GC — measured at 217 -> 358 MB in under two minutes, still climbing. With the
  ring it sawtooths around 210-290 MB over a 7-minute run. Sessions are minutes long; "it settles
  eventually" would be an OOM kill partway through a trial.
- GPU delegate with **CPU fallback**, which is exercised: the emulator has no OpenCL and falls back
  cleanly. The delegate in use is on screen next to the frame rate, because it is a plausible cause
  of a device that cannot hold the target rate.
- **The rear/front toggle must never mirror the front camera.** A selfie preview flips horizontally
  to look natural, and that flip would swap the child's left and right *as MediaPipe sees them* —
  which silently inverts the sign of `live_leg_amplitude_symmetry`, the "which leg does he favour"
  reading this mode exists to produce. CameraX's analysis buffer is un-mirrored as delivered, so the
  correct implementation is to do nothing; the trap is "fixing" the front view because it reads
  oddly when pointed at yourself. It is correct when pointed at Remy, which is the only use.
  Flipping calls `PosePipeline.reset()`, discarding the rolling window: the two lenses differ in
  framing, field of view and sensor, and a window spanning the switch would average sway across a
  discontinuity that has nothing to do with him — the same refusal `longest_run` makes about
  bridging a dropout. The toggle hides itself on a device with only one lens.
- **Never destroy a MediaPipe task while frames are in flight.** The first camera-flip
  implementation rebuilt the whole pipeline, and closing a `PoseLandmarker` on the main thread
  while the analyzer thread sat inside `detectAsync` segfaulted in native code
  (`PacketCreator.nativeCreateProto`) after a handful of flips. A `@Volatile closed` flag narrows
  that window but cannot close it — the check and the call are not atomic. The tasks are therefore
  built **once for the life of the activity**, and a flip only rebinds CameraX and swaps the
  `LiveMetricsComputer`. Anything that feels like it wants to tear down and rebuild the detector
  should be re-examined against this.
- **The analyzer decimates to `Derive.FS`; the sensor is left free-running.** A phase accumulator
  in `PosePipeline.analyze` (`nextDueMs = max(now, nextDueMs) + period`) drops frames *before*
  `toBitmap()`, so a rejected frame costs one `close()`. A naive `sinceLast >= 67` test would beat
  against a jittery 30 fps source and sag to ~14 fps with a 100 ms hole in it. It sits **before**
  the timestamp block on purpose: `timestampMs` is assigned only for accepted frames, from the true
  wall clock, so the kernel sees a genuine 15 Hz jittery timebase and needs **no compensating
  change at all** — `resampleUniform` and `windowSpan` were written for exactly this. Capping the
  sensor instead (`SessionConfig.setFrameRateRange`) was rejected: it doubles the AE exposure
  ceiling to 67 ms in the poor light these sessions actually happen in, and a smeared crawling
  child is a worse input than a slightly warmer phone.
- **Portrait/landscape is an explicit toggle, and rotation must not cost an allocation.**
  `setOutputImageRotationEnabled(true)` makes CameraX rotate inside its own reused `ImageReader`,
  so the delivered frame is upright and `rotationDegrees` stays 0 — which keeps
  `PosePipeline.rotated()` on its zero-copy path. Without it, portrait would do a full
  `createBitmap` per frame: a second ~3.7 MB allocation that `BitmapRing` does *not* serve, on a
  device already rate-limited. `PosePipeline` logs the frame geometry on change for exactly this
  reason — **it must read `rot=0` in both orientations**, and a non-zero value means the fallback
  path is live. `ImageProcessingOptions.setRotationDegrees` was rejected again for the reason the
  existing docstring gives: it would leave the display bitmap in the sensor frame while landmarks
  came back rotated, putting a hand-written inverse transform between the pose and `FaceRedaction`.
  The toggle drives `requestedOrientation` (the *window* must go tall for Compose), and
  `onConfigurationChanged` rebinds — the manifest's `configChanges` is what stops the activity
  being recreated and the MediaPipe tasks torn down. Rotation calls `reset()` for a stronger reason
  than a lens flip: it changes the camera's relation to gravity, so `WORLD_UP` and every
  `live_trunk_angle_*` shift discontinuously, and a window spanning that would average across a
  change that has nothing to do with the child. The resolution request is transposed in portrait,
  since `ResolutionStrategy`'s bound size is read in the target-rotation frame.
- **The overlay is inset by `WindowInsets.safeDrawing`; the video is not.** At `targetSdk = 37`
  edge-to-edge is enforced, and a punch-hole selfie camera sits top-centre in portrait. The chrome
  lives in its own layer with one `windowInsetsPadding`, while the letterboxed video stays
  full-bleed underneath: a cutout over part of the image costs nothing, a cutout over the
  `coverage` row costs the reader the number that says whether to trust the rest. `safeDrawing` is
  used rather than a hand-assembled `displayCutout.union(statusBars)` because it is already that
  union and cannot be re-derived wrongly later. Known cost: in landscape the cutout inset spans the
  whole edge, so ~30 dp of overlay width goes unused.
- **Five overlay controls, top-right**: `menu` (back to the landing screen), pose (`pose`/`raw
  video`), exercise mode (`hold`/`crawl`/`off`), lens (`rear`/`front`) and framing
  (`landscape`/`portrait`).
  All show their current state as a word rather than a glyph, because the operator has to
  know which is live without inferring it from the image. `menu` is duplicated by the back gesture
  and exists anyway: the system bars are hidden, so back is undiscoverable on a phone handed to
  someone who has not used this before. Switching mode discards the rolling
  window —
  unavoidably, since the two modes use different window lengths (5 s vs 6 s, a crawl needing
  several pull cycles before its period CV means anything) — so the readout blanks and re-warms.
- **`off` is an app-layer display flag, not a third mode string**, and it must stay that way.
  `LiveMetricsComputer` is constructed from a mode with a window length and rejects anything
  outside `MODE_WINDOW_S`, so `off` cannot be represented there at all. `MainActivity.overlayOff`
  hides `MetricsPanel` while the pipeline keeps running underneath. **Entering `off` is what calls
  `setMode(HOLD)`; leaving it touches nothing at all** — the cycle returns to `hold` and the two
  modes have different window lengths, so a discard has to happen somewhere on the way round, and
  this spends it while the panel is hidden instead of in the one place the operator would sit and
  watch it re-warm. Reversing those two makes the toggle feel broken without changing a line of
  metrics code. The rejected alternative was to stop pushing frames; it buys a ~3 ms recompute
  against a 67 ms frame and costs a nullable `RenderedFrame.metrics` threaded through the pipeline
  and the render path. `off` hides the panel *entirely*, `coverage` and `fps` with it, so the
  session checklist below means "not while `off`".
- `Overlay` ports `live_draw.py` — same row order, same `--` for NaN, same colour semantics
  (coverage green/red against the gate, `up:` orange), same `sitSteadiness` reading the trunk's
  deviation from its *own* window baseline rather than an absolute upright angle.
  **What each row means for the operator is [`OVERLAY.md`](OVERLAY.md)**, kept out of this file on
  purpose: this one is for someone editing the chain, that one for someone reading the screen
  mid-session. A row's units, sign or caveat changing here has to change there too.

Models are fetched into `app/src/main/assets/` by the `fetchModels` Gradle task, which prefers the
copies the Python side already cached — same URLs, same download-on-first-use, gitignored the same
way. Preferring the local cache also guarantees phone and laptop run the *same bytes*.

## Goldens: the equivalence harness

`pixi run export-fixtures` (`tests/fixtures/export.py`) writes `goldens.json` — the Python kernel's
*actual behaviour*, inputs paired with outputs, from `savgol_filter` up to whole
`LiveMetricsComputer.push` sequences compared frame by frame. Marked `linguist-generated`; never
hand-edit, and **regenerate it whenever a `derive.py` constant changes** — a stale file would pin
the Kotlin port to a filter chain the Python side no longer uses.

Both kinds of test earn their place. Goldens catch a differently-handled Savitzky-Golay edge or a
plausible-but-different prominence walk. The **closed forms** — sway RMS against `amplitude/√2`,
cadence against the driving frequency, the ellipse against `5.991·π·σ²`, the `MIN_CYCLE_EXCURSION_M`
jitter regression, the documented derivative-gain table — pin the *intent*, so when the two
disagree there is something to arbitrate with.

## Commands

Run from `android/`. The wrapper (`gradlew`, `gradle/wrapper/`) is tracked; build outputs are not.

- `./gradlew :metrics:test` — the kernel suite (83 tests, ~1 s, no device, no SDK).
- `./gradlew :app:assembleDebug` — the APK (~43 MB; arm64-v8a + x86_64 only, since MediaPipe's
  native library is 10-15 MB per ABI and all four made a 65 MB debug build).
- `./gradlew build` — everything.

The SDK path lives in `local.properties` (gitignored, per-machine). `:app` pins a Java 17 toolchain
and lets Gradle provision it — AGP needs a real `javac`, and a machine can easily have a JRE-only
Java that runs Gradle and Kotlin fine while providing no compiler at all. `:metrics` deliberately
pins no toolchain so it builds anywhere.

### Installing it on a phone

The APK is **debug-signed**, which is fine for sideloading over `adb` and is not distributable.
Requires Android 8.0 (API 26) or newer, and an **arm64-v8a** device — every phone made in the last
decade, but a very old 32-bit one will refuse to install (add `armeabi-v7a` to `abiFilters` in
`app/build.gradle.kts` if you ever hit that).

1. **Enable USB debugging on the phone.** Settings → About phone → tap *Build number* seven times,
   then Settings → System → Developer options → *USB debugging*.
2. **Plug it in** and accept the *Allow USB debugging?* prompt (tick "always allow" for this
   computer).
3. **Check it is visible.** `adb devices` should list it as `device`. `unauthorized` means step 2
   was not accepted. `no permissions` (Linux only) means the udev rules are missing:
   `sudo apt install android-sdk-platform-tools-common`, then replug. That package is *only* a
   rules file — no binaries, so it cannot shadow the SDK's `adb` — and its rule carries
   `TAG+="uaccess"`, so on a systemd desktop there is no need to join `plugdev` despite what most
   guides say. Often none of this is needed at all; check before fixing.
4. **Build and install in one step:**
   ```
   ./gradlew :app:installDebug
   ```
   (`assembleDebug` + `adb install -r app/build/outputs/apk/debug/app-debug.apk` is the same thing
   in two.)
5. **Launch "remapy"** from the app drawer and grant the camera permission when asked.

**Over Wi-Fi instead** (Android 11+), which is what you want for a phone already on a tripod:
Developer options → *Wireless debugging* → *Pair device with pairing code*, then
`adb pair <host>:<port>` with that code, and `adb connect <host>:<port>` using the (different) port
shown on the main wireless-debugging screen. `installDebug` then works as above with no cable.

The first build needs network for `fetchModels`, unless the Python model caches are already
populated — in which case it copies from them, which is also what keeps phone and laptop on
identical model bytes.

**What to check at the start of every session.** This was the first-run checklist; the first run has
happened and passed on a Pixel 10, but none of these are one-time facts — `fps` moves with thermals,
`coverage` and the redaction move with framing and lighting, and the vertical moves with wherever
the phone was propped. The overlay is where each of them surfaces:

- **`fps`** — green at ≥ 13.5, which is `0.9 * Derive.FS` rather than a written-down number. The
  analyzer decimates *to* 15, so a healthy reading sits at 15 and never above it; below the
  threshold the chain is resampling *up* and inventing correlated samples. The `gpu`/`cpu` label
  beside it is the first thing to suspect if it is low.
- **`coverage`** — green means the torso landmarks are trusted. Persistent red means framing or
  lighting, not a bug; the metrics are supposed to blank rather than guess.
- **`up world_y`** — a reminder that the vertical assumes a level camera. It is why the hold readout
  leads with `trunk dev` (deviation from the window's own baseline) rather than an absolute lean.
- **The face redaction.** Confirm a face is actually covered before pointing this at anyone. If the
  whole frame goes black, no face was located and `blankWhenUnlocated` did its job. Worth re-checking
  after a lens or framing change specifically, since both alter what the detector is looking at.
  And confirm the pose control reads `pose`, not `raw video` — the latter is redaction off, and it
  says so in red where the metrics panel usually is.

Nothing is recorded — no storage permission, no files written, no network. The app opens on a menu
and the camera does not start until the live view is entered; the screen stays on only while it is.
Framing is portrait by default, with a landscape toggle in the overlay for a tripod.

### Running it without a device

```
sdkmanager "emulator" "system-images;android-36;google_apis;x86_64"
avdmanager create avd -n remapy-test -k "system-images;android-36;google_apis;x86_64"
emulator -avd remapy-test -no-window -gpu swiftshader_indirect -camera-back virtualscene
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell pm grant dev.remapy.app android.permission.CAMERA
adb shell am start -n dev.remapy.app/.MainActivity
```

Worth knowing what this can and cannot show. The virtual scene contains **no person**, so pose
detection never fires — which makes it a good test of the no-pose paths (blanked readout, full-frame
redaction) and no test at all of pose quality. It does catch lifecycle, model loading, threading,
bitmap ownership and leaks, which is most of what goes wrong in an app.

## Open risk: landmark parity

Nothing here says Android MediaPipe with a GPU delegate emits the same `pose_world_landmarks` as
the desktop CPU Python build on the same frames. **The kernel agreeing is not the pipeline
agreeing.** Any systematic difference is a pipeline change in the sense `motor_metrics/CLAUDE.md`
warns about — the same category as changing a `derive.py` constant — and it would break the
cross-session trend the package exists for.

One variable is already removed: `tasks-vision` is pinned to **0.10.35**, the same MediaPipe release
`pixi.toml` pins, running the same model bytes. What remains is the delegate and backend difference
(GPU/NNAPI on a phone vs CPU in the Python build), and that is not a small thing to wave through.

Measure it before trusting any Android number: record on the laptop, `pixi run export-video` it, run
that mp4 through the app (needs a file-source screen — **not yet built**), and compare against the
`.h5`'s `/pose/landmarks_world` — then compare the resulting `hold_metrics` / `crawl_metrics`, which
is the figure that actually decides whether phone and laptop sessions can share a trend. Until then,
treat a capture-device change as a new baseline.

## Closed risks, and what closed them

Kept rather than deleted: each of these shaped a design decision that still stands, and the
reasoning is only legible next to the risk it answers.

- **Sustained frame rate — was unmeasured, now measured and held.** The Pixel 10 delivered 15-20
  fps, not the 30 `derive.FS` assumed, so the grid moved to the hardware rather than the other way
  round: `derive.FS` is **15.0** in both languages and `PosePipeline` decimates to exactly that
  instead of letting the rate wander. Capping the *sensor* was rejected — a fixed 15 fps AE range
  doubles the exposure ceiling to 67 ms, and these sessions happen across a room in poor light,
  exactly when AE takes the long exposure and smears a crawling child. A sustained 15 was then
  confirmed on the device. The rate and delegate stay on screen: this closed on *one* phone, and a
  slower or thermally-throttled one is still a different question.
- **Pose quality — observed.** A real person in frame tracks sensibly. This had been open since the
  port was written.
- **Crawl mode — exercised against a real crawl.** The mode the `leg_*` fields exist for has now
  seen the thing it measures.

## Other open risks

- **Landmark parity is still the one that matters, and "it works" does not touch it.** A tracking
  skeleton that looks right on screen says nothing about whether Android's GPU delegate emits the
  *same* `pose_world_landmarks` as the desktop CPU build on the same frames. That is a numeric
  comparison, not a visual one — see the section above for how to run it. Until it is run, phone
  and laptop sessions are separate baselines and must not be pooled into one trend.
- **Every metric moved scale when the grid did.** Recoverable, since metrics are derived on read:
  re-running `pixi run metrics` regenerates past sessions on the 15 Hz chain. Figures already
  written down elsewhere are not.
