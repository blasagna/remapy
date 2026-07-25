# `android/`

The Android port of the live path: camera + pose + live metrics, for an observer watching Remy
(he never looks at the UI). A **separate Gradle build**, not part of the pixi workspace — pixi is
pinned to `linux-64` and Python 3.14, and nothing here is a Python dependency. The two builds share
exactly one artifact: `metrics/src/test/resources/goldens.json`.

## Status

- `metrics/` — **done and verified.** The whole live kernel in Kotlin, 83 tests green against
  Python goldens at 1e-9.
- `app/` — **builds and runs.** CameraX + MediaPipe Tasks LIVE_STREAM + Compose overlay + face
  redaction. Verified on an emulator: launches, loads both models from assets, delivers ~25 fps,
  renders the overlay, and blanks correctly when no pose is present. **Never yet run against a
  real child or a real camera**, so nothing about pose quality or on-device frame rate is
  established — see *Open risks*.

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
  (display rate = analysis rate) and buys the repo's invariant: only redacted frames are ever shown.
  `blankWhenUnlocated` closes the last gap by blanking a frame outright when no face was located.
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
  of a device that cannot hold 30 fps.
- `Overlay` ports `live_draw.py` — same row order, same `--` for NaN, same colour semantics
  (coverage green/red against the gate, `up:` orange), same `sitSteadiness` reading the trunk's
  deviation from its *own* window baseline rather than an absolute upright angle.

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

**What to check on the first real run**, since these are the open risks and the overlay is where
they surface:

- **`fps`** — green at ≥ 25. Below that the 30 Hz filter chain is resampling *up* and inventing
  correlated samples. The `gpu`/`cpu` label beside it is the first thing to suspect if it is low.
- **`coverage`** — green means the torso landmarks are trusted. Persistent red means framing or
  lighting, not a bug; the metrics are supposed to blank rather than guess.
- **`up world_y`** — a reminder that the vertical assumes a level camera. It is why the hold readout
  leads with `trunk dev` (deviation from the window's own baseline) rather than an absolute lean.
- **The face redaction.** Confirm a face is actually covered before pointing this at anyone. If the
  whole frame goes black, no face was located and `blankWhenUnlocated` did its job.

Nothing is recorded — no storage permission, no files written, no network. The screen stays on and
the app is landscape-locked.

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

## Other open risks

- **Sustained frame rate on a real device is unmeasured.** `derive.FS` is 30.0. The chain absorbs
  jitter and dropped frames by design, but a device sustaining well under ~25 fps resamples *up* to
  the grid and starts manufacturing correlated samples, at which point the documented
  derivative-gain table no longer describes the filter. The rate and delegate are on screen for
  exactly this; the emulator's 25 fps under swiftshader says nothing about a phone.
- **Pose quality has never been observed.** No run has yet put a person in front of the camera.
- **No mode switch in the UI.** `LiveMetricsComputer.HOLD` is hard-coded in `MainActivity`; crawl
  mode is reachable only by editing that line.
