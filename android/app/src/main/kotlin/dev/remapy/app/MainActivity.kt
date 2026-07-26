package dev.remapy.app

import android.Manifest
import android.content.pm.ActivityInfo
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import dev.remapy.metrics.LiveMetrics
import dev.remapy.metrics.LiveMetricsComputer
import dev.remapy.metrics.PoseFrame
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * The whole app: a menu, and the live view it leads to.
 *
 * Phase 1 of the Android port: camera + pose + live metrics, for an **observer** watching Remy
 * — he never looks at this screen. Nothing is persisted; the desktop pipeline stays canonical for
 * recordings, annotations and the cross-session trend.
 *
 * **One activity, two screens, and no navigation library.** [screen] is an ordinary activity field
 * like every other piece of UI state here. That is not laziness: the manifest's `configChanges`
 * stops this activity ever being recreated, which is what lets plain fields hold UI state at all,
 * and it exists because a recreation would tear down the MediaPipe tasks mid-frame. A `NavHost`
 * would add a second lifecycle to reason about against exactly one transition.
 *
 * Portrait by default — the orientation a phone is picked up in, and the one that frames a
 * standing or crawling child rather than the floor either side of him. The overlay's toggle
 * switches to landscape at runtime for a tripod.
 */
class MainActivity : ComponentActivity() {

    /** Which screen is showing. Two of them, so an enum rather than a navigation graph. */
    private enum class Screen { MENU, LIVE }

    companion object {
        private const val TAG = "MainActivity"

        /** The same request every desktop capture CLI makes; the device picks the nearest mode. */
        private const val TARGET_WIDTH = 1280
        private const val TARGET_HEIGHT = 720
    }

    private lateinit var analysisExecutor: ExecutorService
    private var pipeline: PosePipeline? = null

    private var bitmap by mutableStateOf<Bitmap?>(null)
    private var frame by mutableStateOf<PoseFrame?>(null)
    private var metrics by mutableStateOf<LiveMetrics?>(null)
    private var fps by mutableStateOf(0.0)
    private var usingGpu by mutableStateOf(false)
    private var hasCamera by mutableStateOf(false)

    /** Which screen is showing. The app opens on the menu; the camera does not start until [enterLive]. */
    private var screen by mutableStateOf(Screen.MENU)

    private var lensFacing by mutableStateOf(CameraSelector.LENS_FACING_BACK)

    /** Which exercise the live readout is measuring. `hold` is the common case, so it is the default. */
    private var liveMode by mutableStateOf(LiveMetricsComputer.HOLD)

    /**
     * Whether the readout is hidden — the third position of the mode toggle.
     *
     * Held here rather than as a third [LiveMetricsComputer] mode string, and that is the whole
     * design. `off` is a *display* state: the pipeline keeps computing [liveMode] underneath while
     * the panel is hidden, so the window is warm by the time it is shown again (see [toggleMode]).
     * It also keeps `off` out of `:metrics` entirely — a mode with no window length is not
     * something [LiveMetricsComputer] can represent, and every constructor there rejects one.
     *
     * The rejected alternative was to stop pushing frames instead. It buys back a ~3 ms recompute
     * against a 67 ms frame, and costs a nullable `RenderedFrame.metrics` threaded through the
     * pipeline and the whole render path — a worse trade than the recompute is worth.
     */
    private var overlayOff by mutableStateOf(false)

    /** Whether the *other* lens exists. Tablets and some phones have only one. */
    private var canFlipCamera by mutableStateOf(false)

    /**
     * Whether the view is framed vertically. Driven by the toggle, never by the sensor.
     *
     * A tripod-mounted phone must not reframe itself because someone picked it up, and every
     * orientation change throws away the rolling window (see [rebindForOrientation]) — so this is
     * a deliberate act, taken between trials, and the button shows which state is live.
     *
     * **Must agree with `android:screenOrientation` in the manifest**, which nothing reconciles at
     * runtime. Disagreeing leaves the window tall while this reads landscape: `startCamera` sends
     * a transposed resolution request, the button shows the wrong word, and the first tap of the
     * toggle is spent resyncing rather than reframing.
     */
    private var portrait by mutableStateOf(true)

    /**
     * Whether pose, face detection and redaction are all off, showing the captured video as-is.
     *
     * **This is the one state in which an unredacted face reaches the screen**, which is why the
     * live view draws a standing warning while it is on and why it is deliberately *not* persisted
     * anywhere: every launch starts redacted, and nothing in this app writes a preference that
     * could change that. It is the same switch the desktop CLIs have always had as
     * `--no-blur-faces`, and it is subject to the same rule — nothing is recorded here either, so
     * a raw frame is on screen and nowhere else.
     *
     * Distinct from [overlayOff], which hides the readout over a redacted, tracked video. This
     * stops the tracking too, so there is nothing left to hide.
     */
    private var videoOnly by mutableStateOf(false)

    private var cameraProvider: ProcessCameraProvider? = null

    private val requestCamera = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        hasCamera = granted
        // Only if the operator is still on the live screen: they can back out to the menu while
        // the system dialog is up, and starting a camera behind a menu is exactly the kind of
        // thing that leaves an indicator lit with nothing on screen to explain it.
        if (granted && screen == Screen.LIVE) startCamera()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Draw behind the system bars and then hide them. The `Fullscreen` platform theme in
        // `themes.xml` no longer does anything on modern API levels, and at `targetSdk = 37`
        // edge-to-edge is enforced regardless — so this is what actually reclaims the status-bar
        // strip for the camera image. It does *not* weaken the cutout handling in `CameraScreen`:
        // `WindowInsets.displayCutout` reports the punch hole whether or not the bars are visible.
        enableEdgeToEdge()
        WindowInsetsControllerCompat(window, window.decorView).apply {
            hide(WindowInsetsCompat.Type.systemBars())
            systemBarsBehavior =
                WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
        }

        analysisExecutor = Executors.newSingleThreadExecutor()

        hasCamera = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED

        setContent {
            // An explicit dark scheme, not the default. `MaterialTheme {}` with no argument
            // resolves to Compose's *light* palette, which lands near-white text on the black
            // `windowBackground` from `themes.xml` — invisible. `CameraScreen` never showed it
            // because it paints its own black, but `PermissionPrompt` has always been dark grey
            // on black, and a menu would be worse.
            MaterialTheme(colorScheme = darkColorScheme()) {
                when {
                    screen == Screen.MENU -> MenuScreen(
                        onEnterLive = ::enterLive,
                        modifier = Modifier.fillMaxSize(),
                    )

                    !hasCamera -> PermissionPrompt(
                        onRequest = { requestCamera.launch(Manifest.permission.CAMERA) },
                        onBack = ::exitLive,
                        modifier = Modifier.fillMaxSize(),
                    )

                    else -> CameraScreen(
                        bitmap = bitmap,
                        frame = frame,
                        metrics = metrics,
                        fps = fps,
                        usingGpu = usingGpu,
                        lensFacing = lensFacing,
                        canFlipCamera = canFlipCamera,
                        onFlipCamera = ::flipCamera,
                        mode = liveMode,
                        overlayOff = overlayOff,
                        onToggleMode = ::toggleMode,
                        portrait = portrait,
                        onToggleOrientation = ::toggleOrientation,
                        videoOnly = videoOnly,
                        onToggleVideoOnly = ::toggleVideoOnly,
                        onExit = ::exitLive,
                        modifier = Modifier.fillMaxSize(),
                    )
                }
            }
        }
    }

    /**
     * Open the live view, starting the camera.
     *
     * The permission request lives here rather than in [onCreate] on purpose: an app that asks for
     * the camera before showing anything gives the operator nothing to decide against. Asked at
     * the moment they tap "live view", the reason is on screen behind the dialog.
     */
    private fun enterLive() {
        screen = Screen.LIVE
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        hasCamera = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED
        if (hasCamera) startCamera() else requestCamera.launch(Manifest.permission.CAMERA)
    }

    /**
     * Return to the menu, stopping the camera.
     *
     * Unbinds CameraX and **keeps the MediaPipe tasks loaded**. Closing them here would be the
     * native use-after-free [PosePipeline.reset] documents — a task destroyed while the analyzer
     * thread is inside `detectAsync`. Unbinding stops frames at the source instead, which is what
     * actually matters: the camera indicator goes out, and re-entering costs no model reload.
     *
     * The rolling window goes with it. Whatever happens between leaving and returning is a gap the
     * window has no business bridging, and [startCamera] rebinds onto a fresh one.
     */
    private fun exitLive() {
        cameraProvider?.unbindAll()
        window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        pipeline?.reset()
        bitmap = null
        frame = null
        metrics = null
        fps = 0.0
        screen = Screen.MENU
    }

    /**
     * Switch to the other lens.
     *
     * Rebinds CameraX and resets the rolling metrics window; the MediaPipe tasks are **kept**. An
     * earlier version rebuilt the whole pipeline here and segfaulted — closing a `PoseLandmarker`
     * on the main thread races the analyzer thread inside `detectAsync`, which is a use-after-free
     * in native code that no flag can reliably guard. See [PosePipeline.reset].
     */
    private fun flipCamera() {
        if (!canFlipCamera) return
        lensFacing = if (lensFacing == CameraSelector.LENS_FACING_BACK) {
            CameraSelector.LENS_FACING_FRONT
        } else {
            CameraSelector.LENS_FACING_BACK
        }
        // Blank the view immediately rather than leaving the last frame of the previous camera on
        // screen while the new one warms up — a stale image with live-looking chrome around it.
        bitmap = null
        frame = null
        metrics = null
        fps = 0.0
        pipeline?.reset()
        startCamera()
    }

    /**
     * Switch between landscape and portrait framing.
     *
     * Driven through [setRequestedOrientation] rather than `ImageAnalysis.setTargetRotation`,
     * because the *window* has to become tall for Compose to lay the overlay out vertically;
     * target rotation alone would rotate the analysis buffer and leave the UI sideways.
     *
     * The activity declares `configChanges="orientation|screenSize"`, so it is **not** recreated
     * here — [onConfigurationChanged] fires instead and rebinds. When the window is already in the
     * requested orientation that callback never comes at all, so rebind directly.
     */
    private fun toggleOrientation() {
        portrait = !portrait
        val alreadyThere =
            (resources.configuration.orientation == Configuration.ORIENTATION_PORTRAIT) == portrait
        requestedOrientation = if (portrait) {
            ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        } else {
            ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE
        }
        if (alreadyThere) rebindForOrientation()
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        // Only while the live view owns the camera. A rebind from the menu would start a camera
        // nothing is displaying — the indicator lit with no explanation on screen.
        if (screen == Screen.LIVE) rebindForOrientation()
    }

    /**
     * Rebind CameraX for the current orientation, discarding the rolling window.
     *
     * The same shape as [flipCamera], and the discard is **less** optional here, not more. Rotating
     * the phone changes the camera's relation to gravity, so `WORLD_UP`'s level-camera assumption
     * and every `live_trunk_angle_*` value shift discontinuously at the moment of the flip. A
     * window spanning that would average a trunk angle across a change that has nothing to do with
     * the child — the same refusal `longest_run` makes about bridging a dropout.
     *
     * Rebinding is also what keeps the analysis buffer upright without any explicit rotation call:
     * a freshly built `ImageAnalysis` picks up the current display rotation as its target, which is
     * exactly what `configChanges` swallowing the recreation would otherwise have cost us.
     *
     * The MediaPipe tasks are **kept**, for the reason [flipCamera] documents.
     */
    private fun rebindForOrientation() {
        bitmap = null
        frame = null
        metrics = null
        fps = 0.0
        pipeline?.reset()
        startCamera()
    }

    /**
     * Cycle the readout: `hold` -> `crawl` -> `off` -> `hold`.
     *
     * Only the metric dispatch changes — the camera keeps running and the models stay loaded. On
     * the two transitions that change the *measurement*, the readout blanks and re-warms over the
     * next few seconds because the rolling window is discarded; see [PosePipeline.setMode] for why
     * that is not avoidable.
     *
     * **Entering `off` switches the pipeline back to `hold` and leaving it touches nothing.**
     * That ordering is the point, and it is what makes `off` free rather than merely quiet. The
     * window has to be discarded somewhere on the way round the cycle — `off` always returns to
     * `hold`, and the two modes have different window lengths — so it is spent while the panel is
     * *hidden*, where a re-warm costs the operator nothing to watch. Five seconds later the window
     * is full, and the tap out of `off` is a pure display flip onto a populated readout. Doing it
     * the other way round would put the same discard in the one place it is visible.
     *
     * Note this hides the whole panel, `coverage` and `fps` included — an `off` that leaves a
     * panel on screen is not off. `android/CLAUDE.md`'s per-session checklist assumes those rows
     * are visible, so it means "not while `off`".
     *
     * Worth knowing which mode you are in beyond the label: `crawl` reads **no vertical at all**
     * (the axis is the body's own trunk vector), which makes it the camera-robust one, while
     * `hold` inherits `WORLD_UP`'s level-camera assumption. The `up` row on the overlay says which.
     */
    private fun toggleMode() {
        if (overlayOff) {
            // Nothing to rebuild: `hold` has been running behind the hidden panel since `off` was
            // entered, so the readout is already there to show.
            overlayOff = false
            return
        }
        if (liveMode == LiveMetricsComputer.HOLD) {
            liveMode = LiveMetricsComputer.CRAWL
        } else {
            overlayOff = true
            liveMode = LiveMetricsComputer.HOLD
        }
        metrics = null
        pipeline?.setMode(liveMode)
    }

    /**
     * Switch between the tracked, redacted view and the captured video on its own.
     *
     * Turning it on stops pose detection, face detection and redaction together, so the skeleton
     * and the readout go with them — there is nothing behind either. **It is also the one state
     * where an unredacted face is on screen**, which is why [CameraScreen] draws a standing
     * warning while it is on rather than relying on the button label.
     *
     * The rolling window is discarded either way. Nothing was tracked while raw, so returning to
     * the tracked view is resuming after a gap, and a window bridging it would average across
     * however long the operator spent looking at the plain video.
     */
    private fun toggleVideoOnly() {
        videoOnly = !videoOnly
        metrics = null
        pipeline?.videoOnly = videoOnly
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            try {
                val provider = providerFuture.get()
                cameraProvider = provider

                val selector = CameraSelector.Builder().requireLensFacing(lensFacing).build()
                if (!provider.hasCamera(selector)) {
                    Log.w(TAG, "no camera for lensFacing=$lensFacing; staying on the current one")
                    return@addListener
                }
                canFlipCamera = provider.hasCamera(CameraSelector.DEFAULT_BACK_CAMERA) &&
                    provider.hasCamera(CameraSelector.DEFAULT_FRONT_CAMERA)

                // Built once for the life of the activity. Loading the 5.7 MB pose bundle is not
                // the reason — the reason is that destroying a MediaPipe task while frames are in
                // flight is a native use-after-free.
                val pipe = pipeline ?: PosePipeline(this, liveMode) { rendered ->
                    // MediaPipe calls back off the main thread; Compose state must be written on it.
                    runOnUiThread {
                        bitmap = rendered.bitmap
                        frame = rendered.frame
                        metrics = rendered.metrics
                        fps = rendered.fps
                    }
                }
                pipeline = pipe
                usingGpu = pipe.usingGpu

                // The bound size is expressed in the *target rotation's* coordinate frame, so a
                // portrait target rotation reads `Size(1280, 720)` as 1280 tall. Swapping it asks
                // for the same sensor mode in both orientations, delivered 720x1280 in portrait.
                val target = if (portrait) {
                    Size(TARGET_HEIGHT, TARGET_WIDTH)
                } else {
                    Size(TARGET_WIDTH, TARGET_HEIGHT)
                }

                val analysis = ImageAnalysis.Builder()
                    .setResolutionSelector(
                        ResolutionSelector.Builder()
                            .setResolutionStrategy(
                                ResolutionStrategy(
                                    target,
                                    ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER,
                                )
                            )
                            .build()
                    )
                    // Drop frames rather than queue them. A readout that is current and
                    // occasionally sparse beats one that is complete and progressively later —
                    // and the metric chain is built for a jittery timebase anyway.
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    // Let CameraX rotate the buffer inside its own reused ImageReader, so the
                    // delivered frame is already upright and `rotationDegrees` stays 0.
                    //
                    // This is a memory decision, not a convenience. `PosePipeline.rotated()` only
                    // hits its zero-copy fast path at 0 degrees; in portrait it would otherwise do
                    // a full `createBitmap` per frame — a second ~3.7 MB allocation that
                    // `BitmapRing` does *not* serve, on top of the one `toBitmap()` already makes.
                    // `BitmapRing`'s own docs record the GC losing that race at ~100 MB/s. The
                    // manual rotation stays in place as a correctness backstop for any device
                    // where this silently declines.
                    .setOutputImageRotationEnabled(true)
                    .build()
                analysis.setAnalyzer(analysisExecutor, pipe)

                provider.unbindAll()
                // No `Preview` use case on purpose: it would put the raw camera stream on screen
                // without passing through face redaction. See `FaceRedaction`.
                //
                // The front camera's frames are deliberately **not mirrored**. A selfie preview
                // flips horizontally to look natural, but that flip would swap the child's left and
                // right as MediaPipe sees them — and the sign of `live_leg_amplitude_symmetry` is
                // precisely "which leg", the signal this whole mode exists to report. A mirrored
                // front-camera session would silently invert it. The un-mirrored view reads oddly
                // if you point it at yourself; it is correct when pointed at Remy, which is the
                // only thing this app is for.
                provider.bindToLifecycle(this, selector, analysis)
            } catch (e: Exception) {
                Log.e(TAG, "camera start failed", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    override fun onDestroy() {
        super.onDestroy()
        pipeline?.close()
        analysisExecutor.shutdown()
    }
}

/**
 * Shown in place of the live view when the camera permission has not been granted.
 *
 * Reached by tapping into the live view, so it needs a way back out: a denied permission would
 * otherwise be a dead end on a screen with no system bars to press back against.
 */
@androidx.compose.runtime.Composable
private fun PermissionPrompt(
    onRequest: () -> Unit,
    onBack: () -> Unit,
    modifier: Modifier = Modifier,
) {
    BackHandler(onBack = onBack)
    Column(
        modifier = modifier.padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp, Alignment.CenterVertically),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text("remapy needs the camera to see the session.")
        Button(onClick = onRequest) { Text("Grant camera access") }
        TextButton(onClick = onBack) { Text("Back to menu") }
    }
}
