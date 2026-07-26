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
 * The live view.
 *
 * Phase 1 of the Android port: camera + pose + live metrics, for an **observer** watching Remy
 * — he never looks at this screen. Nothing is persisted; the desktop pipeline stays canonical for
 * recordings, annotations and the cross-session trend.
 *
 * Screen-on, and landscape by default, matching how the tripod-mounted camera is used in the
 * data-collection runbook — but the framing is switchable at runtime via the overlay's
 * orientation toggle, since a standing or crawling child frames better vertically.
 */
class MainActivity : ComponentActivity() {

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

    private var lensFacing by mutableStateOf(CameraSelector.LENS_FACING_BACK)

    /** Which exercise the live readout is measuring. `hold` is the common case, so it is the default. */
    private var liveMode by mutableStateOf(LiveMetricsComputer.HOLD)

    /** Whether the *other* lens exists. Tablets and some phones have only one. */
    private var canFlipCamera by mutableStateOf(false)

    /**
     * Whether the view is framed vertically. Driven by the toggle, never by the sensor.
     *
     * A tripod-mounted phone must not reframe itself because someone picked it up, and every
     * orientation change throws away the rolling window (see [rebindForOrientation]) — so this is
     * a deliberate act, taken between trials, and the button shows which state is live.
     */
    private var portrait by mutableStateOf(false)

    private var cameraProvider: ProcessCameraProvider? = null

    private val requestCamera = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        hasCamera = granted
        if (granted) startCamera()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

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
            MaterialTheme {
                if (hasCamera) {
                    CameraScreen(
                        bitmap = bitmap,
                        frame = frame,
                        metrics = metrics,
                        fps = fps,
                        usingGpu = usingGpu,
                        lensFacing = lensFacing,
                        canFlipCamera = canFlipCamera,
                        onFlipCamera = ::flipCamera,
                        mode = liveMode,
                        onToggleMode = ::toggleMode,
                        portrait = portrait,
                        onToggleOrientation = ::toggleOrientation,
                        modifier = Modifier.fillMaxSize(),
                    )
                } else {
                    PermissionPrompt(
                        onRequest = { requestCamera.launch(Manifest.permission.CAMERA) },
                        modifier = Modifier.fillMaxSize(),
                    )
                }
            }
        }

        if (hasCamera) startCamera() else requestCamera.launch(Manifest.permission.CAMERA)
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
        rebindForOrientation()
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
     * Switch the live readout between sitting/standing holds and belly-crawl.
     *
     * Only the metric dispatch changes — the camera keeps running and the models stay loaded. The
     * readout blanks and re-warms over the next few seconds because the rolling window is
     * discarded; see [PosePipeline.setMode] for why that is not avoidable.
     *
     * Worth knowing which mode you are in beyond the label: `crawl` reads **no vertical at all**
     * (the axis is the body's own trunk vector), which makes it the camera-robust one, while
     * `hold` inherits `WORLD_UP`'s level-camera assumption. The `up` row on the overlay says which.
     */
    private fun toggleMode() {
        liveMode = if (liveMode == LiveMetricsComputer.HOLD) {
            LiveMetricsComputer.CRAWL
        } else {
            LiveMetricsComputer.HOLD
        }
        metrics = null
        pipeline?.setMode(liveMode)
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

@androidx.compose.runtime.Composable
private fun PermissionPrompt(onRequest: () -> Unit, modifier: Modifier = Modifier) {
    Column(
        modifier = modifier.padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp, Alignment.CenterVertically),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text("remapy needs the camera to see the session.")
        Button(onClick = onRequest) { Text("Grant camera access") }
    }
}
