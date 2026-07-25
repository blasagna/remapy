package dev.remapy.app

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
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
 * Landscape-locked and screen-on, matching how the tripod-mounted camera is used in the
 * data-collection runbook.
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

    private val requestCamera = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        hasCamera = granted
        if (granted) startCamera()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
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

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            try {
                val provider = providerFuture.get()
                val pipe = PosePipeline(this, LiveMetricsComputer.HOLD) { rendered ->
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

                val analysis = ImageAnalysis.Builder()
                    .setResolutionSelector(
                        ResolutionSelector.Builder()
                            .setResolutionStrategy(
                                ResolutionStrategy(
                                    android.util.Size(TARGET_WIDTH, TARGET_HEIGHT),
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
                    .build()
                analysis.setAnalyzer(analysisExecutor, pipe)

                provider.unbindAll()
                // No `Preview` use case on purpose: it would put the raw camera stream on screen
                // without passing through face redaction. See `FaceRedaction`.
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, analysis)
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
