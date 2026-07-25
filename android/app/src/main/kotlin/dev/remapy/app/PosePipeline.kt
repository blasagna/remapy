package dev.remapy.app

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.os.SystemClock
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facedetector.FaceDetector
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import dev.remapy.metrics.LiveMetrics
import dev.remapy.metrics.LiveMetricsComputer
import dev.remapy.metrics.PoseFrame
import java.util.concurrent.ConcurrentHashMap

/**
 * One rendered frame: the image the operator may see, and everything measured from it.
 *
 * [bitmap] has already been through [FaceRedaction] — nothing upstream of this type is safe to
 * display.
 */
class RenderedFrame(
    val bitmap: Bitmap,
    val frame: PoseFrame,
    val metrics: LiveMetrics,
    val fps: Double,
)

/**
 * Camera -> pose -> live metrics, wired to a CameraX `ImageAnalysis`.
 *
 * **`RunningMode.LIVE_STREAM`, not `VIDEO`.** The desktop CLIs use `detect_for_video`, which
 * blocks, because their loop is a synchronous pull. Here the correct shape is the async callback:
 * `LiveMetricsComputer.push` is already callback-shaped, and CameraX's
 * `STRATEGY_KEEP_ONLY_LATEST` drops frames under load rather than queueing them, so the UI stays
 * current instead of falling progressively behind.
 *
 * Dropped frames are *fine* by construction, and that is not luck: `Derive.resampleUniform` puts
 * every signal on a uniform grid before differentiating, and `LiveWindow.windowSpan` selects the
 * window by **time rather than frame count**. Both were written for a jittery 30 Hz webcam and
 * both do exactly the right thing for a phone that occasionally drops to 22 fps. What they cannot
 * absorb is a *sustained* rate well under 30 Hz — see [RenderedFrame.fps], which is on screen for
 * that reason.
 */
class PosePipeline(
    context: Context,
    private val mode: String,
    private val onFrame: (RenderedFrame) -> Unit,
) : ImageAnalysis.Analyzer {

    companion object {
        private const val TAG = "PosePipeline"
        private const val POSE_MODEL = "pose_landmarker_lite.task"
        private const val FACE_MODEL = "blaze_face_short_range.tflite"

        /** Matches the Python `PoseEstimator` defaults, which match MediaPipe's own. */
        private const val MIN_DETECTION_CONFIDENCE = 0.5f
        private const val MIN_PRESENCE_CONFIDENCE = 0.5f
        private const val MIN_TRACKING_CONFIDENCE = 0.5f
    }

    /** Whether the pose model is running on the GPU delegate. Reported on screen. */
    var usingGpu: Boolean = false
        private set

    var redactionMethod: FaceRedaction.Method = FaceRedaction.Method.HYBRID
    var redactionStyle: FaceRedaction.Style = FaceRedaction.Style.BOX

    /**
     * Whether to blank a frame outright when no face could be located.
     *
     * On by default. With [FaceRedaction.Method.POSE] there is no fallback detector, so a frame
     * with no tracked pose would otherwise be displayed unredacted — and even under `HYBRID` both
     * locators can miss. Blanking is jarring; showing an unredacted face is worse.
     */
    var blankWhenUnlocated: Boolean = true

    /**
     * Replaced wholesale by [reset]; never mutated in place.
     *
     * Reassigning a `LiveMetricsComputer` is how the rolling window is discarded at a camera
     * switch. The kernel has no `reset()` of its own and should not grow one for this: it is
     * verified against the Python implementation, which has no such concept because an offline
     * segment cannot span a lens change.
     */
    @Volatile
    private var computer = LiveMetricsComputer(mode)
    private val displayRing = BitmapRing()
    private val pending = ConcurrentHashMap<Long, Pending>()
    private val startedAtMs = SystemClock.elapsedRealtime()
    private var lastTimestampMs = -1L
    private var lastFrameAtMs = 0L
    private var smoothedFps = 0.0

    /**
     * Set by [close]; stops frames being emitted after teardown.
     *
     * Volatile because it is written on the main thread and read on the analyzer and detector
     * callback threads. Without it, an in-flight result from the *previous* camera can land in the
     * UI after a flip — putting a stale frame on screen under live-looking chrome, and reviving a
     * reference to a buffer whose owner has already gone away.
     */
    @Volatile
    private var closed = false

    private class Pending(val bitmap: Bitmap, val image: MPImage)

    private val landmarker: PoseLandmarker = buildLandmarker(context)

    /**
     * Built eagerly even under `POSE` redaction: switching method at runtime must not have to
     * construct a detector on the camera thread, and 230 KB of model is cheaper than the stall.
     */
    private val faceDetector: FaceDetector = FaceDetector.createFromOptions(
        context,
        FaceDetector.FaceDetectorOptions.builder()
            .setBaseOptions(BaseOptions.builder().setModelAssetPath(FACE_MODEL).build())
            .setRunningMode(RunningMode.VIDEO)
            .build(),
    )

    private fun buildLandmarker(context: Context): PoseLandmarker {
        fun build(delegate: Delegate): PoseLandmarker = PoseLandmarker.createFromOptions(
            context,
            PoseLandmarker.PoseLandmarkerOptions.builder()
                .setBaseOptions(
                    BaseOptions.builder()
                        .setModelAssetPath(POSE_MODEL)
                        .setDelegate(delegate)
                        .build()
                )
                .setRunningMode(RunningMode.LIVE_STREAM)
                .setNumPoses(1)
                .setMinPoseDetectionConfidence(MIN_DETECTION_CONFIDENCE)
                .setMinPosePresenceConfidence(MIN_PRESENCE_CONFIDENCE)
                .setMinTrackingConfidence(MIN_TRACKING_CONFIDENCE)
                .setResultListener(::onResult)
                .setErrorListener { e -> Log.e(TAG, "pose landmarker error", e) }
                .build(),
        )

        return try {
            build(Delegate.GPU).also { usingGpu = true }
        } catch (e: RuntimeException) {
            // Not every device has a working GPU delegate, and a session that fails to start is
            // worse than one that runs slower. The fallback is worth *noticing*, though: it is a
            // plausible cause of a device that cannot hold 30 fps.
            Log.w(TAG, "GPU delegate unavailable, falling back to CPU", e)
            build(Delegate.CPU).also { usingGpu = false }
        }
    }

    override fun analyze(image: ImageProxy) {
        if (closed) {
            image.close()
            return
        }
        try {
            val source = image.toBitmap()
            val rotated = source.rotated(image.imageInfo.rotationDegrees)
            // `rotated` may be a new bitmap; if so the source is dead weight on the native heap
            // until a GC that may not come soon enough. Only `rotated` is owned downstream (the
            // MPImage recycles it on close).
            if (rotated !== source) source.recycle()
            // MediaPipe requires strictly increasing timestamps; a stalled clock would otherwise
            // silently drop frames inside the task.
            var timestampMs = SystemClock.elapsedRealtime() - startedAtMs
            if (timestampMs <= lastTimestampMs) timestampMs = lastTimestampMs + 1
            lastTimestampMs = timestampMs

            val mpImage = BitmapImageBuilder(rotated).build()
            pending[timestampMs] = Pending(rotated, mpImage)
            landmarker.detectAsync(mpImage, timestampMs)
        } catch (e: RuntimeException) {
            Log.e(TAG, "analyze failed", e)
        } finally {
            image.close()
        }
    }

    /**
     * The result callback. **Order here is load-bearing**, and mirrors the capture loop in
     * `rerun_viewer/main.py`: detection has already run on the raw frame, the metrics are computed
     * from the raw landmarks, and redaction touches only the bitmap that will be displayed. Pose
     * accuracy is therefore unaffected by redaction, and nothing unredacted reaches the screen.
     */
    private fun onResult(result: PoseLandmarkerResult, @Suppress("UNUSED_PARAMETER") input: MPImage) {
        val timestampMs = result.timestampMs()
        val held = pending.remove(timestampMs) ?: return
        dropStale(timestampMs)
        if (closed) {
            held.image.close()
            return
        }

        try {
            val frame = LandmarkRows.from(result)
            val metrics = computer.push(timestampMs, frame)

            val hasPose = !frame.world[0].isNaN()
            val detections = if (
                redactionMethod == FaceRedaction.Method.DETECTOR ||
                (redactionMethod == FaceRedaction.Method.HYBRID && !hasPose)
            ) {
                faceDetector.detectForVideo(held.image, timestampMs).detections()
            } else {
                emptyList()
            }

            // Redact a *copy*, never the bitmap the MPImage wraps. `MPImage.close()` recycles the
            // bitmap it was built from, and the display bitmap outlives this callback by however
            // long the UI takes to draw it — sharing them crashes the next compose pass with
            // "trying to use a recycled bitmap". The copy comes from a ring rather than a fresh
            // allocation: see `BitmapRing` for why that distinction is load-bearing and not a
            // micro-optimisation.
            val display = displayRing.acquire(held.bitmap.width, held.bitmap.height)
            Canvas(display).drawBitmap(held.bitmap, 0f, 0f, null)
            val redacted = FaceRedaction.redact(
                display,
                if (hasPose) frame else null,
                detections,
                redactionMethod,
                redactionStyle,
            )
            if (!redacted && blankWhenUnlocated) FaceRedaction.redactAll(display, redactionStyle)

            onFrame(RenderedFrame(display, frame, metrics, updateFps()))
        } catch (e: RuntimeException) {
            // Inside the capture path an unexpected throw costs the session, not one frame — the
            // same reasoning `LiveMetricsComputer.compute` documents.
            Log.e(TAG, "result handling failed", e)
        } finally {
            held.image.close()
        }
    }

    /** Frames whose results never arrived would otherwise leak their bitmaps. */
    private fun dropStale(currentMs: Long) {
        val iterator = pending.entries.iterator()
        while (iterator.hasNext()) {
            val entry = iterator.next()
            if (entry.key < currentMs) {
                entry.value.image.close()
                iterator.remove()
            }
        }
    }

    /** Exponentially smoothed delivered frame rate — the number risk #4 is watched with. */
    private fun updateFps(): Double {
        val now = SystemClock.elapsedRealtime()
        if (lastFrameAtMs != 0L) {
            val instant = 1000.0 / (now - lastFrameAtMs).coerceAtLeast(1)
            smoothedFps = if (smoothedFps == 0.0) instant else 0.9 * smoothedFps + 0.1 * instant
        }
        lastFrameAtMs = now
        return smoothedFps
    }

    /**
     * Start over on a fresh rolling window, keeping the loaded models.
     *
     * Called when the camera changes. The window is a trailing few seconds of *one* view of the
     * child, and the two lenses differ in framing, field of view and sensor — a window spanning the
     * switch would average sway across a discontinuity that has nothing to do with him, the same
     * refusal `longest_run` makes about bridging a tracking dropout.
     *
     * **This exists instead of rebuilding the pipeline**, which is what the first version did and
     * what segfaulted: closing a `PoseLandmarker` on the main thread while the analyzer thread is
     * inside `detectAsync` is a use-after-free in native code, and a `closed` flag only narrows
     * that window rather than closing it. Not destroying the task at all removes the race by
     * construction, and makes the flip faster besides — no 5.7 MB model reload.
     *
     * `lastTimestampMs` is deliberately *not* rewound: MediaPipe requires monotonically increasing
     * timestamps across the life of the task, and the task survives this call.
     */
    fun reset() {
        computer = LiveMetricsComputer(mode)
        smoothedFps = 0.0
        lastFrameAtMs = 0L
    }

    fun close() {
        // Flag first: it is what stops a result already in flight from reaching the UI while the
        // rest of this method pulls the resources out from under it. Teardown only happens when
        // the activity is going away, so nothing needs the pipeline afterwards.
        closed = true
        landmarker.close()
        faceDetector.close()
        pending.values.forEach { it.image.close() }
        pending.clear()
        displayRing.release()
    }

    private fun Bitmap.rotated(degrees: Int): Bitmap {
        // Rotate here rather than passing ImageProcessingOptions to MediaPipe: that would leave the
        // landmarks in a rotated frame while this bitmap stayed in the sensor's, and every overlay
        // and redaction box would then need the transform applied by hand.
        //
        // At 0 degrees this returns the input untouched. `OUTPUT_IMAGE_FORMAT_RGBA_8888` already
        // gives a mutable ARGB_8888, so the defensive copy the general path needs would be a
        // wasted 3.7 MB allocation on every frame of the common case.
        if (degrees == 0 && config == Bitmap.Config.ARGB_8888) return this
        return Bitmap.createBitmap(
            this, 0, 0, width, height, Matrix().apply { postRotate(degrees.toFloat()) }, true,
        )
    }
}
