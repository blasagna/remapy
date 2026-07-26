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
import dev.remapy.metrics.Derive
import dev.remapy.metrics.LiveMetrics
import dev.remapy.metrics.LiveMetricsComputer
import dev.remapy.metrics.PoseFrame
import java.util.concurrent.ConcurrentHashMap

/**
 * One rendered frame: the image the operator may see, and everything measured from it.
 *
 * [bitmap] has been through [FaceRedaction] **unless [PosePipeline.videoOnly] is set**, which is
 * the deliberate raw-video state and the only one that reaches here unredacted. Either way nothing
 * *upstream* of this type is safe to display: the redaction decision has already been taken by the
 * time a frame is wrapped in one of these, and no other code gets to revisit it.
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
 * window by **time rather than frame count**. Both were written for a jittery webcam and both do
 * exactly the right thing for a phone that misses the occasional frame. What they cannot absorb is
 * a *sustained* rate well under the grid — see [RenderedFrame.fps], which is on screen for that
 * reason.
 *
 * **The analyzer decimates to `Derive.FS` itself.** The camera is left free-running and the drop
 * happens here, in [analyze], rather than by asking the sensor for a slower capture. Capping the
 * sensor was considered and rejected: a fixed 15 fps AE range doubles the maximum exposure from
 * 33 ms to 67 ms, and these sessions happen across a room in indifferent lighting — precisely when
 * AE takes the long exposure, and a 67 ms exposure smears a crawling child into landmarks no
 * amount of downstream filtering repairs. Leaving the source at 30 also means a slow inference
 * costs 33 ms of recovery rather than 67. The power saving was not worth the input quality.
 */
class PosePipeline(
    context: Context,
    initialMode: String,
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

        /**
         * The rate frames are accepted at, taken from the kernel's grid rather than written down.
         *
         * Capping and the grid are the *same decision*: the point of running at 15 is that
         * `Derive.resampleUniform` then neither interpolates up (manufacturing correlated samples)
         * nor decimates down without an anti-alias filter. A second literal here would be the next
         * thing to drift out of step with `Derive.FS`.
         */
        private val TARGET_FPS: Int = Derive.FS.toInt()
        private val FRAME_PERIOD_MS: Long = (1000.0 / TARGET_FPS).toLong()

        /**
         * How early a frame may arrive and still be accepted, in ms.
         *
         * A source running at 30 fps delivers every ~33 ms with jitter, and a naive
         * `sinceLast >= 67` test beats against that: an arrival at 66 ms is rejected, the next
         * comes at 99, and the effective rate sags to ~14 fps with a 100 ms hole in it. A quarter
         * of a 30 fps period is enough slack to take the 66 ms frame.
         */
        private const val PHASE_TOLERANCE_MS = 8L
    }

    /** Whether the pose model is running on the GPU delegate. Reported on screen. */
    var usingGpu: Boolean = false
        private set

    /** `hold` or `crawl` — which metric the rolling window is dispatched to. Change via [setMode]. */
    var mode: String = initialMode
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
     * Show the captured frame as-is: no pose, no face detection, no redaction.
     *
     * **The one state in which an unredacted face reaches the screen.** The equivalent of the
     * desktop CLIs' `--no-blur-faces`, and subject to the same rule that makes that acceptable —
     * nothing here is recorded, so the frame is on screen and nowhere else. `MainActivity` owns
     * the warning that says so, and never persists this across a launch.
     *
     * Volatile because [analyze] reads it on the analyzer thread while the toggle writes it on
     * main. Reading a stale value costs one frame of the previous mode, which is why the flag is
     * enough on its own and no lock is warranted: unlike [close], there is no native object whose
     * lifetime depends on the answer.
     */
    @Volatile
    var videoOnly: Boolean = false

    /**
     * Replaced wholesale by [reset]; never mutated in place.
     *
     * Reassigning a `LiveMetricsComputer` is how the rolling window is discarded at a camera
     * switch. The kernel has no `reset()` of its own and should not grow one for this: it is
     * verified against the Python implementation, which has no such concept because an offline
     * segment cannot span a lens change.
     */
    @Volatile
    private var computer = LiveMetricsComputer(initialMode)
    private val displayRing = BitmapRing()
    private val pending = ConcurrentHashMap<Long, Pending>()
    private val startedAtMs = SystemClock.elapsedRealtime()
    private var lastTimestampMs = -1L

    // Volatile because [updateFps] is now reached from two threads: the MediaPipe callback on the
    // tracked path, and the analyzer thread directly on the [videoOnly] one. Only one is live at a
    // time, but a toggle can land with a result still in flight, and a torn `double` read would
    // put a nonsense frame rate on screen — the one number the operator uses to judge the rest.
    @Volatile
    private var lastFrameAtMs = 0L

    @Volatile
    private var smoothedFps = 0.0

    /**
     * Blank-frame accounting for [noteBlanked]: consecutive blanks, and the session total.
     *
     * Kept after the HYBRID gate was fixed rather than removed with it, because the screen cannot
     * distinguish a *correct* blank — nobody in frame, so nothing to redact — from the bug coming
     * back, and both look like the video stuttering to black. A count with `hasPose` beside it
     * separates them, and gives the next session a number instead of an impression.
     */
    private var blankRun = 0
    private var blankTotal = 0L

    /**
     * When the next frame is due, as a phase accumulator rather than a "time since last".
     *
     * Advancing by `max(now, nextDueMs) + period` holds phase against a faster source while still
     * re-phasing cleanly after a stall, instead of letting the accepted rate drift with whatever
     * jitter the last accepted frame happened to carry.
     */
    private var nextDueMs = 0L

    /**
     * Last logged frame geometry, so the geometry log fires on change rather than per frame.
     *
     * This log *is* the check on `setOutputImageRotationEnabled`: it must read `rotationDegrees=0`
     * in both orientations. A non-zero value means CameraX declined and every frame is taking the
     * manual `createBitmap` path, which is the allocation pattern `BitmapRing` exists to avoid.
     */
    @Volatile
    private var lastGeometry: String? = null

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
            // plausible cause of a device that cannot hold the target rate.
            Log.w(TAG, "GPU delegate unavailable, falling back to CPU", e)
            build(Delegate.CPU).also { usingGpu = false }
        }
    }

    override fun analyze(image: ImageProxy) {
        if (closed) {
            image.close()
            return
        }
        // Decimate to TARGET_FPS *before* any allocation, so a dropped frame costs one `close()`
        // and nothing else — no `toBitmap`, no MPImage, no ring slot.
        //
        // This deliberately sits before the timestamp block below, and that placement is what
        // keeps the metrics correct rather than merely cheap: `timestampMs` is only ever assigned
        // for *accepted* frames, from the true wall clock, so the kernel sees a genuine ~15 Hz
        // jittery timebase. `resampleUniform` and `windowSpan` were written for exactly that, so
        // decimating needs no compensating change anywhere in `:metrics`. It looks like it should.
        val nowMs = SystemClock.elapsedRealtime()
        if (nowMs < nextDueMs - PHASE_TOLERANCE_MS) {
            image.close()
            return
        }
        nextDueMs = maxOf(nowMs, nextDueMs) + FRAME_PERIOD_MS

        val geometry = "${image.width}x${image.height} rot=${image.imageInfo.rotationDegrees}"
        if (geometry != lastGeometry) {
            lastGeometry = geometry
            Log.i(TAG, "frame geometry $geometry (rot must be 0; non-zero costs a copy per frame)")
        }

        try {
            val source = image.toBitmap()
            val rotated = source.rotated(image.imageInfo.rotationDegrees)
            // `rotated` may be a new bitmap; if so the source is dead weight on the native heap
            // until a GC that may not come soon enough. Only `rotated` is owned downstream (the
            // MPImage recycles it on close).
            if (rotated !== source) source.recycle()

            if (videoOnly) {
                emitRaw(rotated)
                return
            }

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
     * Emit a frame with nothing measured from it and nothing redacted on it.
     *
     * The bypass for [videoOnly]. It exists as its own path because the normal one is driven
     * entirely by the pose result callback — skip `detectAsync` and no frame would ever reach the
     * screen at all.
     *
     * **[rotated] is recycled here, and that is the whole of the memory story on this path.** On
     * the tracked path the `MPImage` owns it and recycles it on close; here nothing downstream
     * does. A 1280x720 ARGB bitmap is ~3.7 MB on the native heap and this runs at 15 Hz, which is
     * the rate `BitmapRing`'s notes record the GC losing to. Dropping the reference and trusting
     * collection reads fine for thirty seconds and OOM-kills partway through a session.
     *
     * The readout is [LiveMetrics.blank] rather than a push into the window. There is nothing to
     * push — no pose ran — and feeding no-pose rows in would fill the rolling window with dropout
     * for as long as the operator stays on plain video. `up` is reported as `n/a` because no
     * vertical was consulted, which is the same string the crawl path uses for the same reason.
     */
    private fun emitRaw(rotated: Bitmap) {
        val display = displayRing.acquire(rotated.width, rotated.height)
        Canvas(display).drawBitmap(rotated, 0f, 0f, null)
        rotated.recycle()
        onFrame(
            RenderedFrame(
                display,
                PoseFrame.noPose(),
                LiveMetrics.blank(mode, computer.windowS, 0, 0.0, 0.0, "n/a"),
                updateFps(),
            )
        )
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
            // Hoisted so the predicate below and `redact` below that cannot be handed different
            // frames — the detector must be run for exactly the frames the pose path will fail on.
            val poseFrame = if (hasPose) frame else null
            val detections = if (
                redactionMethod == FaceRedaction.Method.DETECTOR ||
                // Not `!hasPose`: a tracked body whose face keypoints are all below the visibility
                // gate yields no pose head box either, and that is the case HYBRID's fallback is
                // for. Asking the narrower question left `detections` empty exactly there, so
                // nothing was redacted and `blankWhenUnlocated` blanked the frame — the black
                // frames seen mid-crawl, when the head is down or turned away.
                (redactionMethod == FaceRedaction.Method.HYBRID &&
                    !FaceRedaction.hasPoseFaceBox(poseFrame))
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
                poseFrame,
                detections,
                redactionMethod,
                redactionStyle,
            )
            if (!redacted && blankWhenUnlocated) {
                FaceRedaction.redactAll(display, redactionStyle)
                noteBlanked(hasPose, detections.size)
            } else {
                blankRun = 0
            }

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

    /**
     * Record that a frame was blanked because nothing located a face.
     *
     * Logs once per *burst* rather than per frame: at 15 Hz a per-frame log would bury the rest of
     * the tag, and the useful signal is that a run started and what the redaction inputs looked
     * like when it did. `pose=true` with `detections=0` is the diagnosis — a body tracked, its face
     * keypoints below the gate and the detector finding nothing either — while `pose=false` with an
     * empty frame is the behaviour working as designed.
     */
    private fun noteBlanked(hasPose: Boolean, detectionCount: Int) {
        blankTotal++
        if (blankRun++ == 0) {
            Log.i(TAG, "blanked frame: pose=$hasPose detections=$detectionCount total=$blankTotal")
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
        // Re-phase the decimator too. A rebind can leave `nextDueMs` a whole period in the future,
        // which would drop the first frame of the new camera for no reason and delay the first
        // readout at exactly the moment the operator is looking for it.
        nextDueMs = 0L
        // Only the run, not the total: a lens or framing change is exactly when redaction can start
        // failing, so the first blank after one should log rather than be swallowed by a run that
        // began under the old view.
        blankRun = 0
    }

    /**
     * Switch between `hold` and `crawl`.
     *
     * The rolling window is discarded, so the readout blanks and re-warms over the next few
     * seconds. That is not avoidable by keeping the buffered frames: the two modes use *different
     * window lengths* (5 s for a hold, 6 s for a crawl, because cadence variability needs several
     * pull cycles before it means anything), so there is no single buffer that is correct for both.
     * It is also not a problem in practice — a mode switch is a deliberate act between trials, not
     * something done mid-measurement.
     */
    fun setMode(newMode: String) {
        if (newMode == mode) return
        require(LiveMetricsComputer.MODE_WINDOW_S.containsKey(newMode)) {
            "mode must be one of ${LiveMetricsComputer.MODE_WINDOW_S.keys}, got '$newMode'."
        }
        mode = newMode
        reset()
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
