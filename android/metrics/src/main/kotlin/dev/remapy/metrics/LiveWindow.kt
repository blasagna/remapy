package dev.remapy.metrics

/**
 * One frame's landmarks, as the detector produced them.
 *
 * The Kotlin equivalent of `recording/recorder.py`'s `landmark_rows`, and it exists for the
 * same reason: **the full-NaN convention for a frame with no pose is load-bearing.**
 * [PoseFrames.posePresent] and all of [Quality] key off it — a NaN landmark-0 x implies the
 * whole row is NaN — so a second conversion that filled zeros instead, or wrote NaN
 * coordinates but left visibility at 0.0, would produce frames that read as tracked while
 * carrying nothing.
 *
 * Note what this type is *not*: it carries no MediaPipe class. Converting a
 * `PoseLandmarkerResult` into one of these is the app module's job, which is what keeps the
 * kernel testable on a desktop JVM with no detector present.
 */
class PoseFrame(
    val world: FloatArray,
    val norm: FloatArray,
    val visibility: FloatArray,
    val presence: FloatArray,
) {
    init {
        require(world.size == Landmarks.COUNT * 3) { "world must be ${Landmarks.COUNT * 3} floats." }
        require(norm.size == Landmarks.COUNT * 3) { "norm must be ${Landmarks.COUNT * 3} floats." }
        require(visibility.size == Landmarks.COUNT) { "visibility must be ${Landmarks.COUNT} floats." }
        require(presence.size == Landmarks.COUNT) { "presence must be ${Landmarks.COUNT} floats." }
    }

    companion object {
        /** Exactly what the detector emits when it finds nothing: full-NaN rows. */
        fun noPose(): PoseFrame = PoseFrame(
            FloatArray(Landmarks.COUNT * 3) { Float.NaN },
            FloatArray(Landmarks.COUNT * 3) { Float.NaN },
            FloatArray(Landmarks.COUNT) { Float.NaN },
            FloatArray(Landmarks.COUNT) { Float.NaN },
        )
    }
}

/**
 * A fixed-capacity ring buffer that *is* a recording as far as the metrics care.
 *
 * Port of `motor_metrics.live.LiveWindow`. The point of it is that it implements
 * [PoseFrames], so [Hold.holdMetrics] and [Crawl.crawlMetrics] run over a live buffer
 * **unmodified** instead of against a parallel implementation that would drift from the
 * offline one. There is no annotator live, which is the whole reason this windows on a clock
 * instead of segmenting on marks.
 */
class LiveWindow(capacity: Int) : PoseFrames {

    init {
        require(capacity >= 1) { "capacity must be positive, got $capacity." }
    }

    private val cap = capacity
    private val timestamps = LongArray(cap)
    private val frames = arrayOfNulls<PoseFrame>(cap)

    /** Total ever pushed; the write cursor is `pushed % cap`. */
    private var pushed = 0

    override val size: Int get() = minOf(pushed, cap)

    /** Index into the backing arrays for the [frame]-th oldest retained frame. */
    private fun slot(frame: Int): Int =
        if (pushed < cap) frame else (pushed % cap + frame) % cap

    /** Append one frame, evicting the oldest when full. */
    fun push(timestampMs: Long, frame: PoseFrame) {
        val i = pushed % cap
        frames[i] = frame
        timestamps[i] = timestampMs
        pushed++
    }

    override fun timestampMs(frame: Int): Long = timestamps[slot(frame)]

    override fun worldRow(frame: Int): FloatArray = frames[slot(frame)]!!.world

    override fun normRow(frame: Int): FloatArray = frames[slot(frame)]!!.norm

    override fun visibility(frame: Int, landmark: Int): Float =
        frames[slot(frame)]!!.visibility[landmark]

    override fun presence(frame: Int, landmark: Int): Float =
        frames[slot(frame)]!!.presence[landmark]

    /**
     * The `[start, stop)` covering the last [windowS] seconds of the buffer.
     *
     * Selected by **time, not by a frame count**, so the window keeps its duration whatever
     * rate the camera actually delivers. That property is doing more work on Android than it
     * does on the laptop: MediaPipe's LIVE_STREAM mode drops frames under load rather than
     * queueing them, so the delivered rate is genuinely variable, and a frame-count window
     * would silently become a *shorter* window exactly when the device is struggling.
     */
    fun windowSpan(windowS: Double): Span {
        val n = size
        if (n == 0) return Span(0, 0)
        val cutoff = timestampMs(n - 1) - windowS * 1000.0
        // searchsorted(..., side="left"): first index whose timestamp is >= the cutoff.
        var lo = 0
        var hi = n
        while (lo < hi) {
            val mid = (lo + hi) ushr 1
            if (timestampMs(mid) < cutoff) lo = mid + 1 else hi = mid
        }
        return Span(lo, n)
    }
}
