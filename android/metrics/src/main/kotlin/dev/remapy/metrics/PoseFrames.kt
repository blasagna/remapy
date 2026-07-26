package dev.remapy.metrics

/**
 * The surface every metric here reads: a sequence of timestamped pose frames.
 *
 * This is the Kotlin spelling of the duck type the Python metrics rely on. `hold_metrics`
 * and `crawl_metrics` never open a file — they touch `timestamps_ms`, `landmarks_world`,
 * `landmarks_norm`, `visibility`, `presence` and `pose_present`, which is why
 * `tests.fakes.fake_recording` needs no HDF5 and why `LiveWindow` can be a ring buffer that
 * *is* a recording as far as the metrics care. Making it an interface here preserves the one
 * property that matters: **there is no second implementation of the metric maths** for the
 * live path, so live and offline cannot drift.
 *
 * Coordinates are stored as `Float` and read as `Double`. That split is deliberate and
 * load-bearing for cross-language agreement — see [FrameBuffer].
 */
interface PoseFrames {
    val size: Int

    fun timestampMs(frame: Int): Long

    /** Flat `33 * 3` row of world (metric, hip-centered) coordinates for one frame. */
    fun worldRow(frame: Int): FloatArray

    /** Flat `33 * 3` row of normalized image coordinates for one frame. */
    fun normRow(frame: Int): FloatArray

    fun visibility(frame: Int, landmark: Int): Float

    fun presence(frame: Int, landmark: Int): Float

    /**
     * Whether this frame carries a pose, by the whole-row NaN check.
     *
     * `landmark_rows` writes a full 33x3 NaN row when nothing was detected, so a NaN
     * landmark-0 x implies the whole row is NaN.
     *
     * **This is not a visibility test.** MediaPipe *extrapolates* occluded landmarks rather
     * than dropping them, and those frames pass here while carrying invented coordinates.
     * Gating on what a metric actually reads is [Quality]'s job. Do not "fix" this to mean
     * something stronger — several metrics depend on the two being different.
     */
    fun posePresent(frame: Int): Boolean = !worldRow(frame)[0].isNaN()

    /** Timestamps as doubles, which is what the resampler wants. */
    fun timestampsMs(start: Int = 0, stop: Int = size): DoubleArray =
        DoubleArray((stop - start).coerceAtLeast(0)) { timestampMs(start + it).toDouble() }
}

/**
 * A plain array-backed [PoseFrames]: whole frames held in memory, oldest first.
 *
 * Used by the offline path and by the tests. **Coordinates are `Float`** because that is what
 * the `.h5` stores and what `LiveWindow` buffers, and because the Python metrics widen a
 * float32 array to float64 to compute. Holding doubles end to end here would be *more*
 * precise and therefore *wrong* — it would disagree with the reference implementation in the
 * last digits of every number, which is exactly the drift the goldens exist to catch.
 */
class FrameBuffer(
    private val timestamps: LongArray,
    private val world: Array<FloatArray>,
    private val norm: Array<FloatArray>,
    private val visibility: Array<FloatArray>,
    private val presence: Array<FloatArray>,
) : PoseFrames {

    override val size: Int get() = timestamps.size

    override fun timestampMs(frame: Int): Long = timestamps[frame]

    override fun worldRow(frame: Int): FloatArray = world[frame]

    override fun normRow(frame: Int): FloatArray = norm[frame]

    override fun visibility(frame: Int, landmark: Int): Float = visibility[frame][landmark]

    override fun presence(frame: Int, landmark: Int): Float = presence[frame][landmark]
}

/**
 * A `[start, stop)` frame span.
 *
 * The Python `Span` exists so a caller with a frame range but no annotation has something
 * honest to pass, instead of fabricating an `Annotation` no human ever marked. Same here —
 * and on Android *every* span is of that kind, because there is no annotator live.
 */
data class Span(val start: Int, val stop: Int) {
    val nFrames: Int get() = stop - start
}

/** One frame's world landmarks as an `(x, y, z)` triple, widened to double. */
internal fun FloatArray.point(landmark: Int): DoubleArray = doubleArrayOf(
    this[landmark * 3].toDouble(),
    this[landmark * 3 + 1].toDouble(),
    this[landmark * 3 + 2].toDouble(),
)
