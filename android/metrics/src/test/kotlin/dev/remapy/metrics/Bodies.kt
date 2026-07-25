package dev.remapy.metrics

/**
 * Synthetic *anatomy* for the closed-form tests — the Kotlin counterpart of
 * `tests/fakes.py::body_world`.
 *
 * The distinction that helper's docstring draws matters here too: spreading points along a
 * diagonal is fine for a pass-through test but **is not a body**, and every postural metric
 * needs one. Hips sit symmetrically about the origin (mirroring MediaPipe's hip-centered
 * frame) and shoulders symmetrically about `hipCenter + trunk`, separated along world x.
 */
object Bodies {

    /**
     * Build frames from a per-frame trunk vector (pelvis -> mid-shoulder) and optional limbs.
     *
     * Values are stored through `Float`, matching what a detector actually delivers and what
     * [FrameBuffer] holds — see its docstring for why that is not merely a size choice.
     */
    fun frames(
        trunk: List<DoubleArray>,
        timestampsMs: LongArray = LongArray(trunk.size) { (it * 1000.0 / Derive.FS).toLong() },
        shoulderWidth: Double = 0.25,
        hipWidth: Double = 0.18,
        leftWrist: List<DoubleArray>? = null,
        rightWrist: List<DoubleArray>? = null,
        leftKnee: List<DoubleArray>? = null,
        rightKnee: List<DoubleArray>? = null,
        visibility: Float = 1f,
        normOverride: List<DoubleArray>? = null,
    ): FrameBuffer {
        val n = trunk.size
        val world = Array(n) { FloatArray(Landmarks.COUNT * 3) }
        val norm = Array(n) { FloatArray(Landmarks.COUNT * 3) }

        for (i in 0 until n) {
            place(world[i], Landmarks.LEFT_HIP, hipWidth / 2, 0.0, 0.0)
            place(world[i], Landmarks.RIGHT_HIP, -hipWidth / 2, 0.0, 0.0)
            place(
                world[i], Landmarks.LEFT_SHOULDER,
                trunk[i][0] + shoulderWidth / 2, trunk[i][1], trunk[i][2],
            )
            place(
                world[i], Landmarks.RIGHT_SHOULDER,
                trunk[i][0] - shoulderWidth / 2, trunk[i][1], trunk[i][2],
            )
            leftWrist?.let { place(world[i], Landmarks.LEFT_WRIST, it[i][0], it[i][1], it[i][2]) }
            rightWrist?.let { place(world[i], Landmarks.RIGHT_WRIST, it[i][0], it[i][1], it[i][2]) }
            leftKnee?.let { place(world[i], Landmarks.LEFT_KNEE, it[i][0], it[i][1], it[i][2]) }
            rightKnee?.let { place(world[i], Landmarks.RIGHT_KNEE, it[i][0], it[i][1], it[i][2]) }

            if (normOverride != null) {
                place(norm[i], Landmarks.LEFT_HIP, normOverride[i][0], normOverride[i][1], 0.0)
                place(norm[i], Landmarks.RIGHT_HIP, normOverride[i][0], normOverride[i][1], 0.0)
            }
        }

        val scores = Array(n) { FloatArray(Landmarks.COUNT) { visibility } }
        return FrameBuffer(timestampsMs, world, norm, scores, scores)
    }

    /** A trunk of fixed length tipping side to side: pure medio-lateral sway of [amplitude] m. */
    fun swayingTrunk(n: Int, amplitude: Double, freqHz: Double, trunkLength: Double = 0.35): List<DoubleArray> =
        List(n) { i ->
            val t = i / Derive.FS
            doubleArrayOf(amplitude * kotlin.math.sin(2 * Math.PI * freqHz * t), -trunkLength, 0.0)
        }

    private fun place(row: FloatArray, landmark: Int, x: Double, y: Double, z: Double) {
        row[landmark * 3] = x.toFloat()
        row[landmark * 3 + 1] = y.toFloat()
        row[landmark * 3 + 2] = z.toFloat()
    }
}
