package dev.remapy.metrics

import kotlin.math.acos
import kotlin.math.sqrt

/**
 * Joint angles from pose *world* landmarks (metric, in meters).
 *
 * Port of `pose_estimation/angles.py`. MediaPipe does not output joint angles, but they are
 * easy to derive: the angle at a joint is the angle between the two bone vectors meeting
 * there.
 *
 * [angleBetween] is the codebase's single definition of "angle" — `Signals.trunkFromVertical`
 * reuses it rather than writing its own, and inherits its **unsigned** semantics along with
 * it. Keep it that way; two definitions of an angle is how a lean starts reading differently
 * in two places.
 */
object Angles {

    /** Angle in degrees at [joint], between the vectors joint->[a] and joint->[c]. */
    fun angleBetween(a: DoubleArray, joint: DoubleArray, c: DoubleArray): Double {
        var dot = 0.0
        var normBa = 0.0
        var normBc = 0.0
        for (i in a.indices) {
            val ba = a[i] - joint[i]
            val bc = c[i] - joint[i]
            dot += ba * bc
            normBa += ba * ba
            normBc += bc * bc
        }
        val denom = sqrt(normBa) * sqrt(normBc)
        if (denom == 0.0) return Double.NaN
        val cos = (dot / denom).coerceIn(-1.0, 1.0)
        return Math.toDegrees(acos(cos))
    }

    /** All [Landmarks.JOINT_TRIPLETS] angles for one frame's world landmarks, in degrees. */
    fun jointAngles(worldRow: FloatArray): Map<String, Double> =
        Landmarks.JOINT_TRIPLETS.mapValues { (_, triplet) ->
            val (a, joint, c) = triplet
            angleBetween(worldRow.point(a), worldRow.point(joint), worldRow.point(c))
        }
}
