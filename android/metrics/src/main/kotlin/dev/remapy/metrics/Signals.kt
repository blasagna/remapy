package dev.remapy.metrics

import kotlin.math.sqrt

/**
 * Postural primitives derived from pose *world* landmarks.
 *
 * Port of `motor_metrics/signals.py`, whose docstring carries the reasoning. Two facts from
 * it run through everything downstream and are worth restating where the code is:
 *
 * **The frame is hip-centered.** MediaPipe puts the world origin *at* the midpoint of the
 * hips. So (1) a mid-hip "centre of mass" proxy is identically zero on every frame and sway
 * measured on it is floating-point noise — the signal that carries postural information is
 * [trunkVector], the mid-shoulder relative to the pelvis, and the hip-centered origin is what
 * makes it scale- and calibration-free; (2) translation across the floor is **not
 * recoverable**, and the only whole-body translation signal is [comNorm], in image fractions.
 *
 * **There is no gravity vector in a camera-only recording.** Axes are camera-relative, so
 * [WORLD_UP] is vertical **only if the camera is level** — a real risk on a hand-held or
 * propped phone, and the reason the live hold readout leads with the trunk angle's delta from
 * its own window baseline rather than an absolute lean. Every consumer records which vertical
 * it used, so a tilted session can be found later rather than quietly biasing a year of numbers.
 */
object Signals {

    /** Up in the camera frame: MediaPipe world coords are y-DOWN, so up is -y. */
    val WORLD_UP = doubleArrayOf(0.0, -1.0, 0.0)

    private const val EPS = 1e-9

    /** Midpoint of landmarks [i] and [j] for one frame. */
    fun mid(worldRow: FloatArray, i: Int, j: Int): DoubleArray {
        val a = worldRow.point(i)
        val b = worldRow.point(j)
        return DoubleArray(3) { (a[it] + b[it]) / 2.0 }
    }

    /**
     * Pelvis -> mid-shoulder, in meters. The package's core postural signal.
     *
     * Used two ways: as a **position** whose excursion is sway, and as a **direction** whose
     * angle from vertical is lean. The `- midHip` term is very nearly a no-op since mid-hip
     * *is* the origin; it is written anyway because it costs nothing and makes the signal's
     * meaning legible without knowing MediaPipe's convention.
     */
    fun trunkVector(worldRow: FloatArray): DoubleArray {
        val shoulders = mid(worldRow, Landmarks.LEFT_SHOULDER, Landmarks.RIGHT_SHOULDER)
        val hips = mid(worldRow, Landmarks.LEFT_HIP, Landmarks.RIGHT_HIP)
        return DoubleArray(3) { shoulders[it] - hips[it] }
    }

    /** [trunkVector] over a span of frames, as an `(N, 3)` matrix. */
    fun trunkVectors(frames: PoseFrames, span: Span): Matrix {
        val out = Matrix(span.nFrames, 3)
        for (i in 0 until span.nFrames) {
            val t = trunkVector(frames.worldRow(span.start + i))
            for (c in 0 until 3) out[i, c] = t[c]
        }
        return out
    }

    /**
     * Angle in degrees between the trunk and [up]; 0 = upright.
     *
     * Reuses [Angles.angleBetween] so there is one definition of "angle" in the codebase, and
     * inherits its **unsigned** semantics: this cannot tell a forward lean from a backward or
     * a lateral one. Use [projectHorizontal] when the direction of lean matters. NaN frames
     * propagate to NaN.
     */
    fun trunkFromVertical(frames: PoseFrames, span: Span, up: DoubleArray = WORLD_UP): DoubleArray {
        val upUnit = unit(up)
        val origin = doubleArrayOf(0.0, 0.0, 0.0)
        return DoubleArray(span.nFrames) {
            Angles.angleBetween(trunkVector(frames.worldRow(span.start + it)), origin, upUnit)
        }
    }

    /**
     * Project `(N, 3)` points onto the plane perpendicular to [up], giving `(ML, AP)`.
     *
     * **The axis order is the point of the function.** On a single camera, ML lies in the
     * image plane and is measured well, while AP is MediaPipe's inferred depth and is
     * markedly noisier. A caller that collapses these into one isotropic sway number averages
     * a good estimate with a bad one and cannot tell afterwards which it was looking at.
     *
     * For a level camera this is exactly `ML = world x`, `AP = world z`. For a tilted [up] the
     * basis is the closest thing to that which is still orthonormal.
     */
    fun projectHorizontal(points: Matrix, up: DoubleArray = WORLD_UP): Matrix {
        val upUnit = unit(up)
        // ML is world-x with any `up` component removed, so it stays the image-plane axis.
        var ml = reject(doubleArrayOf(1.0, 0.0, 0.0), upUnit)
        if (norm(ml) < 1e-6) {
            // Degenerate: `up` is (nearly) world-x, i.e. the camera is rolled ~90 degrees.
            // Fall back to world-z so the basis stays defined rather than blowing up.
            ml = reject(doubleArrayOf(0.0, 0.0, 1.0), upUnit)
        }
        ml = unit(ml)
        val ap = unit(cross(upUnit, ml))

        val out = Matrix(points.rows, 2)
        for (i in 0 until points.rows) {
            var a = 0.0
            var b = 0.0
            for (c in 0 until 3) {
                a += points[i, c] * ml[c]
                b += points[i, c] * ap[c]
            }
            out[i, 0] = a
            out[i, 1] = b
        }
        return out
    }

    /**
     * Mid-hip in **normalized image coordinates**, not meters; columns 0 and 1 only.
     *
     * The only whole-body translation signal a recording has, since `landmarks_world` is
     * hip-centered and the pelvis cannot move in it by construction. Column 2 of the source is
     * MediaPipe's relative depth, on a different and much weaker scale, and is deliberately
     * dropped here rather than mixed in.
     *
     * Any speed derived from this is in **image widths per second**, and it is never called
     * metres. Converting it honestly needs a scale reference this pipeline does not have.
     */
    fun comNorm(frames: PoseFrames, span: Span): Matrix {
        val out = Matrix(span.nFrames, 2)
        for (i in 0 until span.nFrames) {
            val m = mid(frames.normRow(span.start + i), Landmarks.LEFT_HIP, Landmarks.RIGHT_HIP)
            out[i, 0] = m[0]
            out[i, 1] = m[1]
        }
        return out
    }

    internal fun norm(v: DoubleArray): Double {
        var acc = 0.0
        for (x in v) acc += x * x
        return sqrt(acc)
    }

    internal fun unit(v: DoubleArray): DoubleArray {
        val n = norm(v)
        return if (n > EPS) DoubleArray(v.size) { v[it] / n } else v.copyOf()
    }

    /** Component of [v] perpendicular to the unit vector [axis]. */
    private fun reject(v: DoubleArray, axis: DoubleArray): DoubleArray {
        var dot = 0.0
        for (i in v.indices) dot += v[i] * axis[i]
        return DoubleArray(v.size) { v[it] - dot * axis[it] }
    }

    private fun cross(a: DoubleArray, b: DoubleArray): DoubleArray = doubleArrayOf(
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )
}
