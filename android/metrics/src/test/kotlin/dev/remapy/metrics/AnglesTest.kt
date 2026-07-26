package dev.remapy.metrics

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/** Joint angles, against `pose_estimation/angles.py`. */
class AnglesTest {

    @Test
    fun `angle between matches Python, degenerate cases included`() {
        for ((name, case) in Goldens.casesByName("angle_between")) {
            val actual = Angles.angleBetween(
                case["a"].doubles(), case["joint"].doubles(), case["c"].doubles(),
            )
            assertClose(case["expected"].num(), actual, 1e-9, "angleBetween[$name]")
        }
    }

    @Test
    fun `known angles come out in degrees`() {
        val origin = doubleArrayOf(0.0, 0.0, 0.0)
        assertClose(90.0, Angles.angleBetween(doubleArrayOf(1.0, 0.0, 0.0), origin, doubleArrayOf(0.0, 1.0, 0.0)), 1e-12)
        assertClose(180.0, Angles.angleBetween(doubleArrayOf(1.0, 0.0, 0.0), origin, doubleArrayOf(-1.0, 0.0, 0.0)), 1e-12)
        assertClose(0.0, Angles.angleBetween(doubleArrayOf(1.0, 0.0, 0.0), origin, doubleArrayOf(2.0, 0.0, 0.0)), 1e-12)
        assertClose(45.0, Angles.angleBetween(doubleArrayOf(1.0, 0.0, 0.0), origin, doubleArrayOf(1.0, 1.0, 0.0)), 1e-12)
    }

    /** A zero-length bone has no angle — NaN, not an exception and not 0. */
    @Test
    fun `a coincident landmark gives NaN`() {
        val origin = doubleArrayOf(0.0, 0.0, 0.0)
        val angle = Angles.angleBetween(origin, origin, doubleArrayOf(1.0, 0.0, 0.0))
        assertTrue(angle.isNaN(), "expected NaN, got $angle")
    }

    /** Rounding must not push the cosine outside [-1, 1] and produce NaN from acos. */
    @Test
    fun `nearly collinear vectors do not fall out of the acos domain`() {
        val origin = doubleArrayOf(0.0, 0.0, 0.0)
        val a = doubleArrayOf(1e-8, 0.0, 0.0)
        val c = doubleArrayOf(3e-8, 0.0, 0.0)
        assertClose(0.0, Angles.angleBetween(a, origin, c), 1e-9)
    }

    @Test
    fun `joint angles cover all eight joints`() {
        val row = FloatArray(Landmarks.COUNT * 3)
        // A right angle at the left elbow: shoulder above it, wrist beside it.
        row[Landmarks.LEFT_SHOULDER * 3 + 1] = -1f
        row[Landmarks.LEFT_ELBOW * 3 + 1] = 0f
        row[Landmarks.LEFT_WRIST * 3] = 1f
        val angles = Angles.jointAngles(row)
        assertEquals(8, angles.size)
        assertEquals(Landmarks.JOINT_TRIPLETS.keys, angles.keys)
        assertClose(90.0, angles.getValue("left_elbow"), 1e-9, "left_elbow")
    }
}
