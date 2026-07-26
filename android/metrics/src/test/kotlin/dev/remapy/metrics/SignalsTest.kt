package dev.remapy.metrics

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue

/** Postural primitives, against `motor_metrics/signals.py`. */
class SignalsTest {

    private val case = Goldens.group("signals").asJsonObject

    private fun frames(): PoseFrames {
        val world = case["world"].points()
        val norm = case["norm"].points()
        val ones = Array(world.size) { FloatArray(Landmarks.COUNT) { 1f } }
        val ts = LongArray(world.size) { (it * 1000L / 30) }
        return FrameBuffer(ts, world, norm, ones, ones)
    }

    private fun span(): Span = Span(0, case["world"].asJsonObject["n"].asInt)

    @Test
    fun `trunk vector matches`() {
        assertClose(case["trunk_vector"].matrix(), Signals.trunkVectors(frames(), span()), 1e-9, "trunkVector")
    }

    @Test
    fun `trunk from vertical matches for both a level and a tilted up`() {
        assertClose(
            case["trunk_from_vertical_world_up"].doubles(),
            Signals.trunkFromVertical(frames(), span()),
            1e-9,
            "trunkFromVertical(world_up)",
        )
        assertClose(
            case["trunk_from_vertical_tilted"].doubles(),
            Signals.trunkFromVertical(frames(), span(), case["tilted_up"].doubles()),
            1e-9,
            "trunkFromVertical(tilted)",
        )
    }

    @Test
    fun `horizontal projection matches, including the rolled-camera fallback basis`() {
        val trunk = Signals.trunkVectors(frames(), span())
        assertClose(
            case["project_horizontal_world_up"].matrix(),
            Signals.projectHorizontal(trunk),
            1e-9,
            "projectHorizontal(world_up)",
        )
        assertClose(
            case["project_horizontal_tilted"].matrix(),
            Signals.projectHorizontal(trunk, case["tilted_up"].doubles()),
            1e-9,
            "projectHorizontal(tilted)",
        )
        // `up` parallel to world-x: the world-x rejection degenerates and the basis has to fall
        // back to world-z rather than blowing up. A camera rolled ~90 degrees is a real way to
        // prop a phone, so this branch is not hypothetical here the way it was on a tripod.
        assertClose(
            case["project_horizontal_rolled"].matrix(),
            Signals.projectHorizontal(trunk, case["rolled_up"].doubles()),
            1e-9,
            "projectHorizontal(rolled)",
        )
    }

    @Test
    fun `com norm matches and reads image fractions, not metres`() {
        val expected = case["com_norm"].matrix()
        val actual = Signals.comNorm(frames(), span())
        // The Python export carries all three columns; only the first two are image fractions,
        // and the port drops the third on purpose — MediaPipe's relative depth is on a
        // different and much weaker scale, and mixing it in silently corrupts a distance.
        for (i in 0 until actual.rows) {
            assertClose(expected[i, 0], actual[i, 0], 1e-9, "comNorm[$i,0]")
            assertClose(expected[i, 1], actual[i, 1], 1e-9, "comNorm[$i,1]")
        }
    }

    /**
     * The regression that keeps the package honest about its own frame.
     *
     * MediaPipe's world origin *is* the mid-hip, so a "centre of mass" proxy taken there is
     * identically zero on every frame and sway measured on it is floating-point noise. The
     * real signal is the trunk over the pelvis.
     */
    @Test
    fun `world mid-hip is identically zero, so it is not a sway signal`() {
        val f = frames()
        for (i in 0 until f.size) {
            val hip = Signals.mid(f.worldRow(i), Landmarks.LEFT_HIP, Landmarks.RIGHT_HIP)
            for (c in 0 until 3) {
                assertTrue(abs(hip[c]) < 1e-6, "mid-hip[$i,$c] = ${hip[c]}, expected ~0")
            }
        }
    }

    /** For a level camera the projection is exactly `(world x, world z)`. */
    @Test
    fun `a level up gives ML equals world x and AP equals world z`() {
        val points = Matrix.ofRows(
            listOf(
                doubleArrayOf(0.3, -0.9, 0.2),
                doubleArrayOf(-0.1, -0.8, -0.4),
            )
        )
        val out = Signals.projectHorizontal(points)
        assertClose(0.3, out[0, 0], 1e-12, "ML")
        assertClose(0.2, out[0, 1], 1e-12, "AP")
        assertClose(-0.1, out[1, 0], 1e-12, "ML")
        assertClose(-0.4, out[1, 1], 1e-12, "AP")
    }

    /** Lean is unsigned by construction — it inherits `angleBetween`'s semantics. */
    @Test
    fun `trunk lean is unsigned, so a left and a right lean read the same`() {
        val ones = Array(2) { FloatArray(Landmarks.COUNT) { 1f } }
        val world = Array(2) { FloatArray(Landmarks.COUNT * 3) }
        // Frame 0 leans +x, frame 1 leans -x, both by the same amount.
        for ((i, sign) in listOf(1.0, -1.0).withIndex()) {
            world[i][Landmarks.LEFT_SHOULDER * 3] = (sign * 0.1 + 0.125).toFloat()
            world[i][Landmarks.LEFT_SHOULDER * 3 + 1] = -0.35f
            world[i][Landmarks.RIGHT_SHOULDER * 3] = (sign * 0.1 - 0.125).toFloat()
            world[i][Landmarks.RIGHT_SHOULDER * 3 + 1] = -0.35f
            world[i][Landmarks.LEFT_HIP * 3] = 0.09f
            world[i][Landmarks.RIGHT_HIP * 3] = -0.09f
        }
        val f = FrameBuffer(longArrayOf(0, 33), world, world, ones, ones)
        val angles = Signals.trunkFromVertical(f, Span(0, 2))
        assertClose(angles[0], angles[1], 1e-12, "left and right lean must be indistinguishable")
        assertTrue(angles[0] > 0.0, "a real lean must not read as upright")
    }
}
