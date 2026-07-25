package dev.remapy.metrics

import kotlin.math.PI
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Static holds, against `motor_metrics/hold.py`.
 *
 * The golden cases cover the trial shapes a real session produces — a clean sway, a mid-trial
 * dropout, an untrusted stretch, a two-frame mis-mark, a tilted camera. The closed forms below
 * are pinned to analytic values rather than to recorded outputs, which is what
 * `tests/test_motor_metrics.py` does and for the same reason: a recorded output pins whatever
 * the pipeline did that day, including its bugs.
 */
class HoldTest {

    @Test
    fun `hold metrics match Python across every trial shape`() {
        for ((name, case) in Goldens.casesByName("hold_metrics")) {
            val frames = case["rec"].asJsonObject.asFrames()
            val seg = Span(case["start"].asInt, case["stop"].asInt)
            val up = if (case.has("up")) case["up"].doubles() else Signals.WORLD_UP
            val windowS = case["window_s"].let { if (it.isJsonNull) null else it.num() }
            val actual = Hold.holdMetrics(frames, seg, up, windowS = windowS)
            assertMetricsMatch(
                case["expected"].asJsonObject, actual.toGoldenMap(), 1e-9, label = "hold[$name]",
            )
        }
    }

    /**
     * A sinusoidal sway of amplitude A has RMS `A / sqrt(2)`.
     *
     * The one closed form that says the sway number *means* what it claims. At 0.4 Hz the
     * filter's documented gain is ~0.99, so the tolerance is loose enough to absorb that
     * rolloff and tight enough to catch a factor-of-two or a missing centring.
     */
    @Test
    fun `medio-lateral sway RMS is amplitude over root two`() {
        val amplitude = 0.03
        val frames = Bodies.frames(Bodies.swayingTrunk(300, amplitude, 0.4))
        val m = Hold.holdMetrics(frames, Span(0, 300))
        assertClose(amplitude / sqrt(2.0), m.swayMlRmsM, 2e-2, "ML RMS")
        // The trunk does not move in depth at all, so AP must be ~0 — and the split must not
        // quietly average the well-measured axis with the noisy inferred one.
        assertTrue(m.swayApRmsM < 1e-6, "AP RMS was ${m.swayApRmsM}, expected ~0")
        assertClose(amplitude / sqrt(2.0), m.rmsM, 2e-2, "radial RMS")
    }

    /** `chi2(2, 0.95) * pi * sqrt(l1 * l2)` — exact for a circle of known radius. */
    @Test
    fun `sway ellipse area is the posturography formula`() {
        val n = 720
        val r = 0.05
        val points = Matrix(n, 2)
        for (i in 0 until n) {
            val theta = 2 * PI * i / n
            points[i, 0] = r * kotlin.math.cos(theta)
            points[i, 1] = r * kotlin.math.sin(theta)
        }
        // A uniform circle has covariance diag(r^2/2, r^2/2), so sqrt(l1*l2) = r^2/2.
        assertClose(5.991 * PI * r * r / 2.0, Hold.swayEllipseArea(points), 1e-4, "ellipse area")
    }

    /**
     * ...and it reads ~0 for one-axis rocking, which is why it must never be read alone.
     *
     * A trunk rocking hard on a single axis traces a line, and a line encloses no area.
     */
    @Test
    fun `ellipse area is about zero for one-axis rocking however hard it rocks`() {
        val n = 300
        val points = Matrix(n, 2)
        for (i in 0 until n) {
            points[i, 0] = 0.2 * kotlin.math.sin(2 * PI * i / 60.0)
            points[i, 1] = 0.0
        }
        assertTrue(Hold.swayEllipseArea(points) < 1e-9, "a line must enclose no area")
    }

    @Test
    fun `path length is the polygon perimeter`() {
        val square = Matrix.ofRows(
            listOf(
                doubleArrayOf(0.0, 0.0),
                doubleArrayOf(3.0, 0.0),
                doubleArrayOf(3.0, 4.0),
                doubleArrayOf(0.0, 4.0),
                doubleArrayOf(0.0, 0.0),
            )
        )
        assertClose(3.0 + 4.0 + 3.0 + 4.0, pathLength(square), 1e-12)
    }

    /**
     * `pathLengthM` is duration-confounded — the trap the module docstring calls out.
     *
     * A longer trial at the *same* steadiness accumulates more path, so comparing two trials on
     * it compares their lengths. `meanVelocityMps` is the duration-free form and must agree
     * between the two.
     */
    @Test
    fun `path length grows with duration while mean velocity does not`() {
        val short = Bodies.frames(Bodies.swayingTrunk(150, 0.03, 0.4))
        val long = Bodies.frames(Bodies.swayingTrunk(450, 0.03, 0.4))
        val a = Hold.holdMetrics(short, Span(0, 150))
        val b = Hold.holdMetrics(long, Span(0, 450))
        assertTrue(b.pathLengthM > 2 * a.pathLengthM, "path length must scale with duration")
        assertClose(a.meanVelocityMps, b.meanVelocityMps, 5e-2, "mean velocity must not")
    }

    /** ...and `windowS` is the escape hatch: truncate both to a common prefix. */
    @Test
    fun `a common window makes unequal trials comparable on path length`() {
        val short = Bodies.frames(Bodies.swayingTrunk(150, 0.03, 0.4))
        val long = Bodies.frames(Bodies.swayingTrunk(450, 0.03, 0.4))
        val a = Hold.holdMetrics(short, Span(0, 150), windowS = 4.0)
        val b = Hold.holdMetrics(long, Span(0, 450), windowS = 4.0)
        assertClose(a.pathLengthM, b.pathLengthM, 1e-9, "windowed path length")
        assertClose(a.trackedS, b.trackedS, 1e-9, "windowed tracked_s")
    }

    /** Sway is measured over the longest good run, never stitched across a dropout. */
    @Test
    fun `a mid-trial dropout shortens tracked_s rather than bridging the gap`() {
        val n = 300
        val trunk = Bodies.swayingTrunk(n, 0.03, 0.4).toMutableList()
        val frames = Bodies.frames(trunk)
        // Blank out frames 100..149 the way `landmark_rows` does: whole-row NaN.
        for (i in 100 until 150) frames.worldRow(i).fill(Float.NaN)
        val m = Hold.holdMetrics(frames, Span(0, n))
        assertClose(1.0 - 50.0 / n, m.coverage, 1e-9, "coverage")
        // The later run (150..299) is the longest, so tracked_s is its length, not the trial's.
        assertClose((n - 150 - 1) / Derive.FS, m.trackedS, 5e-2, "tracked_s")
        assertTrue(m.durationS > m.trackedS, "the marked trial is longer than the measured run")
    }

    @Test
    fun `degenerate segments return NaN measurements rather than throwing`() {
        val frames = Bodies.frames(Bodies.swayingTrunk(10, 0.03, 0.4))
        for (seg in listOf(Span(0, 0), Span(3, 3), Span(0, 1), Span(0, 5))) {
            val m = Hold.holdMetrics(frames, seg)
            assertTrue(m.rmsM.isNaN(), "$seg: rms must be NaN, was ${m.rmsM}")
            assertTrue(m.pathLengthM.isNaN(), "$seg: path length must be NaN")
        }
    }

    @Test
    fun `a fully untrusted trial has zero coverage and NaN sway`() {
        val frames = Bodies.frames(Bodies.swayingTrunk(200, 0.03, 0.4), visibility = 0.2f)
        val m = Hold.holdMetrics(frames, Span(0, 200))
        assertClose(0.0, m.coverage, 0.0, "coverage")
        assertTrue(m.rmsM.isNaN(), "rms must be NaN when nothing is trusted")
        assertClose(0.0, m.trackedS, 0.0, "tracked_s")
    }

    /**
     * `upSource` is on every readout so a tilted session can be found later.
     *
     * On a phone this is not a formality — the camera is level only if someone propped it that
     * way, and `WORLD_UP` silently assumes it is.
     */
    @Test
    fun `up source records which vertical was used`() {
        val frames = Bodies.frames(Bodies.swayingTrunk(200, 0.03, 0.4))
        assertEqualsString("world_y", Hold.holdMetrics(frames, Span(0, 200)).upSource)
        val tilted = doubleArrayOf(0.15, -0.98, 0.05)
        assertEqualsString("custom", Hold.holdMetrics(frames, Span(0, 200), up = tilted).upSource)
    }

    private fun assertEqualsString(expected: String, actual: String) =
        kotlin.test.assertEquals(expected, actual)
}
