package dev.remapy.metrics

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Belly-crawl metrics, against `motor_metrics/crawl.py`.
 *
 * The golden comparison skips the Hilbert-derived reciprocity fields, which this port does not
 * compute — see [HILBERT_FIELDS]. Everything else, both girdles included, must match.
 */
class CrawlTest {

    /** Limbs oscillating along the body axis; the trunk is fixed, as it is in prone. */
    private fun crawler(
        n: Int,
        cadenceHz: Double = 0.9,
        leftArmAmp: Double = 0.09,
        rightArmAmp: Double = 0.085,
        leftLegAmp: Double = 0.07,
        rightLegAmp: Double = 0.035,
    ): FrameBuffer {
        val trunk = List(n) { doubleArrayOf(0.0, -0.30, 0.0) }
        fun limb(amp: Double, phase: Double, lateral: Double) = List(n) { i ->
            val along = 0.20 + amp * sin(2 * PI * cadenceHz * i / Derive.FS + phase)
            doubleArrayOf(lateral, -along, 0.0)
        }
        return Bodies.frames(
            trunk,
            leftWrist = limb(leftArmAmp, 0.0, 0.12),
            rightWrist = limb(rightArmAmp, PI, -0.12),
            leftKnee = limb(leftLegAmp, 0.4, 0.08),
            rightKnee = limb(rightLegAmp, PI + 0.4, -0.08),
        )
    }

    @Test
    fun `crawl metrics match Python for both girdles`() {
        for ((name, case) in Goldens.casesByName("crawl_metrics")) {
            val frames = case["rec"].asJsonObject.asFrames()
            val seg = Span(case["start"].asInt, case["stop"].asInt)
            val actual = Crawl.crawlMetrics(frames, seg)
            assertMetricsMatch(
                case["expected"].asJsonObject,
                actual.toGoldenMap(),
                1e-9,
                skip = HILBERT_FIELDS,
                label = "crawl[$name]",
            )
        }
    }

    @Test
    fun `limb signal matches Python for both wrists and both knees`() {
        val case = Goldens.group("limb_signal").asJsonObject
        val frames = case["rec"].asJsonObject.asFrames()
        val span = Span(case["start"].asInt, case["stop"].asInt)
        val expected = mapOf(
            "left_wrist" to (Crawl.Side.LEFT to Crawl.Marker.WRIST),
            "right_wrist" to (Crawl.Side.RIGHT to Crawl.Marker.WRIST),
            "left_knee" to (Crawl.Side.LEFT to Crawl.Marker.KNEE),
            "right_knee" to (Crawl.Side.RIGHT to Crawl.Marker.KNEE),
        )
        for ((key, spec) in expected) {
            val (side, marker) = spec
            assertClose(case[key].doubles(), Crawl.limbSignal(frames, span, side, marker), 1e-9, key)
        }
    }

    /**
     * The limb signal is the limb's travel along the **trunk axis**, not along gravity.
     *
     * In prone there is no useful vertical, which is what makes crawl the camera-robust mode —
     * it reads no `up` at all, so a phone propped at any angle measures the same cadence.
     */
    @Test
    fun `limb signal recovers travel along the body axis`() {
        val frames = crawler(120)
        val signal = Crawl.limbSignal(frames, Span(0, 120), Crawl.Side.LEFT, Crawl.Marker.WRIST)
        for (i in 0 until 120) {
            val expected = 0.20 + 0.09 * sin(2 * PI * 0.9 * i / Derive.FS)
            assertClose(expected, signal[i], 1e-6, "limbSignal[$i]")
        }
    }

    /** Cadence is the driving frequency, in cycles per minute. */
    @Test
    fun `cadence recovers the driving frequency`() {
        for (cadenceHz in listOf(0.6, 0.9, 1.2)) {
            val n = 600 // 20 s, enough cycles for the count to be insensitive to the edges
            val m = Crawl.crawlMetrics(crawler(n, cadenceHz), Span(0, n))
            assertClose(cadenceHz * 60.0, m.cadenceCpm, 5e-2, "cadence at $cadenceHz Hz")
            assertClose(cadenceHz * 60.0, m.legCadenceCpm, 5e-2, "leg cadence at $cadenceHz Hz")
        }
    }

    /** A metronomic crawl has near-zero period variability. */
    @Test
    fun `a perfectly regular crawl has near-zero cycle period CV`() {
        val m = Crawl.crawlMetrics(crawler(600), Span(0, 600))
        assertTrue(m.cyclePeriodCv < 0.05, "cyclePeriodCv was ${m.cyclePeriodCv}")
    }

    /**
     * "Favours one leg" — the signal that matters most for Remy.
     *
     * A within-window left/right comparison on the excursion ranges, `2 (L - R) / (L + R)`.
     * Distinct from the between-trial symmetry index, which is offline-only; do not conflate
     * them just because they share a formula.
     */
    @Test
    fun `leg amplitude symmetry reports which leg does more of the work`() {
        val even = Crawl.crawlMetrics(crawler(600, leftLegAmp = 0.06, rightLegAmp = 0.06), Span(0, 600))
        assertTrue(abs(even.legAmplitudeSymmetry) < 1e-3, "even legs: ${even.legAmplitudeSymmetry}")

        val favoursLeft = Crawl.crawlMetrics(crawler(600, leftLegAmp = 0.09, rightLegAmp = 0.03), Span(0, 600))
        // 2 * (0.09 - 0.03) / (0.09 + 0.03) = 1.0
        assertClose(1.0, favoursLeft.legAmplitudeSymmetry, 1e-2, "favours left")
        assertTrue(favoursLeft.legAmplitudeSymmetry > 0, "sign must give the side")

        val favoursRight = Crawl.crawlMetrics(crawler(600, leftLegAmp = 0.03, rightLegAmp = 0.09), Span(0, 600))
        assertClose(-1.0, favoursRight.legAmplitudeSymmetry, 1e-2, "favours right")
    }

    /**
     * **The regression that earns [Crawl.MIN_CYCLE_EXCURSION_M] its keep.**
     *
     * The prominence gate alone is relative to the signal's own range, so it normalizes pure
     * jitter up into a textbook crawl: a motionless child reports a confident, entirely
     * fictional cadence. This asserts both halves — that the absolute gate suppresses it, and
     * that removing the gate brings it straight back, so nobody deletes the constant as
     * redundant.
     */
    @Test
    fun `a still child reports no cadence, and removing the excursion gate resurrects it`() {
        val rng = Random(11)
        val jitter = DoubleArray(200) { rng.nextDouble(-0.006, 0.006) }
        assertEquals(0, Crawl.cycles(jitter).size, "jitter must not read as crawl cycles")

        val ungated = Crawl.cycles(jitter, minExcursion = 0.0)
        assertTrue(
            ungated.size > 20,
            "without the absolute gate, jitter should produce a fictional cadence " +
                "(got ${ungated.size} cycles) — if this stops being true the regression is toothless",
        )
    }

    @Test
    fun `a signal just under the excursion floor has no cycles and just over has some`() {
        val n = 300
        fun sine(amp: Double) = DoubleArray(n) { amp * sin(2 * PI * 0.9 * it / Derive.FS) }
        // ptp of a sine of amplitude a is 2a, so the floor sits at a = MIN_CYCLE_EXCURSION_M / 2.
        assertEquals(0, Crawl.cycles(sine(Crawl.MIN_CYCLE_EXCURSION_M / 2 * 0.99)).size)
        assertTrue(Crawl.cycles(sine(Crawl.MIN_CYCLE_EXCURSION_M / 2 * 1.01)).isNotEmpty())
    }

    @Test
    fun `unusable signals yield no cycles rather than throwing`() {
        assertEquals(0, Crawl.cycles(DoubleArray(0)).size)
        assertEquals(0, Crawl.cycles(doubleArrayOf(0.0, 1.0)).size)
        assertEquals(0, Crawl.cycles(DoubleArray(50)).size)
        val withNan = DoubleArray(60) { 0.09 * sin(2 * PI * 0.9 * it / Derive.FS) }
        withNan[5] = Double.NaN
        assertEquals(0, Crawl.cycles(withNan).size, "a NaN anywhere makes the signal unusable")
    }

    /**
     * Girdles gate independently: an occluded wrist must not cost the leg cadence.
     *
     * Legs leave frame in prone far more than arms, and the reverse happens too. If one gate
     * governed both, a single occluded knee would blank a perfectly good arm measurement.
     */
    @Test
    fun `an occluded arm leaves the leg metrics intact`() {
        val n = 600
        val frames = crawler(n)
        // Drop the left wrist's visibility for the middle third of the trial.
        val scores = Array(n) { FloatArray(Landmarks.COUNT) { 1f } }
        for (i in 200 until 400) scores[i][Landmarks.LEFT_WRIST] = 0.1f
        val world = Array(n) { frames.worldRow(it) }
        val gated = FrameBuffer(LongArray(n) { (it * 1000.0 / Derive.FS).toLong() }, world, world, scores, scores)

        val m = Crawl.crawlMetrics(gated, Span(0, n))
        assertTrue(m.coverage < 0.7, "arm coverage should reflect the occlusion, was ${m.coverage}")
        assertClose(1.0, m.legCoverage, 1e-9, "leg coverage must be untouched")
        assertClose(0.9 * 60.0, m.legCadenceCpm, 5e-2, "leg cadence must survive the arm dropout")
    }

    @Test
    fun `degenerate segments return NaN fields rather than throwing`() {
        val frames = crawler(10)
        for (seg in listOf(Span(0, 0), Span(0, 1), Span(0, 4), Span(5, 5))) {
            val m = Crawl.crawlMetrics(frames, seg)
            assertTrue(m.cadenceCpm.isNaN(), "$seg: cadence must be NaN")
            assertEquals(0, m.nCyclesLeft, "$seg: no cycles")
        }
    }

    @Test
    fun `symmetry index is NaN when a side is missing or the medians cancel`() {
        assertTrue(Crawl.symmetryIndex(DoubleArray(0), doubleArrayOf(1.0)).isNaN())
        assertTrue(Crawl.symmetryIndex(doubleArrayOf(1.0), DoubleArray(0)).isNaN())
        assertTrue(Crawl.symmetryIndex(doubleArrayOf(1.0), doubleArrayOf(-1.0)).isNaN())
        assertClose(0.0, Crawl.symmetryIndex(doubleArrayOf(2.0), doubleArrayOf(2.0)), 1e-12)
    }
}
