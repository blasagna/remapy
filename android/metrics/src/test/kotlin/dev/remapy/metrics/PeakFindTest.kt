package dev.remapy.metrics

import kotlin.math.PI
import kotlin.math.sin
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The prominence peak finder, against exported `scipy.signal.find_peaks` output.
 *
 * Risk #3 of the port. "Local maximum standing far enough above the surrounding troughs" has
 * several reasonable readings, and a plausible reimplementation can agree on a clean sine
 * while disagreeing on the plateaus and repeated maxima real landmark signals are full of.
 */
class PeakFindTest {

    @Test
    fun `find peaks matches scipy across plateaus, monotone runs and noise`() {
        for ((name, case) in Goldens.casesByName("find_peaks")) {
            val signal = case["signal"].doubles()
            val expected = case["expected"].ints()
            val actual = PeakFind.findPeaks(signal, case["prominence"].num())
            assertContentEquals(expected, actual, "find_peaks[$name]")
        }
    }

    @Test
    fun `endpoints are never peaks`() {
        // A strict maximum at each end, with a smaller interior one between them.
        val x = doubleArrayOf(5.0, 1.0, 4.0, 1.0, 5.0)
        assertContentEquals(intArrayOf(2), PeakFind.findPeaks(x, 0.5))
    }

    @Test
    fun `a flat plateau is one peak, reported at its midpoint`() {
        val x = doubleArrayOf(0.0, 1.0, 2.0, 2.0, 2.0, 1.0, 0.0)
        assertContentEquals(intArrayOf(3), PeakFind.findPeaks(x, 0.5))
    }

    /**
     * Prominence is measured against the higher of the two flanking bases, not the global
     * minimum — otherwise a small ripple on a tall hillside would inherit the hill's height.
     */
    @Test
    fun `prominence uses the higher flanking base`() {
        //          0    1    2    3    4    5    6
        val x = doubleArrayOf(0.0, 8.0, 6.0, 7.0, 2.0, 9.0, 0.0)
        // The peak at index 3 sits between a trough at 6.0 (left) and one at 2.0 (right);
        // the base is the higher of those, so the prominence is 7.0 - 6.0 = 1.0.
        assertClose(1.0, PeakFind.prominence(x, 3), 1e-12)
    }

    @Test
    fun `a flat signal has no peaks`() {
        assertEquals(0, PeakFind.findPeaks(DoubleArray(20), 0.0).size)
        assertEquals(0, PeakFind.findPeaks(DoubleArray(20), -1.0).size)
    }

    @Test
    fun `signals shorter than three samples have no peaks`() {
        assertEquals(0, PeakFind.findPeaks(DoubleArray(0), 0.1).size)
        assertEquals(0, PeakFind.findPeaks(doubleArrayOf(1.0), 0.1).size)
        assertEquals(0, PeakFind.findPeaks(doubleArrayOf(0.0, 1.0), 0.1).size)
    }

    /** A sine at a known frequency has a known number of maxima. */
    @Test
    fun `a sine yields one peak per period`() {
        val fs = 30.0
        val freq = 1.0
        val n = 300 // 10 seconds
        val x = DoubleArray(n) { sin(2 * PI * freq * it / fs) }
        val peaks = PeakFind.findPeaks(x, 0.5)
        assertEquals(10, peaks.size, "expected one peak per period over 10 s at 1 Hz")
        // ...and they must be about one period apart.
        for (i in 1 until peaks.size) {
            val periodS = (peaks[i] - peaks[i - 1]) / fs
            assertTrue(kotlin.math.abs(periodS - 1.0) < 0.05, "period $periodS not ~1 s")
        }
    }
}
