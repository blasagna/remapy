package dev.remapy.metrics

/**
 * The one piece of `scipy.signal` the live crawl path needs: prominence-gated peak finding.
 *
 * Reimplemented rather than approximated. "Local maximum standing far enough above the
 * surrounding troughs" has several reasonable readings and scipy picks specific ones, so a
 * plausible-looking reimplementation can agree on a clean sine and disagree on real footage —
 * which is where cadence actually gets measured. The two rules that matter:
 *
 * - **Plateaus count once, at their midpoint.** A flat-topped maximum is one peak, not none
 *   (strict `>` on both sides) and not several. Endpoints are never peaks.
 * - **Prominence walks outward to the nearest strictly-higher sample**, on each side, and
 *   takes the *higher* of the two intervening minima as the base. Not the neighbouring
 *   trough, and not the global minimum.
 *
 * `PeakFindTest` pins both against exported `scipy.signal.find_peaks` output, over plateaus,
 * monotone runs, repeated maxima and noise.
 *
 * `scipy.signal.hilbert` is deliberately **not** ported — see [Crawl].
 */
object PeakFind {

    /**
     * Indices of local maxima in [x] whose prominence is at least [minProminence].
     *
     * Mirrors `find_peaks(x, prominence=minProminence)` with `wlen=None`, which is the only
     * form this kernel calls.
     */
    fun findPeaks(x: DoubleArray, minProminence: Double): IntArray {
        val candidates = localMaxima(x)
        val kept = ArrayList<Int>(candidates.size)
        for (peak in candidates) {
            if (prominence(x, peak) >= minProminence) kept.add(peak)
        }
        return kept.toIntArray()
    }

    /**
     * Local maxima, with flat-topped ones reported at the midpoint of their plateau.
     *
     * Index 0 and the last index can never be maxima: scipy scans `1 until n - 1`, and the
     * plateau lookahead is bounded by the same limit, so a plateau running into the end of
     * the signal is not a peak either.
     */
    fun localMaxima(x: DoubleArray): IntArray {
        val peaks = ArrayList<Int>()
        val last = x.size - 1
        var i = 1
        while (i < last) {
            if (x[i - 1] < x[i]) {
                var ahead = i + 1
                while (ahead < last && x[ahead] == x[i]) ahead++
                if (x[ahead] < x[i]) {
                    peaks.add((i + ahead - 1) / 2)
                    i = ahead
                }
            }
            i++
        }
        return peaks.toIntArray()
    }

    /**
     * Topographic prominence of the peak at [peak].
     *
     * Walk left while samples stay at or below the peak, remembering the lowest; do the same
     * to the right; the base is the *higher* of the two minima. NaN stops a walk, because
     * every comparison against it is false — the same behaviour scipy has, and harmless in
     * practice because [Crawl.cycles] rejects non-finite signals before getting here.
     */
    fun prominence(x: DoubleArray, peak: Int): Double {
        val height = x[peak]

        var leftMin = height
        var i = peak
        while (i >= 0 && x[i] <= height) {
            if (x[i] < leftMin) leftMin = x[i]
            i--
        }

        var rightMin = height
        i = peak
        while (i < x.size && x[i] <= height) {
            if (x[i] < rightMin) rightMin = x[i]
            i++
        }

        return height - maxOf(leftMin, rightMin)
    }
}
