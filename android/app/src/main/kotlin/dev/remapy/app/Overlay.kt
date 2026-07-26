package dev.remapy.app

import androidx.compose.ui.graphics.Color
import dev.remapy.metrics.LiveMetrics
import dev.remapy.metrics.LiveMetricsComputer
import kotlin.math.abs

/**
 * The live readout's presentation layer — port of `motor_metrics/live_draw.py`.
 *
 * Kept separate from the drawing code below the way the Python keeps it out of `live.py`: that
 * module stays pure computation and imports no renderer. **The metrics are never recomputed per
 * surface**, so a number here and the same number anywhere else cannot disagree.
 *
 * **Blanking is the point of the layout.** A NaN value renders as `--`, never as a stale number or
 * a zero: coverage below the gate means the tracker has lost the child, and a figure left on
 * screen then reads as a measurement of him. Coverage is drawn first, coloured by whether it
 * clears the gate, so the *reason* for the dashes is always beside them.
 *
 * Unlike the OpenCV original this can use real fonts, so the ASCII-only workarounds
 * (`trunk dev`, `deg`) are no longer forced. They are kept anyway — the two overlays are read
 * side by side when checking the port against the laptop, and matching labels make that possible.
 */
object Overlay {

    /**
     * Display tolerance for the steadiness meter, in degrees.
     *
     * A display/game tolerance, **not a validated threshold**, which is why it lives in the render
     * layer and not in the metrics: changing it cannot change any recorded number.
     */
    const val STEADINESS_TOL_DEG: Double = 10.0

    /** Cyan, matching the joint-angle overlay so the two read as one family. */
    val TEXT_COLOR = Color(0xFF00FFFF)
    val OK_COLOR = Color(0xFF78FF78)
    val WARN_COLOR = Color(0xFFFFC850)
    val BAD_COLOR = Color(0xFFFF5050)

    data class Row(val label: String, val value: String)

    /** The `(label, value)` lines for this readout's mode. */
    fun rows(m: LiveMetrics): List<Row> = if (m.liveMode == LiveMetricsComputer.CRAWL) {
        // Legs lead: Remy's signal is which leg he drives with, not the arms. `leg favor` is the
        // headline, and the per-side cadence shows the same favouring in the cycle counts.
        //
        // Read the favour number on its own scale: it is `symmetry_index`, `2(L-R)/(L+R)` over the
        // two excursion ranges, so it runs to +-2 and **is not a percentage difference**. `0.67` is
        // one leg travelling twice as far as the other; `2.0` is one leg not moving at all.
        listOf(
            Row("arm cad", fmt(m.liveCadenceCpm, 1, " cpm")),
            Row("leg cad", fmt(m.liveLegCadenceCpm, 1, " cpm")),
            Row("  L / R", "${fmt(m.liveLegCadenceCpmLeft, 1)} / ${fmt(m.liveLegCadenceCpmRight, 1)}"),
            Row("leg favor", favor(m.liveLegAmplitudeSymmetry)),
            Row("leg cyc", "${m.liveLegNCyclesLeft} / ${m.liveLegNCyclesRight}"),
            Row("period CV", fmt(m.liveLegCyclePeriodCv, 3)),
        )
    } else {
        listOf(
            Row("sway RMS", fmt(m.liveSwayRmsM, 4, " m")),
            Row("  ML / AP", "${fmt(m.liveSwayMlRmsM, 4)} / ${fmt(m.liveSwayApRmsM, 4)}"),
            Row("sway vel", fmt(m.liveSwayVelocityMps, 4, " m/s")),
            Row("trunk dev", fmt(m.liveTrunkAngleDeltaDeg, 1, " deg")),
        )
    }

    /** A number, or `--` when it is NaN. */
    fun fmt(value: Double, digits: Int = 3, suffix: String = ""): String =
        if (value.isNaN()) "--" else "%.${digits}f%s".format(value, suffix)

    /**
     * Amplitude symmetry as magnitude + favoured side (`0.85 L`), or `even` near zero.
     *
     * A bare signed number does not read as "which leg"; this spells it out.
     */
    fun favor(sym: Double): String = when {
        sym.isNaN() -> "--"
        abs(sym) < 0.05 -> "even"
        else -> "%.2f %s".format(abs(sym), if (sym > 0) "L" else "R")
    }

    /**
     * How close the trunk is to its *own* rolling baseline, in `[0, 1]`, or null.
     *
     * `1.0` = on the window's median lean (steadiest); `0.0` = [STEADINESS_TOL_DEG] or more away.
     * Null when this is not a valid hold readout, so the caller draws nothing rather than a stale
     * bar.
     *
     * **The honesty is entirely in what it reads.** It is built on `liveTrunkAngleDeltaDeg` — the
     * current lean minus the window's own median — and *not* on an absolute upright angle, which
     * would inherit `WORLD_UP`'s level-camera assumption. That assumption is exactly what a phone
     * propped at a random angle breaks, so this is the one form of the meter that survives the
     * move off a tripod. It is also not a "good posture" threshold, which would be the
     * loss-of-posture criterion `hold.py` refuses to invent.
     */
    fun sitSteadiness(m: LiveMetrics?, tolDeg: Double = STEADINESS_TOL_DEG): Double? {
        if (m == null || m.liveMode != LiveMetricsComputer.HOLD || !m.liveValid) return null
        val delta = m.liveTrunkAngleDeltaDeg
        if (delta.isNaN()) return null
        return maxOf(0.0, 1.0 - abs(delta) / tolDeg)
    }

    /** Red (`q=0`) -> yellow (`0.5`) -> green (`q=1`). */
    fun qualityColor(q: Double): Color {
        val c = q.coerceIn(0.0, 1.0)
        return if (c < 0.5) {
            Color(red = 1f, green = (2 * c).toFloat(), blue = 0f)
        } else {
            Color(red = (2 * (1.0 - c)).toFloat(), green = 1f, blue = 0f)
        }
    }

    fun coverageColor(coverage: Double): Color =
        if (coverage >= LiveMetricsComputer.MIN_COVERAGE) OK_COLOR else BAD_COLOR
}
