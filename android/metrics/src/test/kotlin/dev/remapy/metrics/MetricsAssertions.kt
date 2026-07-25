package dev.remapy.metrics

import com.google.gson.JsonObject
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Compare a whole metrics record against its Python golden, field by field.
 *
 * Ints, booleans and strings must match exactly; doubles within [tolerance], with NaN-ness
 * compared first (see `assertClose`). Every key the golden carries must be present in
 * [actual], so a field dropped during the port fails loudly rather than going unchecked.
 *
 * [skip] is for fields this port deliberately does not compute. Passing a name here is a
 * *claim* that the omission is intentional, so keep the set small and state the reason at the
 * call site — an accidental omission and a deliberate one look identical from in here.
 */
fun assertMetricsMatch(
    expected: JsonObject,
    actual: Map<String, Any>,
    tolerance: Double = 1e-9,
    skip: Set<String> = emptySet(),
    label: String = "",
) {
    for ((key, value) in expected.entrySet()) {
        if (key in skip) continue
        val got = actual[key]
        assertTrue(actual.containsKey(key), "$label: golden field '$key' is missing from the port")
        when (got) {
            is Int -> assertEquals(value.asInt, got, "$label.$key")
            is Boolean -> assertEquals(value.asBoolean, got, "$label.$key")
            is String -> assertEquals(value.asString, got, "$label.$key")
            is Double -> assertClose(value.num(), got, tolerance, "$label.$key")
            else -> error("$label.$key: unsupported value type ${got?.javaClass}")
        }
    }
}

/** [Hold.HoldMetrics] keyed by the Python field names. */
fun Hold.HoldMetrics.toGoldenMap(): Map<String, Any> = linkedMapOf(
    "duration_s" to durationS,
    "tracked_s" to trackedS,
    "coverage" to coverage,
    "n_frames" to nFrames,
    "path_length_m" to pathLengthM,
    "mean_velocity_mps" to meanVelocityMps,
    "ellipse_area_m2" to ellipseAreaM2,
    "rms_m" to rmsM,
    "sway_ml_rms_m" to swayMlRmsM,
    "sway_ap_rms_m" to swayApRmsM,
    "trunk_angle_mean_deg" to trunkAngleMeanDeg,
    "trunk_angle_sd_deg" to trunkAngleSdDeg,
    "trunk_angle_range_deg" to trunkAngleRangeDeg,
    "hands_low_frac" to handsLowFrac,
    "up_source" to upSource,
)

/** [Crawl.CrawlMetrics] keyed by the Python field names. */
fun Crawl.CrawlMetrics.toGoldenMap(): Map<String, Any> = linkedMapOf(
    "duration_s" to durationS,
    "tracked_s" to trackedS,
    "coverage" to coverage,
    "n_frames" to nFrames,
    "cadence_cpm" to cadenceCpm,
    "cadence_cpm_left" to cadenceCpmLeft,
    "cadence_cpm_right" to cadenceCpmRight,
    "n_cycles_left" to nCyclesLeft,
    "n_cycles_right" to nCyclesRight,
    "cycle_period_sd_s" to cyclePeriodSdS,
    "cycle_period_cv" to cyclePeriodCv,
    "phase_offset" to phaseOffset,
    "phase_offset_circular_sd" to phaseOffsetCircularSd,
    "amplitude_symmetry" to amplitudeSymmetry,
    "speed_norm_per_s" to speedNormPerS,
    "leg_coverage" to legCoverage,
    "leg_tracked_s" to legTrackedS,
    "leg_cadence_cpm" to legCadenceCpm,
    "leg_cadence_cpm_left" to legCadenceCpmLeft,
    "leg_cadence_cpm_right" to legCadenceCpmRight,
    "leg_n_cycles_left" to legNCyclesLeft,
    "leg_n_cycles_right" to legNCyclesRight,
    "leg_cycle_period_sd_s" to legCyclePeriodSdS,
    "leg_cycle_period_cv" to legCyclePeriodCv,
    "leg_phase_offset" to legPhaseOffset,
    "leg_phase_offset_circular_sd" to legPhaseOffsetCircularSd,
    "leg_amplitude_symmetry" to legAmplitudeSymmetry,
)

/**
 * The crawl fields this port does not compute, because they need `scipy.signal.hilbert`.
 *
 * Reciprocity is excluded from the *live* path on the Python side too — Hilbert's edge effects
 * peak at exactly a short trailing window's edges — so nothing live is lost. What is lost is
 * the offline "bunny haul vs mature crawl" axis, which the phone therefore cannot report.
 */
val HILBERT_FIELDS = setOf(
    "phase_offset",
    "phase_offset_circular_sd",
    "leg_phase_offset",
    "leg_phase_offset_circular_sd",
)

/** Rebuild a `(N, 33, 3)` + visibility export into per-frame [PoseFrame]s, as the live path sees them. */
fun poseFrames(world: Array<FloatArray>, visibility: Array<FloatArray>): List<PoseFrame> =
    world.indices.map { i ->
        // A frame with no pose is a whole-row NaN, exactly as `landmark_rows` writes it.
        if (world[i][0].isNaN()) {
            PoseFrame.noPose()
        } else {
            // The exporter feeds one row to both world and norm, and one score column to both
            // visibility and presence, so the fixture mirrors that rather than inventing values.
            PoseFrame(world[i], world[i], visibility[i], visibility[i])
        }
    }
