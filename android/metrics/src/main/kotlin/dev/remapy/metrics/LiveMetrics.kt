package dev.remapy.metrics

/**
 * One live readout. **Every field name is prefixed `live`** — see [LiveMetricsComputer].
 *
 * Fields are the union across modes, the same choice the offline report table makes: a hold
 * readout carries NaN in the crawl fields and vice versa, so a renderer has one shape to
 * handle rather than two.
 *
 * Numeric fields are NaN whenever [liveValid] is false. That is the **blanking rule** — a
 * stale number left on screen during a tracking dropout reads as a measurement of the child,
 * which is worse than showing nothing. Renderers must draw NaN as `--`, never as 0.
 */
data class LiveMetrics(
    val liveMode: String,
    val liveWindowS: Double,
    val liveValid: Boolean,

    // Quality. Refreshed every frame, and the precondition for everything below.
    val liveNFrames: Int,
    val liveCoverage: Double,
    val liveTrackedS: Double,

    // Hold: sway of the trunk over the pelvis. `path_length_m` has no live counterpart — it is
    // duration-confounded, and a fixed window makes the velocity form the right one for free.
    // `ellipse_area_m2` is dropped too: it reads ~0 for one-axis rocking, so the ML/AP split
    // beside the radial RMS is the honest presentation.
    val liveSwayRmsM: Double,
    /** Image plane; measured well. */
    val liveSwayMlRmsM: Double,
    /** Inferred depth; markedly noisier. */
    val liveSwayApRmsM: Double,
    val liveSwayVelocityMps: Double,

    // Trunk lean. `liveTrunkAngleDeltaDeg` is the one to display: an absolute lean inherits
    // WORLD_UP's level-camera assumption, which is exactly what a phone propped at a random
    // angle breaks. Referencing the window's own median moves a tilted camera into the
    // baseline instead of into the number.
    /** Instantaneous, [LiveMetricsComputer.LIVE_LAG] samples back. */
    val liveTrunkAngleDeg: Double,
    /** Window median. */
    val liveTrunkAngleBaselineDeg: Double,
    val liveTrunkAngleDeltaDeg: Double,
    val liveUpSource: String,

    // Crawl. Reads no vertical at all (the axis is the body's own trunk vector), which makes it
    // the most camera-robust mode here. `speed_norm_per_s` is excluded: it is image-widths per
    // second and only comparable at fixed framing. `phase_offset` too — Hilbert's edge effects
    // are worst at exactly a short trailing window's edges, so reciprocity stays offline.
    /** Arms (wrists). */
    val liveCadenceCpm: Double,
    val liveCadenceCpmLeft: Double,
    val liveCadenceCpmRight: Double,
    val liveNCyclesLeft: Int,
    val liveNCyclesRight: Int,
    val liveCyclePeriodCv: Double,

    // Legs (knees). The developmental signal for Remy is here, not in the arms: he drives with
    // the legs and favours one repeatedly. `liveLegAmplitudeSymmetry` is the "favours one leg"
    // readout (0 = even, sign gives the side); it is a *within-window* left/right comparison,
    // not the between-trial symmetry index that stays offline.
    val liveLegCadenceCpm: Double,
    val liveLegCadenceCpmLeft: Double,
    val liveLegCadenceCpmRight: Double,
    val liveLegNCyclesLeft: Int,
    val liveLegNCyclesRight: Int,
    val liveLegCyclePeriodCv: Double,
    val liveLegAmplitudeSymmetry: Double,
) {
    /**
     * This readout keyed by the **Python field names**, in declaration order.
     *
     * Two jobs. It is what a renderer walks to lay out rows without hard-coding an order, and
     * it is the structural half of the never-mix rule: every key here is `live_`-prefixed, so
     * no field of a live readout can collide with an offline `metrics_table` column and a live
     * row cannot be concatenated into an offline frame by accident. `LiveMetricsTest` pins both
     * the prefix and the exact key set against the Python dataclass.
     *
     * **Live values must never reach a recording or an offline table.** Different window, no
     * human-marked boundaries, and an instantaneous readout that is three samples old: the same
     * name would be a different measurement. Keep the prefix on anything added later.
     */
    fun toMap(): Map<String, Any> = linkedMapOf(
        "live_mode" to liveMode,
        "live_window_s" to liveWindowS,
        "live_valid" to liveValid,
        "live_n_frames" to liveNFrames,
        "live_coverage" to liveCoverage,
        "live_tracked_s" to liveTrackedS,
        "live_sway_rms_m" to liveSwayRmsM,
        "live_sway_ml_rms_m" to liveSwayMlRmsM,
        "live_sway_ap_rms_m" to liveSwayApRmsM,
        "live_sway_velocity_mps" to liveSwayVelocityMps,
        "live_trunk_angle_deg" to liveTrunkAngleDeg,
        "live_trunk_angle_baseline_deg" to liveTrunkAngleBaselineDeg,
        "live_trunk_angle_delta_deg" to liveTrunkAngleDeltaDeg,
        "live_up_source" to liveUpSource,
        "live_cadence_cpm" to liveCadenceCpm,
        "live_cadence_cpm_left" to liveCadenceCpmLeft,
        "live_cadence_cpm_right" to liveCadenceCpmRight,
        "live_n_cycles_left" to liveNCyclesLeft,
        "live_n_cycles_right" to liveNCyclesRight,
        "live_cycle_period_cv" to liveCyclePeriodCv,
        "live_leg_cadence_cpm" to liveLegCadenceCpm,
        "live_leg_cadence_cpm_left" to liveLegCadenceCpmLeft,
        "live_leg_cadence_cpm_right" to liveLegCadenceCpmRight,
        "live_leg_n_cycles_left" to liveLegNCyclesLeft,
        "live_leg_n_cycles_right" to liveLegNCyclesRight,
        "live_leg_cycle_period_cv" to liveLegCyclePeriodCv,
        "live_leg_amplitude_symmetry" to liveLegAmplitudeSymmetry,
    )

    companion object {
        /** A readout with the quality figures filled in and every measurement NaN. */
        fun blank(
            mode: String,
            windowS: Double,
            nFrames: Int,
            coverage: Double,
            trackedS: Double,
            upSource: String,
        ): LiveMetrics = LiveMetrics(
            liveMode = mode,
            liveWindowS = windowS,
            liveValid = false,
            liveNFrames = nFrames,
            liveCoverage = coverage,
            liveTrackedS = trackedS,
            liveSwayRmsM = Double.NaN,
            liveSwayMlRmsM = Double.NaN,
            liveSwayApRmsM = Double.NaN,
            liveSwayVelocityMps = Double.NaN,
            liveTrunkAngleDeg = Double.NaN,
            liveTrunkAngleBaselineDeg = Double.NaN,
            liveTrunkAngleDeltaDeg = Double.NaN,
            liveUpSource = upSource,
            liveCadenceCpm = Double.NaN,
            liveCadenceCpmLeft = Double.NaN,
            liveCadenceCpmRight = Double.NaN,
            liveNCyclesLeft = 0,
            liveNCyclesRight = 0,
            liveCyclePeriodCv = Double.NaN,
            liveLegCadenceCpm = Double.NaN,
            liveLegCadenceCpmLeft = Double.NaN,
            liveLegCadenceCpmRight = Double.NaN,
            liveLegNCyclesLeft = 0,
            liveLegNCyclesRight = 0,
            liveLegCyclePeriodCv = Double.NaN,
            liveLegAmplitudeSymmetry = Double.NaN,
        )
    }
}
