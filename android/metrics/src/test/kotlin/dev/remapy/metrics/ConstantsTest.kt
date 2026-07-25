package dev.remapy.metrics

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Every constant this port retyped by hand, checked against the Python it came from.
 *
 * These are the cheapest possible tests and they guard the most embarrassing possible bug. A
 * landmark index off by one measures the wrong joint and still produces plausible numbers; a
 * silently retuned `WINDOW_S` produces numbers that are not comparable to any previous
 * session, which is the one thing `derive.py` is written to prevent.
 */
class ConstantsTest {

    private val constants = Goldens.group("constants").asJsonObject

    @Test
    fun `filter chain constants match derive dot py`() {
        assertClose(constants["FS"].num(), Derive.FS, 0.0, "FS")
        assertClose(constants["WINDOW_S"].num(), Derive.WINDOW_S, 0.0, "WINDOW_S")
        assertEquals(constants["POLY"].asInt, Derive.POLY, "POLY")
        assertEquals(constants["window_length"].asInt, Derive.windowLength(), "windowLength")
    }

    @Test
    fun `live constants match live dot py`() {
        assertEquals(constants["LIVE_LAG"].asInt, LiveMetricsComputer.LIVE_LAG)
        assertEquals(constants["RECOMPUTE_EVERY"].asInt, LiveMetricsComputer.RECOMPUTE_EVERY)
        assertClose(constants["MIN_COVERAGE"].num(), LiveMetricsComputer.MIN_COVERAGE, 0.0)
        val modes = constants["MODE_WINDOW_S"].asJsonObject
        assertEquals(modes.keySet(), LiveMetricsComputer.MODE_WINDOW_S.keys)
        for ((mode, windowS) in modes.entrySet()) {
            assertClose(windowS.num(), LiveMetricsComputer.MODE_WINDOW_S.getValue(mode), 0.0, mode)
        }
    }

    /** [LiveMetricsComputer.LIVE_LAG] is not a free parameter — it is the fit's half-width. */
    @Test
    fun `LIVE_LAG is the savgol half-width, not a tuning knob`() {
        assertEquals(Derive.windowLength() / 2, LiveMetricsComputer.LIVE_LAG)
    }

    @Test
    fun `crawl gates match crawl dot py`() {
        assertClose(constants["CYCLE_PROMINENCE_FRAC"].num(), Crawl.CYCLE_PROMINENCE_FRAC, 0.0)
        assertClose(constants["MIN_CYCLE_EXCURSION_M"].num(), Crawl.MIN_CYCLE_EXCURSION_M, 0.0)
    }

    @Test
    fun `gate defaults match quality dot py`() {
        val gate = Quality.Gate()
        assertClose(constants["gate_min_visibility"].num(), gate.minVisibility, 0.0)
        assertClose(constants["gate_min_presence"].num(), gate.minPresence, 0.0)
    }

    @Test
    fun `WORLD_UP matches signals dot py`() {
        assertClose(constants["WORLD_UP"].doubles(), Signals.WORLD_UP, 0.0)
    }

    @Test
    fun `landmark groups match the PoseLandmark values`() {
        val groups = mapOf(
            "TORSO" to Landmarks.TORSO,
            "ARMS" to Landmarks.ARMS,
            "LEGS" to Landmarks.LEGS,
            "WRISTS" to Landmarks.WRISTS,
            "KNEES" to Landmarks.KNEES,
            "ARM_LANDMARKS" to Crawl.ARM_LANDMARKS,
            "LEG_LANDMARKS" to Crawl.LEG_LANDMARKS,
        )
        for ((name, actual) in groups) {
            assertEquals(constants[name].ints().toList(), actual.toList(), name)
        }
    }

    @Test
    fun `the skeleton edge list matches, pair for pair and in order`() {
        val expected = constants["POSE_CONNECTIONS"].asJsonArray.map {
            val pair = it.ints()
            pair[0] to pair[1]
        }
        assertEquals(35, expected.size, "the fixture itself should carry 35 pairs")
        assertEquals(expected, Landmarks.POSE_CONNECTIONS, "POSE_CONNECTIONS")
    }

    @Test
    fun `joint triplets match angles dot py`() {
        val expected = constants["JOINT_TRIPLETS"].asJsonObject
        assertEquals(expected.keySet(), Landmarks.JOINT_TRIPLETS.keys)
        for ((name, triplet) in expected.entrySet()) {
            val (a, joint, c) = Landmarks.JOINT_TRIPLETS.getValue(name)
            assertEquals(triplet.ints().toList(), listOf(a, joint, c), name)
        }
    }

    /**
     * The never-mix rule, structurally.
     *
     * Every live field is `live_`-prefixed so a live readout cannot be concatenated into an
     * offline table by accident — different window, no marked trial, and a readout three
     * samples old means the same name would be a different measurement. Checked here against
     * Python's own field list rather than against a copy of it, so the two cannot drift.
     */
    @Test
    fun `live field names match the Python dataclass exactly and all carry the prefix`() {
        val expected = constants["live_field_names"].asJsonArray.map { it.asString }
        val actual = LiveMetrics.blank("hold", 5.0, 0, 0.0, 0.0, "world_y").toMap().keys.toList()
        assertEquals(expected, actual, "live field names")
        assertTrue(actual.all { it.startsWith("live_") }, "every field must carry the live_ prefix")
    }
}
