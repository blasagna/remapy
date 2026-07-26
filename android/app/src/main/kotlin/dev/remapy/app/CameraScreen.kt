package dev.remapy.app

import android.graphics.Bitmap
import androidx.activity.compose.BackHandler
import androidx.camera.core.CameraSelector
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import dev.remapy.metrics.Derive
import dev.remapy.metrics.Landmarks
import dev.remapy.metrics.LiveMetrics
import dev.remapy.metrics.PoseFrame

/** Landmarks below this are drawn dimmed — they are extrapolated, not measured. */
private const val MIN_VISIBILITY = 0.5f

/**
 * Fraction of `Derive.FS` at which the fps readout stays green.
 *
 * Some slack is needed because the displayed rate is an EMA of *delivered* frames and a decimator
 * targeting the grid exactly will always sit a shade under it. Enough slack to not cry wolf, not
 * so much that a genuinely struggling device reads healthy.
 */
private const val FPS_OK_FRACTION = 0.9

/**
 * The live view: the redacted camera frame, the skeleton, and the metrics readout.
 *
 * The frame comes in already redacted — see [FaceRedaction] for why this screen draws bitmaps
 * itself instead of using CameraX's `PreviewView`.
 */
@Composable
fun CameraScreen(
    bitmap: Bitmap?,
    frame: PoseFrame?,
    metrics: LiveMetrics?,
    fps: Double,
    usingGpu: Boolean,
    lensFacing: Int,
    canFlipCamera: Boolean,
    onFlipCamera: () -> Unit,
    mode: String,
    overlayOff: Boolean,
    onToggleMode: () -> Unit,
    portrait: Boolean,
    onToggleOrientation: () -> Unit,
    videoOnly: Boolean,
    onToggleVideoOnly: () -> Unit,
    onExit: () -> Unit,
    modifier: Modifier = Modifier,
) {
    BackHandler(onBack = onExit)
    Box(modifier = modifier.fillMaxSize().background(Color.Black)) {
        // The video is **full-bleed on purpose**, drawn before and underneath the chrome layer.
        // Insetting it would shrink the image on every device with a cutout in order to protect
        // what is mostly letterbox bar; a punch hole sitting over part of the camera image costs
        // the operator nothing, while a punch hole over the `coverage` row costs them the one
        // number that says whether to trust the rest.
        if (bitmap != null) {
            Canvas(modifier = Modifier.fillMaxSize()) {
                val scale = minOf(size.width / bitmap.width, size.height / bitmap.height)
                val drawnW = bitmap.width * scale
                val drawnH = bitmap.height * scale
                val originX = (size.width - drawnW) / 2f
                val originY = (size.height - drawnH) / 2f

                drawImage(
                    image = bitmap.asImageBitmap(),
                    dstOffset = androidx.compose.ui.unit.IntOffset(originX.toInt(), originY.toInt()),
                    dstSize = androidx.compose.ui.unit.IntSize(drawnW.toInt(), drawnH.toInt()),
                )
                // No skeleton in raw video: there is no pose behind it to draw. This is the
                // one thing `videoOnly` and the readout's `off` state do *not* share — `off`
                // keeps the skeleton precisely because tracking is still running under it.
                if (frame != null && !videoOnly) {
                    drawSkeleton(frame, originX, originY, drawnW, drawnH)
                }
            }
        }

        // Everything readable sits inside the safe area. One `windowInsetsPadding` on the layer
        // rather than on each aligned child: on a wrap-content node anchored to a corner, the
        // far-edge insets would inflate the node for nothing.
        //
        // `safeDrawing` is already the union of system bars, display cutout and IME. Assembling
        // `displayCutout.union(statusBars)` by hand would mean re-deriving it — wrongly — later.
        // Known cost: in landscape the cutout inset applies to the *whole* edge even though the
        // hole occupies a third of it, so ~30 dp of overlay width goes unused. Accepted; the
        // alternative is per-cutout-bounds geometry for a problem that does not warrant it.
        Box(
            modifier = Modifier
                .fillMaxSize()
                .windowInsetsPadding(WindowInsets.safeDrawing)
        ) {
            // Drawn before the early return so the controls stay reachable while the camera is
            // starting up — including right after a flip, a rotation or a mode change, when the
            // window is empty and there is briefly no frame yet.
            Column(
                modifier = Modifier.align(Alignment.TopEnd).padding(12.dp),
                horizontalAlignment = Alignment.End,
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                // The way back. The system bars are hidden, so the back gesture is there but
                // undiscoverable — which matters most on a phone handed to a therapist or a
                // family member who has not used this before.
                OverlayButton(
                    label = "menu",
                    glyph = "‹",
                    onClick = onExit,
                )
                OverlayButton(
                    // Whether anything is being measured at all, and the only control that
                    // changes what is *shown* of the child rather than what is drawn over him.
                    label = if (videoOnly) "raw video" else "pose",
                    glyph = "◉",
                    onClick = onToggleVideoOnly,
                )
                OverlayButton(
                    // The exercise being watched, and the thing that changes what every row below
                    // means. Named rather than iconified for that reason — and `off` is named the
                    // same way, so a hidden panel is never mistaken for a stalled one.
                    label = if (overlayOff) "off" else mode,
                    glyph = "⇄",
                    onClick = onToggleMode,
                )
                if (canFlipCamera) {
                    OverlayButton(
                        label = if (lensFacing == CameraSelector.LENS_FACING_FRONT) "front" else "rear",
                        glyph = "⟲",
                        onClick = onFlipCamera,
                    )
                }
                OverlayButton(
                    label = if (portrait) "portrait" else "landscape",
                    glyph = "▯",
                    onClick = onToggleOrientation,
                )
            }

            if (bitmap == null) {
                Text(
                    "waiting for camera...",
                    color = Color.Gray,
                    modifier = Modifier.align(Alignment.Center),
                )
            } else if (videoOnly) {
                // Takes the panel's place rather than sitting beside it: in this state there are
                // no metrics to crowd, and the warning belongs where the operator's eye already
                // goes for the coverage row.
                RawVideoWarning(modifier = Modifier.align(Alignment.TopStart).padding(12.dp))
            } else if (!overlayOff) {
                MetricsPanel(
                    metrics = metrics,
                    fps = fps,
                    usingGpu = usingGpu,
                    modifier = Modifier.align(Alignment.TopStart).padding(12.dp),
                )
            }
        }
    }
}

/**
 * Port of `pose_estimation/draw.py::draw_skeleton`.
 *
 * Two behaviours carried over, both load-bearing rather than cosmetic. **NaN points and bones are
 * skipped** — untracked frames are full-NaN rows by convention, and a NaN coordinate is not a
 * position. And **landmarks below [MIN_VISIBILITY] are dimmed**, because MediaPipe *extrapolates*
 * occluded points rather than dropping them: an invented coordinate would otherwise look identical
 * to a measured one. The threshold matches `Quality.Gate`, so what looks solid on screen is what
 * the metrics will accept.
 */
private fun DrawScope.drawSkeleton(
    frame: PoseFrame,
    originX: Float,
    originY: Float,
    width: Float,
    height: Float,
) {
    fun pointOf(i: Int): Offset? {
        val x = frame.norm[i * 3]
        val y = frame.norm[i * 3 + 1]
        if (x.isNaN() || y.isNaN()) return null
        return Offset(originX + x * width, originY + y * height)
    }

    fun colorOf(i: Int): Color =
        if (frame.visibility[i] < MIN_VISIBILITY) Color(0xFF206060) else Color(0xFF00E5E5)

    for ((a, b) in Landmarks.POSE_CONNECTIONS) {
        val pa = pointOf(a) ?: continue
        val pb = pointOf(b) ?: continue
        val dim = frame.visibility[a] < MIN_VISIBILITY || frame.visibility[b] < MIN_VISIBILITY
        drawLine(
            color = if (dim) Color(0xFF206060) else Color(0xFF00B4B4),
            start = pa,
            end = pb,
            strokeWidth = 3f,
        )
    }
    for (i in 0 until Landmarks.COUNT) {
        val p = pointOf(i) ?: continue
        drawCircle(color = colorOf(i), radius = 4f, center = p)
    }
}

/**
 * A control in the top-right stack.
 *
 * Top-right keeps it well clear of the metrics panel in the top-left. Sized generously because it
 * gets pressed one-handed while the other hand is steadying a child, and each shows its **current
 * state as a word** rather than a bare glyph — the operator needs to know which camera, which
 * exercise and which framing are live without studying the image to infer it.
 */
@Composable
private fun OverlayButton(
    label: String,
    glyph: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Box(
        modifier = modifier
            .clip(RoundedCornerShape(6.dp))
            .background(Color(0xB0000000))
            .clickable(onClick = onClick)
            .padding(horizontal = 16.dp, vertical = 12.dp)
    ) {
        Text(
            "$glyph  $label",
            color = Color.White,
            fontFamily = FontFamily.Monospace,
            fontSize = 14.sp,
        )
    }
}

/**
 * The standing notice that face redaction is off.
 *
 * Deliberately not left to the `raw video` button label. Every other control on this screen
 * changes what is drawn *over* the child; this one changes what is shown *of* him, and it is the
 * single state in which an unredacted face reaches the screen. A word in the corner of a button is
 * not proportionate to that, so this is red, permanent while the state holds, and says what is and
 * is not happening to the frame.
 *
 * The second line matters as much as the first: raw is only defensible because nothing here is
 * written anywhere, and the operator should be able to read that off the screen rather than
 * remember it.
 */
@Composable
private fun RawVideoWarning(modifier: Modifier = Modifier) {
    Column(
        modifier = modifier
            .background(Color(0xB0000000))
            .padding(horizontal = 10.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(2.dp),
    ) {
        Text(
            "RAW VIDEO — FACE NOT REDACTED",
            color = Overlay.BAD_COLOR,
            fontFamily = FontFamily.Monospace,
            fontSize = 13.sp,
        )
        Text(
            "no pose, no metrics, nothing recorded",
            color = Color.White,
            fontFamily = FontFamily.Monospace,
            fontSize = 11.sp,
        )
    }
}

@Composable
private fun MetricsPanel(
    metrics: LiveMetrics?,
    fps: Double,
    usingGpu: Boolean,
    modifier: Modifier = Modifier,
) {
    if (metrics == null) return
    Column(
        modifier = modifier
            // The translucent panel is what buys legibility over busy footage — the same job
            // `draw.shade_box` does on the desktop overlay.
            .background(Color(0xB0000000))
            .padding(horizontal = 10.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(2.dp),
    ) {
        // Coverage first, and coloured by whether it clears the gate: it is the reason any of the
        // values below might be dashes, and it must be visible next to them.
        MetricRow(
            "coverage",
            "%.0f%%  (%.1fs)".format(metrics.liveCoverage * 100, metrics.liveTrackedS),
            Overlay.coverageColor(metrics.liveCoverage),
        )
        for (row in Overlay.rows(metrics)) MetricRow(row.label, row.value, Overlay.TEXT_COLOR)

        // `up:` carries the level-camera assumption. Orange because it is a caveat, not a
        // measurement — and on a hand-held phone it is the caveat most likely to bite.
        MetricRow("up", metrics.liveUpSource, Overlay.WARN_COLOR)

        // Sustained frame rate: the number that says whether the filter chain is being fed what it
        // assumes. The threshold is derived from the grid, not written down — below it, resampling
        // *up* to the grid starts inventing correlated samples and the documented derivative-gain
        // table stops describing the filter. The pipeline decimates *to* `Derive.FS`, so a healthy
        // reading sits at the grid rate rather than above it.
        MetricRow(
            "fps",
            "%.0f  %s".format(fps, if (usingGpu) "gpu" else "cpu"),
            if (fps >= FPS_OK_FRACTION * Derive.FS) Overlay.OK_COLOR else Overlay.BAD_COLOR,
        )

        Overlay.sitSteadiness(metrics)?.let { SteadinessMeter(it) }
    }
}

@Composable
private fun MetricRow(label: String, value: String, color: Color) {
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(label, color = color, fontFamily = FontFamily.Monospace, fontSize = 13.sp)
        Text(value, color = color, fontFamily = FontFamily.Monospace, fontSize = 13.sp)
    }
}

/**
 * The child-facing piece: a red -> green fill bar on a continuum, no pass/fail line.
 *
 * See [Overlay.sitSteadiness] for what it reads and why that choice is the honest one.
 */
@Composable
private fun SteadinessMeter(quality: Double) {
    Column(modifier = Modifier.padding(top = 6.dp)) {
        Text("steady", color = Color.White, fontFamily = FontFamily.Monospace, fontSize = 13.sp)
        Canvas(modifier = Modifier.padding(top = 3.dp).size(width = 160.dp, height = 14.dp)) {
            drawRect(color = Color(0xFF303030), size = size)
            drawRect(
                color = Overlay.qualityColor(quality),
                size = Size((size.width * quality).toFloat().coerceIn(0f, size.width), size.height),
            )
        }
    }
}
