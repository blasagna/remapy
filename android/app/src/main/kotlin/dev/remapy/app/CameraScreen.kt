package dev.remapy.app

import android.graphics.Bitmap
import androidx.camera.core.CameraSelector
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
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
import dev.remapy.metrics.Landmarks
import dev.remapy.metrics.LiveMetrics
import dev.remapy.metrics.PoseFrame

/** Landmarks below this are drawn dimmed — they are extrapolated, not measured. */
private const val MIN_VISIBILITY = 0.5f

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
    modifier: Modifier = Modifier,
) {
    Box(modifier = modifier.fillMaxSize().background(Color.Black)) {
        // Drawn before the early return so the control stays reachable while the camera is
        // starting up — including right after a flip, when there is briefly no frame yet.
        if (canFlipCamera) {
            FlipCameraButton(
                lensFacing = lensFacing,
                onClick = onFlipCamera,
                modifier = Modifier.align(Alignment.TopEnd).padding(12.dp),
            )
        }

        if (bitmap == null) {
            Text(
                "waiting for camera...",
                color = Color.Gray,
                modifier = Modifier.align(Alignment.Center),
            )
            return@Box
        }

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
            if (frame != null) drawSkeleton(frame, originX, originY, drawnW, drawnH)
        }

        MetricsPanel(
            metrics = metrics,
            fps = fps,
            usingGpu = usingGpu,
            modifier = Modifier.align(Alignment.TopStart).padding(12.dp),
        )
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
 * Rear/front lens toggle.
 *
 * Top-right, well clear of the metrics panel in the top-left. Sized generously because it gets
 * pressed one-handed while the other hand is steadying a child, and it names the lens rather than
 * using a bare flip glyph — the operator needs to know which camera is live without looking at the
 * image to work it out.
 */
@Composable
private fun FlipCameraButton(lensFacing: Int, onClick: () -> Unit, modifier: Modifier = Modifier) {
    val label = if (lensFacing == CameraSelector.LENS_FACING_FRONT) {
        "front"
    } else {
        "rear"
    }
    Box(
        modifier = modifier
            .clip(RoundedCornerShape(6.dp))
            .background(Color(0xB0000000))
            .clickable(onClick = onClick)
            .padding(horizontal = 16.dp, vertical = 12.dp)
    ) {
        Text(
            "⟲  $label",
            color = Color.White,
            fontFamily = FontFamily.Monospace,
            fontSize = 14.sp,
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

        // Sustained frame rate: the number that says whether the 30 Hz filter chain is being fed
        // what it assumes. Red below 25, where resampling up to the grid starts inventing
        // correlated samples.
        MetricRow(
            "fps",
            "%.0f  %s".format(fps, if (usingGpu) "gpu" else "cpu"),
            if (fps >= 25.0) Overlay.OK_COLOR else Overlay.BAD_COLOR,
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
