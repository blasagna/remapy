package dev.remapy.app

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/**
 * The screen the app opens on.
 *
 * It exists so that launching the app is not the same act as pointing a camera at a child. Before
 * this, `onCreate` requested the camera permission and bound CameraX immediately, which meant the
 * first thing anyone saw was a system dialog with nothing behind it to explain the request — and
 * meant the camera was live from launch whether or not a session was about to start.
 *
 * Deliberately plain. It is passed through on the way to the live view, and the only thing it has
 * to do well is make the entry obvious to someone holding the phone for the first time.
 */
@Composable
fun MenuScreen(onEnterLive: () -> Unit, modifier: Modifier = Modifier) {
    Column(
        modifier = modifier
            // The window background is already black, but the live view paints its own and this
            // should not depend on which screen was shown before it.
            .background(Color.Black)
            .padding(32.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp, Alignment.CenterVertically),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text(
            "remapy",
            color = Color.White,
            fontFamily = FontFamily.Monospace,
            fontSize = 32.sp,
        )
        Text(
            "live motor metrics",
            color = Color.Gray,
            fontFamily = FontFamily.Monospace,
            fontSize = 14.sp,
            modifier = Modifier.padding(bottom = 16.dp),
        )

        Button(
            onClick = onEnterLive,
            modifier = Modifier.widthIn(min = 220.dp),
        ) {
            Text("Live view", fontSize = 18.sp)
        }

        // Present and disabled rather than absent. The landmark-parity measurement — running a
        // laptop recording's exported mp4 through this app and diffing the landmarks against the
        // `.h5` — is the one thing standing between phone sessions and the laptop's cross-session
        // trend, and it needs a file source. Showing the slot it will occupy keeps that visible
        // instead of leaving it as a line in a TODO nobody opens mid-session.
        Button(
            onClick = {},
            enabled = false,
            modifier = Modifier.widthIn(min = 220.dp),
        ) {
            Text("File source", fontSize = 18.sp)
        }
        Text(
            "not built yet — needed to check this phone's\nlandmarks against the laptop's",
            color = Color.DarkGray,
            fontFamily = FontFamily.Monospace,
            fontSize = 11.sp,
            modifier = Modifier.fillMaxWidth().padding(top = 4.dp),
        )
    }
}
