package app.gauge.wear.ui

import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.wear.compose.material.Chip
import androidx.wear.compose.material.ChipDefaults
import androidx.wear.compose.material.MaterialTheme
import androidx.wear.compose.material.ScalingLazyColumn
import androidx.wear.compose.material.Text
import androidx.wear.compose.material.rememberScalingLazyListState
import app.gauge.wear.emotion.EmotionDevBus
import app.gauge.wear.emotion.EmotionFlag
import app.gauge.wear.emotion.EmotionUi

/** What the screen says about the latest phone state — pure, so it is unit-tested. */
internal fun emotionHeadline(ui: EmotionUi): String = when {
    ui.label == null -> "Waiting for the phone"
    ui.paused -> "Paused"
    else -> "${ui.label} ${Math.round((ui.prob ?: 0.0) * 100)}%"
}

internal fun emotionFootnote(ui: EmotionUi): String? = when {
    ui.sendFailed -> "Not sent - no connection to the phone"
    ui.flagged != null -> "Flagged: ${ui.flagged.label}"
    else -> null
}

/**
 * Emotion dev tester screen: the latest emotion the PHONE heard (an experimental voice model — not
 * a verdict on you), an optional one-line suggestion, pause/resume, and the three flags. Flags apply
 * to the latest buzz (if it was within 10 minutes) and are recorded on the phone; an un-tapped buzz
 * stays "unrated" there.
 */
@Composable
fun EmotionDevScreen(modifier: Modifier = Modifier) {
    val ui by EmotionDevBus.model.state.collectAsState()
    val listState = rememberScalingLazyListState()
    ScalingLazyColumn(modifier = modifier.fillMaxWidth(), state = listState) {
        item { Text(text = "Emotion (dev)", style = MaterialTheme.typography.caption1) }
        item { Text(text = emotionHeadline(ui), style = MaterialTheme.typography.title3) }
        ui.suggestion?.let { s -> item { Text(text = s, style = MaterialTheme.typography.caption2) } }
        emotionFootnote(ui)?.let { f -> item { Text(text = f, style = MaterialTheme.typography.caption3) } }
        item {
            Chip(
                modifier = Modifier.fillMaxWidth(),
                label = { Text(if (ui.paused) "Resume" else "Pause") },
                colors = ChipDefaults.primaryChipColors(),
                onClick = { EmotionDevBus.model.setPaused(!ui.paused) },
            )
        }
        for (flag in EmotionFlag.entries) {
            item {
                Chip(
                    modifier = Modifier.fillMaxWidth(),
                    label = {
                        Text(
                            when (flag) {
                                EmotionFlag.BAD -> "✗ Wrong"
                                EmotionFlag.GOOD -> "✓ Good"
                                EmotionFlag.IGNORE -> "Ignore (media)"
                            },
                        )
                    },
                    colors = if (ui.flagged == flag) ChipDefaults.primaryChipColors() else ChipDefaults.secondaryChipColors(),
                    enabled = !ui.paused,
                    onClick = { EmotionDevBus.model.flag(flag) },
                )
            }
        }
    }
}
