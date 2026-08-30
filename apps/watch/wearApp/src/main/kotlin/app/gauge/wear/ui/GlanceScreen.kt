package app.gauge.wear.ui

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Devices
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.wear.compose.material.Chip
import androidx.wear.compose.material.ChipDefaults
import androidx.wear.compose.material.MaterialTheme
import androidx.wear.compose.material.ScalingLazyColumn
import androidx.wear.compose.material.Text
import androidx.wear.compose.material.rememberScalingLazyListState
import app.gauge.shared.sentinel.Mode
import app.gauge.shared.signals.SignalKind

private const val SPARKLINE_WINDOW = 30

/** v0.3.1 status perimeter: the one at-a-glance answer to "is it on?" — green frame when the
 * sentinel is on, red when off, independent of the meter/nudge color inside it. */
private const val STATUS_ON_GREEN = 0xFF2E7D32
private const val STATUS_OFF_RED = 0xFFC62828

/**
 * The glanceable home screen. Purely presentational: every value shown comes straight from
 * [GlanceUi] (produced by [GaugeViewModel]) — this composable makes no sentinel/mode-mapping
 * decisions of its own, only "which intent does this button press send" and "was the mic
 * permission I just asked for denied".
 *
 * Task 11 redesign: the old "Arm"/"Disarm" chip is gone — the central ring ([GlanceRing]) is
 * itself the on/off toggle now, and mic permission is requested proactively on first open
 * ([LaunchedEffect] below) rather than deferred to the first tap, since a ring that's just a
 * status readout (not an obvious button) was exactly the "looks tappable but isn't"/"doesn't look
 * tappable but is" confusion the real-device feedback called out.
 *
 * [ScalingLazyColumn] (not a plain `Column`) so content degrades gracefully on round screens.
 */
@Composable
fun GlanceScreen(
    uiState: GlanceUi,
    onOpenModeScreen: () -> Unit,
    onOpenSignalScreen: () -> Unit,
    onOpenSettingsScreen: () -> Unit,
    onOpenSignInScreen: () -> Unit,
    onOpenCouplesScreen: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    var micPermissionDenied by remember { mutableStateOf(false) }
    // Set only by an explicit ring tap (never by the first-open prime below) — see
    // [shouldArmOnGrant]'s KDoc for why this exists: a permission *grant* alone must never be
    // enough to arm the mic-recording foreground service.
    var pendingArmRequest by remember { mutableStateOf(false) }

    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions(),
    ) { grants ->
        // Only RECORD_AUDIO gates arming — POST_NOTIFICATIONS denial just means no notification
        // is shown, the sentinel still runs.
        val granted = grants[Manifest.permission.RECORD_AUDIO] == true
        val explicitRequest = pendingArmRequest
        pendingArmRequest = false
        if (shouldArmOnGrant(granted = granted, explicitArmRequest = explicitRequest)) {
            micPermissionDenied = false
            sendArm(context)
        } else {
            micPermissionDenied = !granted
        }
    }

    // Ask for the mic permission proactively the first time the screen appears, rather than
    // waiting for the wearer to tap the ring — see this function's own KDoc for why. This never
    // sets [pendingArmRequest], so [shouldArmOnGrant] never arms off the resulting grant: the
    // ring tap stays the only thing that turns the sentinel on.
    LaunchedEffect(Unit) {
        if (!hasRecordAudioPermission(context)) {
            permissionLauncher.launch(runtimePermissions())
        }
    }

    val listState = rememberScalingLazyListState()

    // Wave C Task 13: retro-capture consent step. showRetroCaptureConfirm gates a second,
    // one-item confirmation list (this app has no Dialog composable precedent yet, so this
    // reuses the existing ScalingLazyColumn/Chip idiom rather than introducing a new one).
    // retroCaptureResult mirrors RetroCaptureBus.lastResult — cleared 2s after it first becomes
    // non-null so a "Saved"/"Couldn't save" line reads as a toast, not a permanent state, and a
    // re-open of this screen later doesn't replay a stale result from an unrelated earlier tap.
    var showRetroCaptureConfirm by remember { mutableStateOf(false) }
    val retroCaptureResult by app.gauge.wear.capture.RetroCaptureBus.lastResult.collectAsState()

    LaunchedEffect(retroCaptureResult) {
        if (retroCaptureResult != null) {
            kotlinx.coroutines.delay(2000)
            app.gauge.wear.capture.RetroCaptureBus.clear()
        }
    }

    // Journal A/B toggle state. journalOn mirrors the pref (the service reads the pref itself
    // each tick, so this is display state only); the consent confirmation reuses the same
    // ScalingLazyColumn/Chip idiom as the retro-capture confirm above — turning Journal ON asks
    // ONCE, and the confirm tap is the consent artifact (GaugePrefs.enableJournal stores it with
    // a timestamp; toggling off clears it, so the next ON asks again).
    var journalOn by remember { mutableStateOf(app.gauge.wear.prefs.GaugePrefs.journalMode(context)) }
    var showJournalConsent by remember { mutableStateOf(false) }
    var journalNeedsPairing by remember { mutableStateOf(false) }

    // Task 11's arm/disarm toggle body, extracted so both center visualizations share the exact
    // same tap gesture (Addendum 2 gesture contract — inviolable, see class KDoc: the center tap
    // is the arm/disarm toggle regardless of which view is showing, never a display switcher).
    val onToggle: () -> Unit = {
        if (uiState.isOn) {
            sendDisarm(context)
        } else if (hasRecordAudioPermission(context)) {
            sendArm(context)
        } else {
            pendingArmRequest = true
            permissionLauncher.launch(runtimePermissions())
        }
    }

    ScalingLazyColumn(
        modifier = modifier.fillMaxWidth(),
        state = listState,
    ) {
        item {
            // Addendum 2: exactly ONE center visualization at a time (never both). The tap gesture
            // is the arm/disarm toggle on BOTH views — inviolable; the switcher is Settings-only.
            when (uiState.centerDisplay) {
                CenterDisplay.SPARKLINE -> CenterSparkline(uiState = uiState, onToggle = onToggle)
                CenterDisplay.DIAL -> GlanceRing(uiState = uiState, onToggle = onToggle)
            }
        }
        if (uiState.showEpisode) {
            uiState.vectorIcon?.let { icon ->
                item { Text(text = icon, style = MaterialTheme.typography.title1) }
            }
        }
        // P4-10: the honest "why there's no number" line, directly under the ring and above the
        // signal chip. Precomputed by GaugeViewModel (GlanceUi.signalStatus) — this composable
        // decides nothing, same rule as every other field here. Absent (no item emitted at all)
        // whenever there's nothing honest to say, so the layout doesn't reserve dead space.
        uiState.signalStatus?.let { status ->
            item {
                Text(
                    text = status,
                    style = MaterialTheme.typography.caption2,
                    color = MaterialTheme.colors.onSurfaceVariant,
                    textAlign = TextAlign.Center,
                )
            }
        }
        if (micPermissionDenied) {
            item {
                Text(
                    text = "Mic permission required",
                    style = MaterialTheme.typography.caption2,
                    color = MaterialTheme.colors.error,
                )
            }
        }
        item {
            Chip(
                modifier = Modifier.fillMaxWidth(),
                // P4-6 (Issue A fix): uiState.signal is always the wearer's TRUE selection now
                // (see GaugeViewModel's constructor KDoc) — the "(no reading)" suffix is an honest
                // addendum, never a substitute selection, so the wearer never sees a selection they
                // didn't make.
                label = {
                    val suffix = if (uiState.signalHasReading) "" else " (no reading)"
                    Text("Signal: ${uiState.signal.displayLabel()}$suffix ▾")
                },
                colors = ChipDefaults.secondaryChipColors(),
                onClick = onOpenSignalScreen,
            )
        }
        if (uiState.signedIn) {
            item {
                Chip(
                    modifier = Modifier.fillMaxWidth(),
                    label = { Text("💞 You & your person") },
                    colors = ChipDefaults.secondaryChipColors(),
                    onClick = onOpenCouplesScreen,
                )
            }
        }
        if (uiState.signedIn && uiState.retroCaptureAvailableSeconds > 0 && !showRetroCaptureConfirm) {
            item {
                Chip(
                    modifier = Modifier.fillMaxWidth(),
                    label = { Text("🎙 Save last 2 min") },
                    colors = ChipDefaults.secondaryChipColors(),
                    onClick = { showRetroCaptureConfirm = true },
                )
            }
        }
        if (showRetroCaptureConfirm) {
            item {
                Text(
                    text = "Save the last 2 minutes of your OWN audio? Only your voice is captured — you can review it later in the captures workbench.",
                    style = MaterialTheme.typography.caption2,
                    textAlign = TextAlign.Center,
                )
            }
            item {
                Chip(
                    modifier = Modifier.fillMaxWidth(),
                    label = { Text("Confirm & save") },
                    colors = ChipDefaults.primaryChipColors(),
                    onClick = {
                        showRetroCaptureConfirm = false
                        // Post-Task-11 correction (see Task 12's own note): the wearer's tap on
                        // THIS chip is the one and only consent artifact — carried explicitly as
                        // an Intent extra so SentinelService's onActionRetroCapture never has to
                        // assume consent just because the action fired, and so
                        // RetroCaptureUploader.upload's required consentConfirmed parameter is
                        // fed a real signal, not a hardcoded true reintroducing the gap Task 11
                        // closed. Cancel (below) sends no intent at all -- dismiss = no upload,
                        // buffer untouched.
                        context.startService(
                            android.content.Intent(context, app.gauge.wear.service.SentinelService::class.java)
                                .setAction(app.gauge.wear.service.SentinelService.ACTION_RETRO_CAPTURE)
                                .putExtra(app.gauge.wear.service.SentinelService.EXTRA_CONSENT_CONFIRMED, true),
                        )
                    },
                )
            }
            item {
                Chip(
                    modifier = Modifier.fillMaxWidth(),
                    label = { Text("Cancel") },
                    colors = ChipDefaults.secondaryChipColors(),
                    onClick = { showRetroCaptureConfirm = false },
                )
            }
        }
        retroCaptureResult?.let { result ->
            item {
                Text(
                    text = if (result == app.gauge.wear.capture.RetroCaptureResult.SAVED) "Saved" else "Couldn't save — try again",
                    style = MaterialTheme.typography.caption2,
                    color = if (result == app.gauge.wear.capture.RetroCaptureResult.SAVED) MaterialTheme.colors.primary else MaterialTheme.colors.error,
                )
            }
        }
        // Journal A/B: the toggle + its one-time consent step. Requires a paired watch (the
        // captures API rejects legacy callers), surfaced honestly rather than failing silently.
        item {
            Chip(
                modifier = Modifier.fillMaxWidth(),
                label = { Text(if (journalOn) "📓 Journal — keep what I say · On" else "📓 Journal — keep what I say") },
                colors = if (journalOn) ChipDefaults.primaryChipColors() else ChipDefaults.secondaryChipColors(),
                onClick = {
                    journalNeedsPairing = false
                    if (journalOn) {
                        // OFF clears the stored consent too (session-long consent semantics).
                        app.gauge.wear.prefs.GaugePrefs.disableJournal(context)
                        journalOn = false
                        showJournalConsent = false
                        app.gauge.wear.journal.logJournalToggle(context, false)
                    } else if (!uiState.signedIn) {
                        journalNeedsPairing = true
                    } else {
                        showJournalConsent = true
                    }
                },
            )
        }
        item {
            Text(
                text = "uploads a few minutes at a time; only your voice is kept after processing",
                style = MaterialTheme.typography.caption2,
                color = MaterialTheme.colors.onSurfaceVariant,
                textAlign = TextAlign.Center,
            )
        }
        if (journalNeedsPairing) {
            item {
                Text(
                    text = "Pair your watch first — Journal uploads need a paired watch.",
                    style = MaterialTheme.typography.caption2,
                    color = MaterialTheme.colors.error,
                    textAlign = TextAlign.Center,
                )
            }
        }
        if (showJournalConsent) {
            item {
                Text(
                    text = "While the sentinel is on, Journal uploads a few minutes of your OWN audio at a time. After processing, only the stretches matching your enrolled voice are kept; raw uploads are deleted within 48 hours.",
                    style = MaterialTheme.typography.caption2,
                    textAlign = TextAlign.Center,
                )
            }
            item {
                Chip(
                    modifier = Modifier.fillMaxWidth(),
                    label = { Text("Confirm — turn Journal on") },
                    colors = ChipDefaults.primaryChipColors(),
                    onClick = {
                        showJournalConsent = false
                        // The wearer's tap on THIS chip is the one consent artifact — stored
                        // with its timestamp, cleared when the toggle goes off (see
                        // GaugePrefs.enableJournal/disableJournal). Cancel below stores nothing.
                        app.gauge.wear.prefs.GaugePrefs.enableJournal(
                            context, app.gauge.wear.journal.journalNowIso(),
                        )
                        journalOn = true
                        app.gauge.wear.journal.logJournalToggle(context, true)
                    },
                )
            }
            item {
                Chip(
                    modifier = Modifier.fillMaxWidth(),
                    label = { Text("Cancel") },
                    colors = ChipDefaults.secondaryChipColors(),
                    onClick = { showJournalConsent = false },
                )
            }
        }
        item {
            Chip(
                modifier = Modifier.fillMaxWidth(),
                label = { Text("⚙ Settings") },
                colors = ChipDefaults.secondaryChipColors(),
                onClick = onOpenSettingsScreen,
            )
        }
        item {
            Chip(
                modifier = Modifier.fillMaxWidth(),
                label = { Text(uiState.mode.displayLabel()) },
                colors = ChipDefaults.secondaryChipColors(),
                onClick = onOpenModeScreen,
            )
        }
        if (!uiState.signedIn) {
            item {
                Chip(
                    modifier = Modifier.fillMaxWidth(),
                    label = { Text("Sign in") },
                    colors = ChipDefaults.secondaryChipColors(),
                    onClick = onOpenSignInScreen,
                )
            }
        }
    }
}

private fun hasRecordAudioPermission(context: Context): Boolean =
    ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
        PackageManager.PERMISSION_GRANTED

/**
 * Whether a RECORD_AUDIO permission-result callback should go on to arm the sentinel (review fix,
 * post-Task-11): [GlanceScreen] shares one launcher between the proactive first-open permission
 * prime ([LaunchedEffect]) and the explicit ring tap, so the callback alone can't tell which one
 * triggered it — [explicitArmRequest] threads that through. Arming is only correct when BOTH the
 * permission was actually granted AND the request was an explicit tap on the ring (the app's one
 * deliberate on/off control): a first-open prime grant must leave the sentinel Off, never silently
 * start the mic-recording foreground service just because the wearer tapped "Allow" on a system
 * dialog they didn't associate with turning anything on.
 */
internal fun shouldArmOnGrant(granted: Boolean, explicitArmRequest: Boolean): Boolean =
    granted && explicitArmRequest

/** POST_NOTIFICATIONS is only a runtime permission from API 33 (Tiramisu) on; requesting it below
 * that is a harmless no-op on-device, but there's no reason to ask for something the platform
 * doesn't gate. */
private fun runtimePermissions(): Array<String> =
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
        arrayOf(Manifest.permission.RECORD_AUDIO, Manifest.permission.POST_NOTIFICATIONS)
    } else {
        arrayOf(Manifest.permission.RECORD_AUDIO)
    }

/**
 * The central ring: now a tappable on/off toggle (Task 11), not just a status readout. Sweep
 * angle reflects [GlanceUi.meterFraction] when a meter reading is available (a partial arc — how
 * close the wearer is to their own threshold), full circle otherwise (no reading yet, or an
 * episode in progress where the nudge-level color matters more than the meter position).
 *
 * Center text: precomputed by [GaugeViewModel] as [GlanceUi.centerText] — always the live meter
 * (v0.2.4: calm score is gone from the product), so this composable needs no fallback logic of
 * its own.
 */
@Composable
private fun GlanceRing(uiState: GlanceUi, onToggle: () -> Unit) {
    val sweepAngle = uiState.meterFraction?.let { it * 360f } ?: 360f
    val statusColor = Color(if (uiState.isOn) STATUS_ON_GREEN else STATUS_OFF_RED)

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .aspectRatio(1f)
            .padding(8.dp)
            .clickable(onClick = onToggle),
        contentAlignment = Alignment.Center,
    ) {
        Canvas(modifier = Modifier.fillMaxWidth().aspectRatio(1f)) {
            val statusStroke = size.minDimension * 0.025f
            // v0.3.1: thin full-circle status perimeter OUTSIDE the meter arc — on/off at a glance.
            drawCircle(
                color = statusColor,
                radius = (size.minDimension - statusStroke) / 2f,
                style = Stroke(width = statusStroke),
            )
            val inset = statusStroke * 2.5f
            val strokeWidth = size.minDimension * 0.08f
            drawArc(
                color = Color(uiState.ringColor),
                startAngle = -90f,
                sweepAngle = sweepAngle,
                useCenter = false,
                topLeft = Offset(inset, inset),
                size = Size(size.width - 2 * inset, size.height - 2 * inset),
                style = Stroke(width = strokeWidth, cap = StrokeCap.Round),
            )
        }
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(
                text = uiState.centerText,
                style = if (uiState.showEpisode) MaterialTheme.typography.display1 else MaterialTheme.typography.title2,
                textAlign = TextAlign.Center,
            )
            if (uiState.centerText != uiState.armedLabel) {
                Text(text = uiState.armedLabel, style = MaterialTheme.typography.caption2, color = statusColor)
            }
        }
    }
}

/**
 * The sparkline-first center (Addendum 2, the new default): a large live trace on the honest
 * fixed scale (values arrive pre-normalized — see GaugeViewModel.resolveSparkline), with the
 * threshold line drawn at its own fraction and the center number overlaid. Purely presentational,
 * and the SAME tap-to-toggle surface as [GlanceRing] — the center tap gesture is the arm/disarm
 * toggle regardless of which visualization is showing.
 */
@Composable
private fun CenterSparkline(uiState: GlanceUi, onToggle: () -> Unit) {
    val statusColor = Color(if (uiState.isOn) STATUS_ON_GREEN else STATUS_OFF_RED)

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(110.dp)
            .padding(8.dp)
            .clickable(onClick = onToggle),
        contentAlignment = Alignment.Center,
    ) {
        Canvas(
            modifier = Modifier
                .fillMaxWidth()
                .height(96.dp),
        ) {
            // v0.3.1 status perimeter: a dome over the live area + a baseline under it — the wearer's
            // requested "top of a circle and a line underneath" frame. Green = on, red = off.
            drawArc(
                color = statusColor,
                startAngle = 180f,
                sweepAngle = 180f,
                useCenter = false,
                topLeft = Offset(0f, 0f),
                size = Size(size.width, size.height * 0.9f),
                style = Stroke(width = 3f),
            )
            drawLine(
                color = statusColor,
                start = Offset(0f, size.height),
                end = Offset(size.width, size.height),
                strokeWidth = 3f,
            )

            // No series -> no line and no threshold: an honest blank (e.g. no baseline yet), never
            // an auto-normalized fake. The overlaid centerText still says Off/On/value.
            if (uiState.sparklineNorm.isEmpty()) return@Canvas
            val thresholdY = size.height - uiState.sparklineThresholdFrac * size.height
            drawLine(
                color = Color(0x66FFFFFF),
                start = Offset(0f, thresholdY),
                end = Offset(size.width, thresholdY),
                strokeWidth = 1.5f,
            )
            val values = uiState.sparklineNorm.takeLast(SPARKLINE_WINDOW)
            if (values.size < 2) return@Canvas
            val stepX = size.width / (values.size - 1)
            val path = Path()
            values.forEachIndexed { index, v ->
                val x = index * stepX
                val y = size.height - v * size.height
                if (index == 0) path.moveTo(x, y) else path.lineTo(x, y)
            }
            drawPath(path, color = Color(uiState.ringColor), style = Stroke(width = 4f))
        }
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(
                text = uiState.centerText,
                style = MaterialTheme.typography.title2,
                textAlign = TextAlign.Center,
            )
            if (uiState.centerText != uiState.armedLabel) {
                Text(text = uiState.armedLabel, style = MaterialTheme.typography.caption2, color = statusColor)
            }
        }
    }
}

private fun previewEpisodeState() = GlanceUi(
    armedLabel = "Episode",
    ringColor = 0xFFEF6C00,
    vectorIcon = "📢",
    sparklineNorm = listOf(0.2f, 0.35f, 0.3f, 0.5f, 0.8f, 0.6f, 0.45f, 0.7f, 0.95f, 0.75f, 0.5f, 0.35f, 0.4f, 0.6f, 0.85f),
    sparklineThresholdFrac = 0.5f,
    mode = Mode.STANDARD,
    showEpisode = true,
    isOn = true,
    meterValue = -18.0,
    meterFraction = 0.8f,
    meterOver = true,
    meterHasThreshold = true,
    signal = SignalKind.VOLUME,
    signalHasReading = true,
    centerText = "-18.0dB",
    signalStatus = null,
    centerDisplay = CenterDisplay.SPARKLINE,
)

private fun previewOnState() = GlanceUi(
    armedLabel = "On",
    ringColor = 0xFF2E7D32,
    vectorIcon = null,
    sparklineNorm = listOf(0.1f, 0.15f, 0.1f, 0.2f, 0.15f, 0.12f),
    sparklineThresholdFrac = 0.5f,
    mode = Mode.STANDARD,
    showEpisode = false,
    isOn = true,
    meterValue = -32.4,
    meterFraction = 0.2f,
    meterOver = false,
    meterHasThreshold = true,
    signal = SignalKind.VOLUME,
    signalHasReading = true,
    centerText = "-32.4dB",
    // P4-10: not honest for VOLUME (Health Services is HR-only), but previews the layout with a
    // status caption present — see this function's call site KDoc.
    signalStatus = "off-body — wear snug",
    centerDisplay = CenterDisplay.SPARKLINE,
)

@Preview(name = "Round · episode", device = Devices.WEAR_OS_LARGE_ROUND, showBackground = true)
@Composable
private fun GlanceScreenRoundEpisodePreview() {
    MaterialTheme {
        GlanceScreen(
            uiState = previewEpisodeState(),
            onOpenModeScreen = {},
            onOpenSignalScreen = {},
            onOpenSettingsScreen = {},
            onOpenSignInScreen = {},
            onOpenCouplesScreen = {},
        )
    }
}

@Preview(name = "Square · episode", device = Devices.WEAR_OS_SQUARE, showBackground = true)
@Composable
private fun GlanceScreenSquareEpisodePreview() {
    MaterialTheme {
        GlanceScreen(
            uiState = previewEpisodeState(),
            onOpenModeScreen = {},
            onOpenSignalScreen = {},
            onOpenSettingsScreen = {},
            onOpenSignInScreen = {},
            onOpenCouplesScreen = {},
        )
    }
}

@Preview(name = "Round · on, live meter", device = Devices.WEAR_OS_LARGE_ROUND, showBackground = true)
@Composable
private fun GlanceScreenRoundOnPreview() {
    MaterialTheme {
        GlanceScreen(
            uiState = previewOnState(),
            onOpenModeScreen = {},
            onOpenSignalScreen = {},
            onOpenSettingsScreen = {},
            onOpenSignInScreen = {},
            onOpenCouplesScreen = {},
        )
    }
}

@Preview(name = "Round · dial view", device = Devices.WEAR_OS_LARGE_ROUND, showBackground = true)
@Composable
private fun GlanceScreenRoundDialPreview() {
    MaterialTheme {
        GlanceScreen(
            uiState = previewOnState().copy(centerDisplay = CenterDisplay.DIAL),
            onOpenModeScreen = {},
            onOpenSignalScreen = {},
            onOpenSettingsScreen = {},
            onOpenSignInScreen = {},
            onOpenCouplesScreen = {},
        )
    }
}
