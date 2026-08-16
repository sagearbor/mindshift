package app.gauge.wear.ui

import android.content.Context
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.wear.compose.material.Chip
import androidx.wear.compose.material.ChipDefaults
import androidx.wear.compose.material.MaterialTheme
import androidx.wear.compose.material.ScalingLazyColumn
import androidx.wear.compose.material.Text
import androidx.wear.compose.material.rememberScalingLazyListState
import app.gauge.wear.BuildConfig
import app.gauge.wear.auth.AccountBus
import app.gauge.wear.auth.AccountPrefs
import app.gauge.wear.auth.DevicePairingClient
import app.gauge.wear.auth.PairingPollOutcome
import app.gauge.wear.auth.PairingPoller
import app.gauge.wear.net.WatchApiClient
import app.gauge.wear.telemetry.Telemetry
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

private const val POLL_INTERVAL_MS = 3_000L

/**
 * Short-code pairing screen (Wave C Task 7) — see Task 5's KDoc for why this
 * shape exists instead of on-wrist OAuth. Purely a one-shot flow driven from
 * this screen's own [LaunchedEffect]: no background service, no polling once
 * the screen isn't visible (battery discipline — polling only happens while
 * this exact screen is on top).
 *
 * T5-review fix (Important, controller-resolved): the poll loop below is a thin
 * driver — real `delay`/[DevicePairingClient] I/O only — while every actual decision
 * (keep polling / signed in / give up, and why) is made by [PairingPoller], a pure,
 * independently unit-tested state machine with its own give-up bound (see that class's
 * KDoc) so this screen can never hang on a non-terminal state forever, even under a
 * persistent network outage where every [DevicePairingClient.poll] call returns `null`.
 * Compose shells in this app are compile+lint gated only (no emulator/device in-loop —
 * see CLAUDE.md); extracting the state machine is what makes this give-up guarantee
 * actually testable rather than merely asserted in a comment.
 *
 * Display-text fix (found while wiring [PairingPoller] in): the text line below reads
 * `errorText ?: code?.let {...} ?: "Getting a code…"` — `errorText` deliberately checked
 * FIRST, not last. A version that checks `code` first (as if `code?.let{...} ?: errorText`)
 * would never show an error once a code had already been obtained, since `code?.let{}`
 * always short-circuits to a non-null String the moment `code` itself is non-null — so a
 * later `errorText` (e.g. "Code expired") would silently never render, leaving the wearer
 * staring at a stale "Enter XXXXX..." prompt with no indication anything had gone wrong.
 * That would defeat the give-up bound above just as thoroughly as an unbounded loop would.
 */
@Composable
fun SignInScreen(onSignedIn: () -> Unit, modifier: Modifier = Modifier) {
    val context = LocalContext.current
    var code by remember { mutableStateOf<String?>(null) }
    var errorText by remember { mutableStateOf<String?>(null) }

    LaunchedEffect(Unit) {
        val client = DevicePairingClient(
            baseUrl = BuildConfig.GAUGE_API_BASE,
            onSwallowedError = { runCatching { Telemetry.log("warn", "DevicePairing", it) } },
        )
        val start = client.start()
        if (start == null) {
            errorText = "Couldn't reach the server. Try again."
            // v0.3.1 final-review fix: a terminal sign-in failure must flush the diagnostic
            // ring -- the error event is otherwise trapped in memory and dies with the process.
            runCatching { Telemetry.flushAsync() }
            return@LaunchedEffect
        }
        code = start.code
        val poller = PairingPoller(nowMs = { System.currentTimeMillis() })
        while (true) {
            delay(POLL_INTERVAL_MS)
            val status = client.poll(start.pairingId)
            when (val outcome = poller.onPollResult(status)) {
                PairingPollOutcome.KeepPolling -> Unit
                is PairingPollOutcome.SignedIn -> {
                    val deviceToken = outcome.deviceToken
                    AccountPrefs.setSignedIn(context, outcome.accountId, deviceToken)
                    AccountBus.publish(true)
                    val api = WatchApiClient(baseUrl = BuildConfig.GAUGE_API_BASE, deviceToken = { deviceToken })
                    CoroutineScope(Dispatchers.IO).launch {
                        api.claimLegacy() // ApiResult ignored here — a failed claim just means
                        // the pre-sign-in episodes stay attributed to "default"; nothing crashes
                        // or blocks on it, and it's safe to retry (server-side no-ops on repeat).
                    }
                    onSignedIn()
                    return@LaunchedEffect
                }
                is PairingPollOutcome.Failed -> {
                    errorText = outcome.message
                    // see the matching flush above
                    runCatching { Telemetry.flushAsync() }
                    return@LaunchedEffect
                }
            }
        }
    }

    val listState = rememberScalingLazyListState()
    ScalingLazyColumn(modifier = modifier.fillMaxWidth(), state = listState) {
        item {
            Text(
                text = errorText ?: code?.let { "Enter $it on your phone or gauge.app/pair" }
                    ?: "Getting a code…",
                style = MaterialTheme.typography.body1,
            )
        }
        item {
            Chip(
                modifier = Modifier.fillMaxWidth(),
                label = { Text("Cancel") },
                colors = ChipDefaults.secondaryChipColors(),
                onClick = onSignedIn, // pops back without signing in; caller treats this as "done"
            )
        }
    }
}

fun currentAccountId(context: Context): String = AccountPrefs.accountId(context) ?: "default"
