package app.gauge.wear.ui

import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
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
import app.gauge.wear.control.CouplesCardUi
import app.gauge.wear.control.CouplesLoadOutcome
import app.gauge.wear.control.resolveCouplesLoad
import app.gauge.wear.net.ApiResult
import app.gauge.wear.net.WatchApiClient

/**
 * The couples/solo standing card (Wave C). Fetches exactly once per visit (or per explicit
 * [Chip] retry tap below — never a background poll, battery discipline) via [LaunchedEffect].
 *
 * Task 9 addition (coordinator-directed, closes the deferred wave-c minor "no task acts on a
 * 401"): every actual honest-state decision (loaded / needs sign-in / network-failed) is made by
 * [resolveCouplesLoad], a pure state machine independently unit-tested the same way Task 7's
 * `PairingPoller` is — this composable is a thin driver, same posture as [SignInScreen]. A never-
 * signed-in watch AND a watch whose device token was revoked/expired (a `401` from
 * [WatchApiClient.myStanding]) both resolve to the SAME [CouplesLoadOutcome.NeedsSignIn] action:
 * sign out locally (so nothing here keeps retrying against a token the server will never accept
 * again) and hand off to [onNeedsSignIn], which the caller wires to [SignInScreen]'s route — the
 * exact T7 flow, not a second one. Any OTHER failure (including a `null`-code transport failure)
 * resolves to [CouplesLoadOutcome.NetworkFailed] — an honest, retryable line, never mistaken for
 * "you're signed out."
 */
@Composable
fun CouplesScreen(onNeedsSignIn: () -> Unit, modifier: Modifier = Modifier) {
    val context = LocalContext.current
    var ui by remember { mutableStateOf<CouplesCardUi?>(null) }
    var loadFailed by remember { mutableStateOf(false) }
    // Bumped by the "Retry" chip below to force LaunchedEffect to re-run — an explicit,
    // wearer-initiated re-fetch, never a timer (battery discipline: no polling loop on a
    // glanceable card).
    var retryToken by remember { mutableIntStateOf(0) }

    LaunchedEffect(retryToken) {
        ui = null
        loadFailed = false
        val token = AccountPrefs.deviceToken(context)
        val accountId = AccountPrefs.accountId(context)
        if (token == null || accountId == null) {
            onNeedsSignIn()
            return@LaunchedEffect
        }
        val api = WatchApiClient(baseUrl = BuildConfig.GAUGE_API_BASE, deviceToken = { token })
        val myStandingResult = api.myStanding()
        val groupsResult = api.listGroups()
        val pairGroup = (groupsResult as? ApiResult.Ok)?.value?.find { it.kind == "pair" && it.members.size == 2 }
        val groupStandingResult = pairGroup?.let { g -> api.groupStanding(g.id) }
        when (val outcome = resolveCouplesLoad(myStandingResult, pairGroup, groupStandingResult, accountId)) {
            is CouplesLoadOutcome.Loaded -> ui = outcome.ui
            CouplesLoadOutcome.NeedsSignIn -> {
                AccountPrefs.signOut(context)
                AccountBus.publish(false)
                onNeedsSignIn()
            }
            CouplesLoadOutcome.NetworkFailed -> loadFailed = true
        }
    }

    val listState = rememberScalingLazyListState()
    ScalingLazyColumn(modifier = modifier.fillMaxWidth(), state = listState) {
        item {
            Text(
                text = when {
                    loadFailed -> "Couldn't load your standing right now."
                    ui == null -> "Loading…"
                    else -> ui!!.headline
                },
                style = MaterialTheme.typography.body1,
            )
        }
        ui?.aheadNote?.let { note ->
            item { Text(text = note, style = MaterialTheme.typography.caption2) }
        }
        if (loadFailed) {
            item {
                Chip(
                    modifier = Modifier.fillMaxWidth(),
                    label = { Text("Retry") },
                    colors = ChipDefaults.secondaryChipColors(),
                    onClick = { retryToken++ },
                )
            }
        }
    }
}
