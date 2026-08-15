package app.gauge.wear.control

import app.gauge.shared.Group
import app.gauge.shared.GroupMember
import app.gauge.shared.GroupStanding
import app.gauge.shared.MemberStanding
import app.gauge.shared.PeriodStats
import app.gauge.wear.net.ApiResult
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Task 9 addition (coordinator-directed): the pure network-result-triage logic behind
 * [app.gauge.wear.ui.CouplesScreen]'s fetch, extracted the same way Task 7's `PairingPoller` was
 * extracted from `SignInScreen` — Compose shells in this app are compile+lint gated only, so the
 * actual honest-state decisions (not-signed-in vs. revoked-token vs. network-fail vs. loaded) live
 * here where they're independently unit-testable, not buried in a `LaunchedEffect`.
 *
 * Closes the deferred wave-c minor "no task acts on a 401": a 401 from [WatchApiClient.myStanding]
 * means the device token was revoked/expired server-side — this resolves to the SAME
 * [CouplesLoadOutcome.NeedsSignIn] outcome as never having signed in at all, so the caller signs
 * out locally and routes back to [app.gauge.wear.ui.SignInScreen], the same T7 flow. Every OTHER
 * failure (including a `null`-code transport failure) resolves to [CouplesLoadOutcome.NetworkFailed]
 * — a distinct, honest, retryable state, never conflated with "you're signed out."
 */
class CouplesLoadTest {

    private fun stats(episodes: Int, calm: Double?) = PeriodStats(episodes, calm, 0, 0)

    private fun standing(accountId: String = "me") = MemberStanding(
        accountId = accountId, current = stats(2, 70.0), prior = stats(1, 60.0),
        deltaVsSelf = 10.0, improving = true,
    )

    private fun pairGroup() = Group(
        id = "g1", kind = "pair", createdBy = "me", createdAt = "2026-08-04T00:00:00Z",
        members = listOf(GroupMember("me", "2026-08-04T00:00:00Z"), GroupMember("partner", "2026-08-04T00:00:00Z")),
    )

    @Test
    fun revokedTokenResolvesToNeedsSignIn() {
        val outcome = resolveCouplesLoad(
            myStandingResult = ApiResult.Failure(401, "not signed in"),
            pairGroup = null,
            groupStandingResult = null,
            viewerAccountId = "me",
        )
        assertEquals(CouplesLoadOutcome.NeedsSignIn, outcome)
    }

    @Test
    fun otherFailureCodesResolveToNetworkFailedNeverSignIn() {
        val outcome = resolveCouplesLoad(
            myStandingResult = ApiResult.Failure(503, "storage not configured"),
            pairGroup = null,
            groupStandingResult = null,
            viewerAccountId = "me",
        )
        assertEquals(CouplesLoadOutcome.NetworkFailed, outcome)
    }

    @Test
    fun transportFailureWithNoCodeResolvesToNetworkFailed() {
        // DevicePairingClient/WatchApiClient's own contract: a null-code Failure means transport
        // trouble (connection refused, timeout), not a server-issued rejection -- must never be
        // mistaken for "you're signed out."
        val outcome = resolveCouplesLoad(
            myStandingResult = ApiResult.Failure(null, "connection refused"),
            pairGroup = null,
            groupStandingResult = null,
            viewerAccountId = "me",
        )
        assertEquals(CouplesLoadOutcome.NetworkFailed, outcome)
    }

    @Test
    fun successWithNoGroupBuildsSoloCard() {
        val outcome = resolveCouplesLoad(
            myStandingResult = ApiResult.Ok(standing()),
            pairGroup = null,
            groupStandingResult = null,
            viewerAccountId = "me",
        )
        assertTrue(outcome is CouplesLoadOutcome.Loaded)
        assertFalse((outcome as CouplesLoadOutcome.Loaded).ui.hasPartner)
    }

    @Test
    fun groupStandingFailureDegradesSoftlyToSoloCardNeverFatal() {
        // The primary call (myStanding) succeeded; the partner-specific call failed. Never fatal --
        // "never fabricate a standing" cuts both ways: the card falls back to solo, it doesn't
        // crash or redirect to sign-in over a call the wearer didn't even ask this screen for.
        val outcome = resolveCouplesLoad(
            myStandingResult = ApiResult.Ok(standing()),
            pairGroup = pairGroup(),
            groupStandingResult = ApiResult.Failure(503, "unavailable"),
            viewerAccountId = "me",
        )
        assertTrue(outcome is CouplesLoadOutcome.Loaded)
        assertFalse((outcome as CouplesLoadOutcome.Loaded).ui.hasPartner)
    }

    @Test
    fun successWithGroupStandingBuildsPairedCard() {
        val standing = GroupStanding(
            groupId = "g1", periodDays = 7, periodStart = "s", periodEnd = "e",
            bothImproving = true, ahead = "me",
            members = listOf(standing("me"), standing("partner")),
        )
        val outcome = resolveCouplesLoad(
            myStandingResult = ApiResult.Ok(standing("me")),
            pairGroup = pairGroup(),
            groupStandingResult = ApiResult.Ok(standing),
            viewerAccountId = "me",
        )
        assertTrue(outcome is CouplesLoadOutcome.Loaded)
        assertTrue((outcome as CouplesLoadOutcome.Loaded).ui.hasPartner)
    }
}
