package app.gauge.wear.auth

import app.gauge.shared.PairingStatus
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * T5-review fix (Important, controller-resolved): [PairingPoller] is the pure, independently
 * testable state machine [app.gauge.wear.ui.SignInScreen]'s poll loop is a thin driver for — see
 * that class's own KDoc. Every test here uses a hand-advanced fake clock (a plain mutable `Long`
 * closed over by [PairingPoller]'s `nowMs` lambda), never a real `delay`/sleep, so the give-up-bound
 * tests run in milliseconds of wall-clock time despite simulating up to 20 minutes of polling.
 *
 * Pinned server contract (controller ruling): every pairing lifecycle state — "pending", "claimed",
 * "expired" — arrives as a decoded, non-null [PairingStatus]; `null` from
 * [app.gauge.wear.auth.DevicePairingClient.poll] means transport failure ONLY, never a lifecycle
 * state. [PairingPoller] is built on exactly that distinction.
 */
class PairingPollerTest {

    @Test
    fun keepsPollingOnPending() {
        var now = 0L
        val poller = PairingPoller(nowMs = { now })
        val outcome = poller.onPollResult(PairingStatus(status = "pending"))
        assertEquals(PairingPollOutcome.KeepPolling, outcome)
    }

    @Test
    fun keepsPollingOnTransportFailureWithinTheBound() {
        var now = 0L
        val poller = PairingPoller(nowMs = { now })
        now += 300_000L // 5 minutes elapsed, well inside the 20-minute give-up bound
        val outcome = poller.onPollResult(null)
        assertEquals(PairingPollOutcome.KeepPolling, outcome)
    }

    @Test
    fun resolvesToSignedInOnClaimedWithBothFields() {
        val poller = PairingPoller(nowMs = { 0L })
        val outcome = poller.onPollResult(
            PairingStatus(status = "claimed", accountId = "acct1", deviceToken = "secret-token"),
        )
        assertEquals(PairingPollOutcome.SignedIn("acct1", "secret-token"), outcome)
    }

    @Test
    fun malformedClaimedMissingFieldsFailsHonestlyRatherThanHangingSilently() {
        // Contract violation: "claimed" should always carry both fields. The brief's own literal
        // code silently stopped polling with no feedback in this case (return@LaunchedEffect with
        // neither onSignedIn() nor errorText set) -- an even worse outcome than looping forever,
        // since the screen would sit frozen showing the code with zero indication anything is
        // wrong. This state machine never produces that silent-hang outcome.
        val poller = PairingPoller(nowMs = { 0L })
        val outcome = poller.onPollResult(PairingStatus(status = "claimed"))
        assertTrue(outcome is PairingPollOutcome.Failed)
    }

    @Test
    fun resolvesToFailedOnServerReportedExpired() {
        val poller = PairingPoller(nowMs = { 0L })
        val outcome = poller.onPollResult(PairingStatus(status = "expired"))
        assertTrue(outcome is PairingPollOutcome.Failed)
    }

    @Test
    fun givesUpAfterMaxDurationOfRepeatedTransportFailures() {
        // The residual case the T5 review flagged: a persistent network outage where every poll
        // returns null (transport failure) forever. Without an independent bound, this would spin
        // indefinitely -- the give-up bound is what guarantees a terminal state regardless.
        var now = 0L
        val poller = PairingPoller(nowMs = { now }, maxDurationMs = 1_200_000L) // 20 minutes
        var outcome: PairingPollOutcome = PairingPollOutcome.KeepPolling
        var iterations = 0
        while (outcome == PairingPollOutcome.KeepPolling && iterations < 1000) {
            now += 3_000L // one POLL_INTERVAL_MS tick per iteration -- fake clock, no real sleep
            outcome = poller.onPollResult(null)
            iterations++
        }
        assertTrue(iterations < 1000, "poller never gave up within the bounded iteration cap")
        assertTrue(outcome is PairingPollOutcome.Failed)
        // 1,200,000ms / 3,000ms per tick = 400 ticks to cross the bound.
        assertTrue(iterations in 395..405, "expected ~400 iterations to cross the bound, got $iterations")
    }

    @Test
    fun givesUpAfterMaxDurationOfRepeatedPending() {
        // Same backstop, but for a server that's genuinely up and honestly answering "pending"
        // every time -- past 2x the documented ~10min code TTL, the code must actually be dead
        // server-side even if this client never sees an explicit "expired" (e.g. the server's own
        // status endpoint is degraded but not fully unreachable).
        var now = 0L
        val poller = PairingPoller(nowMs = { now }, maxDurationMs = 1_200_000L)
        var outcome: PairingPollOutcome = PairingPollOutcome.KeepPolling
        var iterations = 0
        while (outcome == PairingPollOutcome.KeepPolling && iterations < 1000) {
            now += 3_000L
            outcome = poller.onPollResult(PairingStatus(status = "pending"))
            iterations++
        }
        assertTrue(outcome is PairingPollOutcome.Failed)
    }

    @Test
    fun aGenuineClaimedResultWinsEvenAtOrPastTheBoundary() {
        // A real success must never be discarded just because it happened to arrive on the exact
        // tick the give-up bound would otherwise fire -- terminal server signals always take
        // priority over the client-side time-based backstop.
        val poller = PairingPoller(nowMs = { 1_200_000L }, maxDurationMs = 1_200_000L)
        val outcome = poller.onPollResult(
            PairingStatus(status = "claimed", accountId = "acct1", deviceToken = "secret-token"),
        )
        assertEquals(PairingPollOutcome.SignedIn("acct1", "secret-token"), outcome)
    }

    @Test
    fun defaultBoundIsTwiceTheDocumentedTenMinuteCodeTtl() {
        // Pins the actual default constant against the plan's documented server-side TTL
        // ("TTL ~10 min", Open Questions) -- 2x = 20 minutes = 1,200,000ms.
        assertEquals(1_200_000L, PairingPoller.DEFAULT_MAX_POLLING_DURATION_MS)
    }
}
