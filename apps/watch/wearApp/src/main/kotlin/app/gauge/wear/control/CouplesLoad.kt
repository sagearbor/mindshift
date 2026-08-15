package app.gauge.wear.control

import app.gauge.shared.Group
import app.gauge.shared.GroupStanding
import app.gauge.shared.MemberStanding
import app.gauge.wear.net.ApiResult

/**
 * Task 9 addition (coordinator-directed): the pure network-result-triage logic behind
 * [app.gauge.wear.ui.CouplesScreen]'s fetch, extracted the same way Task 7's `PairingPoller` was
 * extracted from `SignInScreen` — Compose shells in this app are compile+lint gated only, so the
 * actual honest-state decisions live here, independently unit-tested, rather than inline in a
 * `LaunchedEffect`.
 */
sealed interface CouplesLoadOutcome {
    /** A real, presentation-ready card — see [buildCouplesCardUi]. */
    data class Loaded(val ui: CouplesCardUi) : CouplesLoadOutcome

    /** Either never signed in, or [myStandingResult] came back 401 (the device token was
     * revoked/expired server-side) — both resolve to the SAME action: sign out locally and route
     * back to [app.gauge.wear.ui.SignInScreen]. Closes the deferred wave-c minor "no task acts on
     * a 401" — the watch must know it's signed out, not silently retry forever against a token
     * the server will never accept again. */
    object NeedsSignIn : CouplesLoadOutcome

    /** Any other failure (a `null`-code transport failure, or a non-401 server error) — an honest,
     * retryable state, never conflated with "you're signed out." */
    object NetworkFailed : CouplesLoadOutcome
}

/**
 * Resolves [myStandingResult] (the primary, gating call — `GET /me/standing`) plus the
 * best-effort `pairGroup`/`groupStandingResult` (soft-degrading: any failure there just means
 * [buildCouplesCardUi] falls back to a solo card, never fatal — the wearer didn't ask this screen
 * for the partner-specific calls, only the primary one) into one honest [CouplesLoadOutcome].
 *
 * Per [app.gauge.wear.auth.DevicePairingClient]/[app.gauge.wear.net.WatchApiClient]'s own pinned
 * contract: a `null` [ApiResult.Failure.code] means transport failure only, never a server-issued
 * rejection — so it always resolves to [CouplesLoadOutcome.NetworkFailed], never
 * [CouplesLoadOutcome.NeedsSignIn]. Only an explicit server-issued `401` means "this token is no
 * longer valid."
 */
fun resolveCouplesLoad(
    myStandingResult: ApiResult<MemberStanding>,
    pairGroup: Group?,
    groupStandingResult: ApiResult<GroupStanding>?,
    viewerAccountId: String,
): CouplesLoadOutcome {
    if (myStandingResult is ApiResult.Failure) {
        return if (myStandingResult.code == 401) {
            CouplesLoadOutcome.NeedsSignIn
        } else {
            CouplesLoadOutcome.NetworkFailed
        }
    }
    val myStanding = (myStandingResult as ApiResult.Ok).value
    val groupStanding = (groupStandingResult as? ApiResult.Ok)?.value
    return CouplesLoadOutcome.Loaded(buildCouplesCardUi(myStanding, pairGroup, groupStanding, viewerAccountId))
}
