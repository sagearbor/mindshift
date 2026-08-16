package app.gauge.wear.auth

import app.gauge.shared.PairingStatus

/**
 * T5-review fix (Important, controller-resolved): the pure decision logic behind
 * [app.gauge.wear.ui.SignInScreen]'s poll loop, extracted so it's independently unit-testable
 * (Compose shells in this app are compile+lint gated only — see [app.gauge.wear.ui.SignInScreen]'s
 * own KDoc). [SignInScreen]'s `LaunchedEffect` is a thin driver: real `delay`/[DevicePairingClient]
 * calls live there; every actual decision (keep polling / signed in / give up, and why) is made
 * here.
 *
 * Pinned server contract (controller ruling, watch-side code built on this distinction): every
 * pairing lifecycle state — "pending"/"claimed"/"expired" — arrives as a decoded, non-null
 * [PairingStatus]. `null` from [DevicePairingClient.poll] means transport failure ONLY (couldn't
 * reach or parse the server), never a lifecycle state.
 *
 * Give-up bound (the T5-review fix itself): without an INDEPENDENT bound on total polling
 * duration, a persistent network outage — every poll returning `null` (transport failure) forever
 * — would keep [onPollResult] returning [PairingPollOutcome.KeepPolling] indefinitely, since a
 * transport failure alone never reaches the server long enough to hear an honest "expired". A
 * watch that can never reach the pairing endpoint would otherwise poll forever with no way out.
 * [maxDurationMs] (default [DEFAULT_MAX_POLLING_DURATION_MS], 2x the ~10min code TTL this plan's
 * Open Questions document for the server side) guarantees a terminal [PairingPollOutcome.Failed]
 * regardless of what (or whether) the server ever answers. A genuine terminal server signal
 * ("claimed" or "expired") always wins immediately over the bound, even if it happens to land on
 * the exact tick the bound would otherwise fire — the bound is a backstop for the absence of a
 * real answer, never a reason to discard one that arrives.
 */
class PairingPoller(
    private val nowMs: () -> Long,
    private val maxDurationMs: Long = DEFAULT_MAX_POLLING_DURATION_MS,
) {
    private val startedAtMs: Long = nowMs()

    /** Call once per poll attempt (including a `null` transport failure) — never call `delay`
     * or make the network call yourself here, that's the caller's job. */
    fun onPollResult(status: PairingStatus?): PairingPollOutcome {
        if (status != null) {
            when (status.status) {
                "claimed" -> {
                    val accountId = status.accountId
                    val deviceToken = status.deviceToken
                    return if (accountId != null && deviceToken != null) {
                        PairingPollOutcome.SignedIn(accountId, deviceToken)
                    } else {
                        // Contract violation (server should never send "claimed" without both
                        // fields) — an honest terminal failure, never a silent stuck screen.
                        PairingPollOutcome.Failed("Something went wrong finishing sign-in. Try again.")
                    }
                }
                "expired" -> return PairingPollOutcome.Failed("Code expired. Reopen this screen for a new one.")
                else -> Unit // "pending" (or any other value): fall through to the bound check below.
            }
        }
        // status == null (transport failure) or "pending": keep polling, UNLESS the independent
        // give-up bound has elapsed.
        if (nowMs() - startedAtMs >= maxDurationMs) {
            return PairingPollOutcome.Failed("Code expired — get a new one.")
        }
        return PairingPollOutcome.KeepPolling
    }

    companion object {
        /** 2x the ~10min code TTL this plan's Open Questions document for the server side (no
         * fixed value is specified in this task's own brief) — generous enough to never fire
         * during genuinely healthy polling (the server's own "expired" response fires first, well
         * before this), tight enough that a real network outage can't hang the screen forever. */
        const val DEFAULT_MAX_POLLING_DURATION_MS = 1_200_000L
    }
}

/** Every outcome [PairingPoller.onPollResult] can produce — see that class's own KDoc. */
sealed interface PairingPollOutcome {
    /** No terminal answer yet and the give-up bound hasn't elapsed — the caller should `delay`
     * and poll again. */
    object KeepPolling : PairingPollOutcome

    /** The server confirmed the code was claimed and minted a real device token. */
    data class SignedIn(val accountId: String, val deviceToken: String) : PairingPollOutcome

    /** A terminal, honest failure — the server said the code expired, the give-up bound elapsed,
     * or the server's "claimed" response was malformed. [message] is ready to show as-is. */
    data class Failed(val message: String) : PairingPollOutcome
}
