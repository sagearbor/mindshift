package app.gauge.wear.control

import app.gauge.shared.Group
import app.gauge.shared.GroupStanding
import app.gauge.shared.MemberStanding
import java.util.Locale
import kotlin.math.abs

/** Presentation-ready state for the watch's couples/solo standing card (Wave C).
 * `hasPartner = false` means "no paired group yet" — the card still shows the
 * wearer's own solo standing (from `GET /me/standing`), never an empty screen. */
data class CouplesCardUi(
    val hasPartner: Boolean,
    val headline: String,
    val aheadNote: String?,
)

/**
 * Pure mapper — cribs its copy VERBATIM from `webApp/src/trends/standing.ts` +
 * `pair.ts` (read 2026-08-04; both already reviewed/pytest-backed on the server
 * side) so the watch's framing matches the dashboard's exactly: supportive,
 * self-relative, "a snapshot, not a score" — never a scoreboard. This function
 * makes no arithmetic decisions of its own (mirrors `standing.ts`'s own "never
 * recompute a delta" rule) — every number it reads was already decided by
 * `server/aggregates.py` and shipped on the wire.
 *
 * Deliberately simpler than the dashboard's own `pairHeadline` in the
 * not-both-improving case: the web version concatenates BOTH members' own
 * headlines (converting the partner's to third person); the watch shows only
 * the viewer's own headline — a 1.2" screen has no room for two sentences, and
 * `aheadNote` already covers the partner-relative framing. The underlying
 * sentence text itself (`standingHeadline`'s three branches) is unchanged from
 * the dashboard's pytest-tested wording either way.
 */
fun buildCouplesCardUi(
    myStanding: MemberStanding?,
    group: Group?,
    groupStanding: GroupStanding?,
    viewerAccountId: String,
): CouplesCardUi {
    // Unreached by this plan's sole caller today (CouplesLoad.kt's resolveCouplesLoad always
    // unwraps a successful ApiResult.Ok before calling this function) -- kept for any future
    // direct-construction caller (e.g. a Tile/complication reading a cached/never-fetched
    // standing, explicitly descoped from this plan) that can't guarantee a fresh myStanding.
    if (myStanding == null) {
        return CouplesCardUi(hasPartner = false, headline = "Not enough history yet — this fills in after a second period with episodes.", aheadNote = null)
    }
    val hasPartner = group != null && group.kind == "pair" && group.members.size == 2 && groupStanding != null
    if (!hasPartner) {
        return CouplesCardUi(hasPartner = false, headline = standingHeadline(myStanding), aheadNote = null)
    }
    val headline = if (groupStanding!!.bothImproving) {
        "You both improved on your own previous periods."
    } else {
        standingHeadline(myStanding)
    }
    val aheadNote = groupStanding.ahead?.let { aheadId ->
        if (aheadId == viewerAccountId) {
            "You're slightly ahead this period — a snapshot, not a score."
        } else {
            val partnerName = groupStanding.members.find { it.accountId == aheadId }?.displayName ?: "Your partner"
            "$partnerName is slightly ahead this period — a snapshot, not a score."
        }
    }
    return CouplesCardUi(hasPartner = true, headline = headline, aheadNote = aheadNote)
}

/** Direct port of `standing.ts`'s `standingHeadline` — same three branches, same wording. */
private fun standingHeadline(st: MemberStanding): String {
    val delta = st.deltaVsSelf
    if (delta == null) {
        return "Not enough history yet — this fills in after a second period with episodes."
    }
    if (delta == 0.0) {
        return "Your calm score is level with your own previous period."
    }
    // Locale.US explicitly (review fix, this task): the brief's literal String.format("%.1f", ...)
    // implicitly uses the default locale, a real lint finding (DefaultLocale) some locales' comma
    // decimal separator would make worse than cosmetic -- matches the established precedent for
    // this exact formatting already used by GaugeViewModel.formatMeterValue.
    val magnitude = String.format(Locale.US, "%.1f", abs(delta))
    return if (st.improving == true) {
        "Your calm score is $magnitude higher than your own previous period."
    } else {
        "Your calm score is $magnitude lower than your own previous period."
    }
}
