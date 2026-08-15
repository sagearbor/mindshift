package app.gauge.wear.control

import app.gauge.shared.Group
import app.gauge.shared.GroupMember
import app.gauge.shared.GroupStanding
import app.gauge.shared.MemberStanding
import app.gauge.shared.PeriodStats
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull

class CouplesStandingTest {

    private fun stats(episodes: Int, calm: Double?) = PeriodStats(episodes, calm, 0, 0)

    @Test
    fun noGroupShowsSoloStandingOnly() {
        val my = MemberStanding(
            accountId = "me", current = stats(2, 70.0), prior = stats(1, 60.0),
            deltaVsSelf = 10.0, improving = true,
        )
        val ui = buildCouplesCardUi(myStanding = my, group = null, groupStanding = null, viewerAccountId = "me")
        assertFalse(ui.hasPartner)
        assertEquals("Your calm score is 10.0 higher than your own previous period.", ui.headline)
        assertNull(ui.aheadNote)
    }

    @Test
    fun notEnoughHistoryYet() {
        val my = MemberStanding(accountId = "me", current = stats(0, null), prior = stats(0, null))
        val ui = buildCouplesCardUi(my, null, null, "me")
        assertEquals("Not enough history yet — this fills in after a second period with episodes.", ui.headline)
    }

    @Test
    fun bothImprovingHeadlineWinsOverIndividualDeltas() {
        val group = Group(
            id = "g1", kind = "pair", createdBy = "me", createdAt = "2026-08-01T00:00:00Z",
            members = listOf(GroupMember("me", "2026-08-01T00:00:00Z"), GroupMember("partner", "2026-08-01T00:00:00Z")),
        )
        val standing = GroupStanding(
            groupId = "g1", periodDays = 7, periodStart = "2026-07-28T00:00:00Z", periodEnd = "2026-08-04T00:00:00Z",
            bothImproving = true, ahead = "me",
            members = listOf(
                MemberStanding("me", "You", stats(3, 80.0), stats(2, 70.0), 10.0, true),
                MemberStanding("partner", "Partner", stats(3, 75.0), stats(2, 65.0), 10.0, true),
            ),
        )
        val ui = buildCouplesCardUi(standing.members[0], group, standing, "me")
        assertEquals("You both improved on your own previous periods.", ui.headline)
        assertEquals("You're slightly ahead this period — a snapshot, not a score.", ui.aheadNote)
    }

    @Test
    fun aheadNoteNamesThePartnerInThirdPerson() {
        val group = Group(
            id = "g1", kind = "pair", createdBy = "me", createdAt = "2026-08-01T00:00:00Z",
            members = listOf(GroupMember("me", "2026-08-01T00:00:00Z"), GroupMember("partner", "2026-08-01T00:00:00Z")),
        )
        val standing = GroupStanding(
            groupId = "g1", periodDays = 7, periodStart = "2026-07-28T00:00:00Z", periodEnd = "2026-08-04T00:00:00Z",
            bothImproving = false, ahead = "partner",
            members = listOf(
                MemberStanding("me", "You", stats(1, 50.0), stats(1, 55.0), -5.0, false),
                MemberStanding("partner", "Alex", stats(1, 60.0), stats(1, 50.0), 10.0, true),
            ),
        )
        val ui = buildCouplesCardUi(standing.members[0], group, standing, "me")
        // Review fix (T8 round 2): pin the headline too, not just aheadNote -- this is the
        // hasPartner=true/bothImproving=false simplification branch (falls through to the
        // viewer's own standingHeadline, per buildCouplesCardUi's own KDoc on why it's simpler
        // than the dashboard's two-sentence pairHeadline) -- previously only hand-verified.
        assertEquals("Your calm score is 5.0 lower than your own previous period.", ui.headline)
        assertEquals("Alex is slightly ahead this period — a snapshot, not a score.", ui.aheadNote)
    }

    @Test
    fun flatDeltaReadsAsLevel() {
        val my = MemberStanding("me", current = stats(2, 70.0), prior = stats(2, 70.0), deltaVsSelf = 0.0, improving = false)
        val ui = buildCouplesCardUi(my, null, null, "me")
        assertEquals("Your calm score is level with your own previous period.", ui.headline)
    }

    @Test
    fun nullMyStandingShowsNotEnoughHistory() {
        // Review fix (T8 round 2): buildCouplesCardUi's myStanding == null branch has no direct
        // caller yet within this plan (Task 9's CouplesScreen always unwraps a successful
        // ApiResult.Ok before calling this function -- see CouplesLoad.kt), but the parameter is
        // deliberately nullable for a future direct-construction caller (e.g. a Tile/complication
        // surface reading a cached/never-fetched standing without a fresh API round trip --
        // explicitly descoped from this plan, not forgotten). Exercised directly here so the
        // function's own documented null contract is actually pinned, not just asserted in prose.
        val ui = buildCouplesCardUi(myStanding = null, group = null, groupStanding = null, viewerAccountId = "me")
        assertFalse(ui.hasPartner)
        assertEquals("Not enough history yet — this fills in after a second period with episodes.", ui.headline)
        assertNull(ui.aheadNote)
    }
}
