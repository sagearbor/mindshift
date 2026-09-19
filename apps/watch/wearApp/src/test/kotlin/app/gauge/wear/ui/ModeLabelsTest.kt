package app.gauge.wear.ui

import app.gauge.shared.sentinel.Mode
import kotlin.test.Test
import kotlin.test.assertEquals

/** Tier B: the mode picker's Companion chip wording, and a guard that every mode keeps a
 * distinct human label (the `when` in [Mode.displayLabel] is exhaustive, so a new enum entry
 * can't compile without one — this pins that none of them collide). */
class ModeLabelsTest {
    @Test fun companionChipSaysThePhoneListens() {
        assertEquals("Companion — phone listens", Mode.COMPANION.displayLabel())
    }

    @Test fun everyModeHasADistinctLabel() {
        val labels = Mode.values().map { it.displayLabel() }
        assertEquals(labels.size, labels.toSet().size, "mode labels must be distinct: $labels")
    }
}
