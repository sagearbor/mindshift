package app.gauge.wear.ui

import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Review fix (post-Task-11): [shouldArmOnGrant] is the pure decision [GlanceScreen]'s shared
 * RECORD_AUDIO permission-result callback delegates to, so it's unit-testable on its own without
 * pulling in Compose — see its own KDoc for why a grant alone must never be sufficient to arm.
 */
class GlanceScreenPermissionTest {

    @Test
    fun grantedAndExplicitTapArms() {
        assertTrue(shouldArmOnGrant(granted = true, explicitArmRequest = true))
    }

    @Test
    fun grantedButPrimingOnlyDoesNotArm() {
        assertFalse(shouldArmOnGrant(granted = true, explicitArmRequest = false))
    }

    @Test
    fun deniedAndExplicitTapDoesNotArm() {
        assertFalse(shouldArmOnGrant(granted = false, explicitArmRequest = true))
    }

    @Test
    fun deniedAndPrimingDoesNotArm() {
        assertFalse(shouldArmOnGrant(granted = false, explicitArmRequest = false))
    }
}
