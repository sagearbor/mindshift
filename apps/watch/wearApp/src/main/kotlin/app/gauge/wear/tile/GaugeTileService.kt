package app.gauge.wear.tile

import android.os.Handler
import android.os.Looper
import androidx.wear.tiles.ActionBuilders
import androidx.wear.tiles.LayoutElementBuilders
import androidx.wear.tiles.ModifiersBuilders
import androidx.wear.tiles.RequestBuilders
import androidx.wear.tiles.TileBuilders
import androidx.wear.tiles.TileService
import androidx.wear.tiles.TimelineBuilders
import app.gauge.shared.sentinel.Mode
import app.gauge.shared.sentinel.SentinelState
import app.gauge.wear.control.ControllerStateBus
import app.gauge.wear.ui.displayLabel
import app.gauge.wear.ui.sendArm
import app.gauge.wear.ui.sendDisarm
import com.google.common.util.concurrent.ListenableFuture
import java.util.concurrent.Executor
import java.util.concurrent.TimeUnit

/**
 * Clickable ids [tileLayout] assigns to its one button, and that [GaugeTileService.onTileRequest]
 * reads back off [RequestBuilders.TileRequest]'s `state.lastClickableId` to know which action to
 * forward — see [GaugeTileService]'s own KDoc for the full click round-trip.
 */
internal const val CLICK_ID_ARM = "click_arm"
internal const val CLICK_ID_DISARM = "click_disarm"

/**
 * The three strings/ids [tileLayout] renders, factored out as a plain data class so they're
 * TDD-able on their own (see TileLayoutTest) without going through
 * [LayoutElementBuilders.LayoutElement.toLayoutElementProto] — that method is public and does
 * compile, but the generated proto message classes it returns
 * (`androidx.wear.protolayout.proto.LayoutElementProto`) live in `protolayout-proto`, which
 * `androidx.wear.protolayout:protolayout` (and therefore `androidx.wear.tiles:tiles`) pulls in as
 * a **runtime-only** transitive dependency, not a compile-time one — so calling
 * `toLayoutElementProto()` compiles fine in this file (main sourceSet compile classpath happens
 * to have it) but a JVM unit test that does the same fails with "Cannot access class ...
 * LayoutElementProto.LayoutElement" on the unit-test compile classpath. Per the task brief's own
 * documented fallback for exactly this case, [tileStrings] is the pure function under test
 * instead; [tileLayout] itself is exercised indirectly by `assembleDebug`/`lintDebug` (it has to
 * compile and build against the real protolayout builder API) but not by a dedicated unit test.
 */
data class TileStrings(
    val statusText: String,
    val buttonText: String,
    val clickId: String,
)

/**
 * Pure string/id builder for the swipe-right Gauge tile — see [TileStrings]'s KDoc for why this,
 * not [tileLayout] directly, is what TileLayoutTest exercises.
 *
 * "On · <mode>" when on / "Off" when not (same wording as
 * [app.gauge.wear.service.notificationText]'s ARMED case, minus the sentinel/streaming detail a
 * tile has no room for), a button labeled with the action it's *about to perform* ("Turn off"
 * while on, "Turn on" while not — never the current state), and that action's click id (unchanged
 * from the pre-Task-11 copy — only the text renders "On"/"Off" now), so
 * [GaugeTileService.onTileRequest] can read it straight back as an unambiguous command.
 */
fun tileStrings(armed: Boolean, mode: Mode): TileStrings = TileStrings(
    statusText = if (armed) "On · ${mode.displayLabel()}" else "Off",
    buttonText = if (armed) "Turn off" else "Turn on",
    clickId = if (armed) CLICK_ID_DISARM else CLICK_ID_ARM,
)

/**
 * Pure layout for the swipe-right Gauge tile, built from [tileStrings]. [GaugeTileService] just
 * calls this once per `onTileRequest` and wraps the result in a single-entry
 * [TimelineBuilders.Timeline]. Two elements only, mirroring
 * [app.gauge.wear.ui.GlanceScreen]'s own Arm/Disarm chip: a status Text and one clickable button
 * Text.
 */
fun tileLayout(armed: Boolean, mode: Mode): LayoutElementBuilders.LayoutElement {
    val strings = tileStrings(armed, mode)

    val status = LayoutElementBuilders.Text.Builder()
        .setText(strings.statusText)
        .build()

    val button = LayoutElementBuilders.Text.Builder()
        .setText(strings.buttonText)
        .setModifiers(
            ModifiersBuilders.Modifiers.Builder()
                .setClickable(
                    ModifiersBuilders.Clickable.Builder()
                        .setId(strings.clickId)
                        // LoadAction, not LaunchAction: nothing in the tiles API lets a Clickable
                        // start a foreground service directly, so the click round-trips back
                        // through onTileRequest instead of opening an activity — see
                        // GaugeTileService's KDoc.
                        .setOnClick(ActionBuilders.LoadAction.Builder().build())
                        .build(),
                )
                .build(),
        )
        .build()

    return LayoutElementBuilders.Column.Builder()
        .addContent(status)
        .addContent(button)
        .build()
}

/**
 * Swipe-right Gauge tile: arm/disarm + current mode, one tap away from the watch face. THIN by
 * design — [tileLayout] holds every layout/text decision; this class only does the Android
 * plumbing tiles require:
 *
 *  - [onTileRequest] reads the tapped button's id back off `requestParams.state.lastClickableId`
 *    (only ever non-null on the request that follows a [ActionBuilders.LoadAction] click) and
 *    forwards it to `SentinelService` via the exact same [app.gauge.wear.ui.sendArm] /
 *    [app.gauge.wear.ui.sendDisarm] helpers `GlanceScreen`'s own Arm/Disarm chip uses — never a
 *    retyped `ACTION_ARM` / `ACTION_DISARM` string.
 *
 *  - REFRESH RACE + mitigation: the layout built right after forwarding a click reads
 *    [ControllerStateBus.state] *immediately*, before `SentinelService` has necessarily processed
 *    the intent — `sendArm`/`sendDisarm` return as soon as the intent is handed to the system, not
 *    after `SentinelService`'s own handler thread has run `armOnLoopThread`/`disarm()` and
 *    republished the bus. So the tile can render the *pre*-tap state for one refresh immediately
 *    after a tap, before catching up. [onTileRequest] schedules a one-shot
 *    `TileService.getUpdater(applicationContext).requestUpdate(GaugeTileService::class.java)`
 *    [DELAYED_REFRESH_MILLIS] (1.5s) after forwarding a click — long enough for
 *    `SentinelService`'s handler thread to have processed the intent and republished the bus in
 *    the overwhelming majority of cases, short enough that the window is barely perceptible —
 *    which shrinks the worst-case staleness from up to [FRESHNESS_INTERVAL_MILLIS] (60s) down to
 *    ~1.5s on every tap of this tile's own Arm/Disarm button. This is a bounded, stock-SDK
 *    mitigation entirely within this file; it does not eliminate the race on its own (a
 *    sufficiently slow handler-thread tick could still lose it). P4-4 landed the complete fix
 *    alongside it: `SentinelService.publishAndNotify` now also calls
 *    `TileService.getUpdater(...).requestUpdate(GaugeTileService::class.java)` the moment the bus
 *    actually changes (`runCatching`-wrapped there, same telemetry-never-breaks-sentinel posture
 *    as every other best-effort call in that class) — this method's own 1.5s delayed refresh
 *    remains as a same-file backstop for the narrow window between a click being forwarded and
 *    that publish landing, not the only refresh path anymore. PUSH ON CHANGE ONLY (review fix,
 *    round 2): "the moment the bus actually changes" means changes to [tileStrings]'s own pure
 *    output specifically, not every republish — `tick()` runs on a ~1s cadence for the entire
 *    armed session (not just episodes), so `SentinelService` compares each tick's `tileStrings`
 *    output against the last value it actually pushed and only calls `requestUpdate` when that
 *    changed (armed flag flips, or mode changes) — see `SentinelService.pushFaceUpdates`'s own
 *    KDoc for the full reasoning.
 *
 *  - [onResourcesRequest] is left at [TileService]'s own default (empty resources) — the layout
 *    uses no images, only text.
 */
class GaugeTileService : TileService() {

    override fun onTileRequest(
        requestParams: RequestBuilders.TileRequest,
    ): ListenableFuture<TileBuilders.Tile> {
        when (requestParams.state?.lastClickableId) {
            CLICK_ID_ARM -> {
                sendArm(applicationContext)
                scheduleDelayedRefresh()
            }
            CLICK_ID_DISARM -> {
                sendDisarm(applicationContext)
                scheduleDelayedRefresh()
            }
        }

        val snapshot = ControllerStateBus.state.value
        val layout = tileLayout(armed = snapshot.sentinel != SentinelState.DISARMED, mode = snapshot.mode)

        val tile = TileBuilders.Tile.Builder()
            .setResourcesVersion(RESOURCES_VERSION)
            .setFreshnessIntervalMillis(FRESHNESS_INTERVAL_MILLIS)
            .setTimeline(
                TimelineBuilders.Timeline.Builder()
                    .addTimelineEntry(
                        TimelineBuilders.TimelineEntry.Builder()
                            .setLayout(LayoutElementBuilders.Layout.Builder().setRoot(layout).build())
                            .build(),
                    )
                    .build(),
            )
            .build()

        return immediateFuture(tile)
    }

    /**
     * Bounded refresh-race mitigation (see this class's own KDoc): posts a one-shot
     * `requestUpdate` [DELAYED_REFRESH_MILLIS] after a click, giving `SentinelService`'s handler
     * thread time to have actually processed the arm/disarm intent and republished
     * [ControllerStateBus] before this tile re-renders — without which the very next render can
     * still show the pre-tap state.
     */
    private fun scheduleDelayedRefresh() {
        Handler(Looper.getMainLooper()).postDelayed(
            { TileService.getUpdater(applicationContext).requestUpdate(GaugeTileService::class.java) },
            DELAYED_REFRESH_MILLIS,
        )
    }

    private companion object {
        const val RESOURCES_VERSION = "1"
        const val FRESHNESS_INTERVAL_MILLIS = 60_000L
        const val DELAYED_REFRESH_MILLIS = 1_500L
    }
}

/**
 * Minimal synchronous [ListenableFuture]: [GaugeTileService.onTileRequest] never has anything to
 * actually await (the layout is built from data already in hand), so this avoids pulling in a
 * `Futures`-style helper dependency purely to wrap an already-known value — `androidx.wear.tiles`
 * only compiles against the bare `com.google.guava:listenablefuture` interface jar (no `Futures`
 * utility class), and `androidx.concurrent:concurrent-futures` (which has one, via
 * `CallbackToFutureAdapter`) is only a *runtime* transitive dependency here, not safe to reference
 * at compile time without adding it explicitly.
 */
private fun <T> immediateFuture(value: T): ListenableFuture<T> = object : ListenableFuture<T> {
    override fun addListener(listener: Runnable, executor: Executor) {
        executor.execute(listener)
    }

    override fun cancel(mayInterruptIfRunning: Boolean) = false
    override fun isCancelled() = false
    override fun isDone() = true
    override fun get(): T = value
    override fun get(timeout: Long, unit: TimeUnit): T = value
}
