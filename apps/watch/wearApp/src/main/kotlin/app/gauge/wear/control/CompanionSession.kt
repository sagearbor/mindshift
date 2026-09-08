package app.gauge.wear.control

import java.time.Instant
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter

/**
 * Tier B (Companion mode): the deterministic per-day, per-account live-session id the watch opens
 * its no-mic nudge socket under — `companion-YYYYMMDD-<account>`.
 *
 * The server relay (server/watch/relay.py) keys its registry by the authenticated ACCOUNT of the
 * open socket, so any id would work; deterministic-by-day just makes server logs coherent across
 * the reconnects a whole day of pocket use produces (each reconnect reuses the same id instead of
 * minting a fresh UUID per attempt — deliberately unlike a triggered episode's reconnect, which
 * mints a new id because each episode doc is a distinct capture; a companion session persists
 * nothing, see the server's `companion` hello handling).
 *
 * The account id is sanitized to URL-path-safe characters (it rides in the WS path) and truncated
 * — it only disambiguates logs, it is NOT the auth (the `?token=`/`?account=` query param is).
 */
fun companionSessionId(accountId: String, nowMs: Long = System.currentTimeMillis()): String {
    val day = DateTimeFormatter.ofPattern("yyyyMMdd")
        .withZone(ZoneOffset.UTC)
        .format(Instant.ofEpochMilli(nowMs))
    val safeAccount = accountId.filter { it.isLetterOrDigit() || it == '-' || it == '_' }
        .take(24)
        .ifEmpty { "anon" }
    return "companion-$day-$safeAccount"
}
