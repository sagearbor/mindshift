/**
 * "Hey Google, start my journal" deep links (docs/plans/2026-08-30-voice-shortcut.md):
 *
 *   mindshift://journal/start   → open Live Coach, select Journal mode, start it
 *   mindshift://journal/stop    → stop a running journal session
 *
 * Mirrors src/nav/callLink.ts: also accepts the https origin's path and a bare
 * "/journal/start|stop" (the shape the web build sees in window.location).
 * The screen executes the action honoring its existing gates (enrolled owner
 * voiceprint, mic permission) — this module only parses.
 */
import { APP_SCHEME } from "./callLink";

export type JournalLinkAction = "start" | "stop";

/** The journal action inside a deep link, or null when the url isn't one. */
export function parseJournalLink(url: string | null | undefined): JournalLinkAction | null {
  if (!url || typeof url !== "string") return null;
  const trimmed = url.trim();
  let rest: string | null = null;
  const schemeMatch = trimmed.match(/^([a-z][a-z0-9+.-]*):\/\/(.*)$/i);
  if (schemeMatch) {
    const scheme = schemeMatch[1].toLowerCase();
    let tail = schemeMatch[2];
    if (scheme === APP_SCHEME) {
      // mindshift://journal/start  (host is "journal")
      rest = `/${tail}`;
    } else if (scheme === "http" || scheme === "https") {
      const slash = tail.indexOf("/");
      tail = slash === -1 ? "" : tail.slice(slash);
      rest = tail;
    } else {
      return null;
    }
  } else if (trimmed.startsWith("/")) {
    rest = trimmed;
  } else {
    return null;
  }
  const path = rest.split("#")[0].split("?")[0];
  const m = path.match(/^\/journal\/(start|stop)\/?$/i);
  return m ? (m[1].toLowerCase() as JournalLinkAction) : null;
}
