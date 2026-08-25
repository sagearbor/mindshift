/**
 * Call invite links, both shapes, both roles:
 *
 *   mindshift://call/<code>[?role=therapist]          (the native app)
 *   https://arborfam-hub.web.app/call/<code>[?role=…] (Safari — Mom)
 *
 * `parseCallLink` accepts either (and a bare path "/call/<code>" the web
 * build sees in window.location) and pulls out the code AND the role: a
 * therapist link carries ?role=therapist, everything else is a participant.
 * The code is whatever the server issued, kept to a safe alphabet so it can
 * ride a URL path and a REST path.
 */
import type { CallRole } from "../live/call/types";

export const WEB_ORIGIN = "https://arborfam-hub.web.app";
export const APP_SCHEME = "mindshift";

const CODE_RE = /^[A-Za-z0-9_-]{3,64}$/;

export function isJoinCode(value: string): boolean {
  return CODE_RE.test(value);
}

export interface ParsedCallLink {
  code: string;
  role: CallRole;
}

/** The code (+ role) inside an invite link, or null when the url isn't one. */
export function parseCallLink(url: string | null | undefined): ParsedCallLink | null {
  if (!url || typeof url !== "string") return null;
  let rest: string | null = null;
  const trimmed = url.trim();
  const schemeMatch = trimmed.match(/^([a-z][a-z0-9+.-]*):\/\/(.*)$/i);
  if (schemeMatch) {
    const scheme = schemeMatch[1].toLowerCase();
    let tail = schemeMatch[2];
    if (scheme === APP_SCHEME) {
      // mindshift://call/<code>  (host is "call")
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
  const [path, query = ""] = rest.split("#")[0].split("?");
  const m = path.match(/^\/call\/([^/]+)\/?$/);
  if (!m) return null;
  let code: string;
  try {
    code = decodeURIComponent(m[1]);
  } catch {
    return null;
  }
  if (!isJoinCode(code)) return null;
  const roleParam = new URLSearchParams(query).get("role");
  const role: CallRole = roleParam === "therapist" ? "therapist" : "participant";
  return { code, role };
}

export function callWebUrl(code: string, role: CallRole = "participant"): string {
  const base = `${WEB_ORIGIN}/call/${encodeURIComponent(code)}`;
  return role === "therapist" ? `${base}?role=therapist` : base;
}

export function callDeepLink(code: string, role: CallRole = "participant"): string {
  const base = `${APP_SCHEME}://call/${encodeURIComponent(code)}`;
  return role === "therapist" ? `${base}?role=therapist` : base;
}

/** The invite text for the share sheet / clipboard, for the role the host
 *  is inviting. `joinUrl` is the server's own link when it gave one AND it
 *  isn't role-specific; otherwise the hosted web URL carrying the role. */
export function inviteMessage(
  code: string,
  joinUrl?: string | null,
  role: CallRole = "participant",
): { message: string; url: string } {
  const url =
    role === "participant" && joinUrl && /^https?:\/\//.test(joinUrl) ? joinUrl : callWebUrl(code, role);
  const who = role === "therapist" ? "watch my MindShift call (as the therapist)" : "join my MindShift call";
  const message =
    `Please ${who}: ${url}\n` +
    `(Code: ${code} — in the app: Live Coach → Call → Join with code, or open ${callDeepLink(code, role)})`;
  return { message, url };
}
