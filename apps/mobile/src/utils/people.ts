/**
 * People labeling — pure helpers shared by the People screen, the
 * "Who is this?" sheet, and the transcript/session views.
 *
 * Everything here is deterministic and network-free so the screen logic can
 * be unit-tested with a mocked client; the honest-refusal copy lives here so
 * every surface renders the server's three 422 reasons the same way.
 */
import type { SpeakerLabel, VoicePerson } from "../api/client";

export const SELF_PERSON_ID = "self";
export const SELF_DISPLAY_NAME = "You";

/** The server's slug rule for a person id (speaker_id.PERSON_ID_PATTERN):
 *  lowercase, starts alphanumeric, then [a-z0-9_-], max 40 chars. */
const PERSON_ID_RE = /^[a-z0-9][a-z0-9_-]{0,39}$/;

/**
 * Derive a storage slug from a typed display name ("Mom" → "mom", "Aunt
 * Béa" → "aunt-bea"). Falls back to "person" when nothing survives the
 * cleanup (all punctuation / non-Latin); `taken` ids get a numeric suffix
 * so "Mom" and a second "Mom" never collide (the server keys people by
 * slug). Never returns the reserved "self".
 */
export function slugifyPersonId(
  name: string,
  taken: Iterable<string> = [],
): string {
  const base =
    name
      .normalize("NFKD")
      .replace(/[̀-ͯ]/g, "") // strip combining accents
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 32) || "person";
  const stem = base === SELF_PERSON_ID ? "person" : base;
  const used = new Set(taken);
  if (!used.has(stem) && PERSON_ID_RE.test(stem)) return stem;
  for (let i = 2; i < 1000; i += 1) {
    const candidate = `${stem}-${i}`;
    if (!used.has(candidate) && PERSON_ID_RE.test(candidate)) return candidate;
  }
  return `${stem}-${Date.now().toString(36)}`;
}

/** The name a person renders under — "You" for the owner, the given name
 *  otherwise, the id as a last resort (never an empty string). */
export function personDisplayName(
  p: { person_id?: string | null; display_name?: string | null; is_self?: boolean } | null | undefined,
): string {
  if (!p) return "";
  if (p.is_self || p.person_id === SELF_PERSON_ID) return SELF_DISPLAY_NAME;
  const name = (p.display_name ?? "").trim();
  return name || (p.person_id ?? "").trim() || "Someone";
}

/** Owner ("You") pinned first, then partners A→Z by display name. */
export function sortPeople<T extends VoicePerson>(people: readonly T[]): T[] {
  return [...people].sort((a, b) => {
    if (a.is_self !== b.is_self) return a.is_self ? -1 : 1;
    return personDisplayName(a).localeCompare(personDisplayName(b));
  });
}

/** Total seconds of learned speech across a person's samples, when the
 *  server measured any (null when no sample carries `seconds`). */
export function personSeconds(p: VoicePerson): number | null {
  let total = 0;
  let any = false;
  for (const s of p.samples ?? []) {
    if (typeof s.seconds === "number" && Number.isFinite(s.seconds)) {
      total += s.seconds;
      any = true;
    }
  }
  return any ? Math.round(total) : null;
}

/** "3 samples · 42 s" — the People row's one-line summary. */
export function personSummary(p: VoicePerson): string {
  const n = p.enroll_count ?? 0;
  const parts = [`${n} sample${n === 1 ? "" : "s"}`];
  const secs = personSeconds(p);
  if (secs !== null) parts.push(`${secs} s of speech`);
  return parts.join(" · ");
}

/**
 * Is this speaker's label a PERSON the app will recognize again? True for
 * the enrolled rung (a voiceprint match) and for a manual-person label whose
 * person id is in the enrolled list. A plain manual name is not — it labels
 * this one recording only.
 */
export function isEnrolledPersonLabel(
  entry: SpeakerLabel | null | undefined,
  people: readonly { person_id: string }[] = [],
): boolean {
  if (!entry) return false;
  if (entry.label_source === "enrolled") return true;
  if (entry.label_source === "manual-person") {
    const pid = entry.person_id ?? "";
    return pid === SELF_PERSON_ID || people.some((p) => p.person_id === pid);
  }
  return false;
}

/** The person id a label points at, when it does. */
export function labelPersonId(
  entry: SpeakerLabel | null | undefined,
): string | null {
  if (!entry) return null;
  if (entry.label_source === "manual-person") return entry.person_id ?? null;
  if (entry.label_source === "enrolled") {
    return entry.display_label === SELF_DISPLAY_NAME ? SELF_PERSON_ID : null;
  }
  return null;
}

export type EnrollRefusal =
  | "too-little-speech"
  | "sounds-like-someone-else"
  | "no-audio"
  | "unavailable"
  | "other";

/**
 * Classify a failed enroll-from-recording call. The server's 422 detail
 * starts with a stable bracketed tag; a 503 is "voice ID unavailable";
 * anything else is a generic failure. Returns the tag plus the server's
 * own sentence (tag stripped) so the UI can show the honest reason inline.
 */
export function enrollRefusalReason(
  err: unknown,
): { kind: EnrollRefusal; message: string } {
  const e = (err ?? {}) as { status?: number; detail?: string; message?: string };
  const raw = (e.detail || e.message || "").trim();
  const tagged = /^\[([a-z-]+)\]\s*(.*)$/s.exec(raw);
  if (e.status === 422 && tagged) {
    const tag = tagged[1];
    const message = tagged[2] || raw;
    if (tag === "too-little-speech" || tag === "sounds-like-someone-else" || tag === "no-audio") {
      return { kind: tag, message };
    }
    return { kind: "other", message };
  }
  if (e.status === 503) {
    return {
      kind: "unavailable",
      message: raw || "Voice recognition isn’t available on this server.",
    };
  }
  if (e.status === 422 && raw) return { kind: "other", message: raw };
  return {
    kind: "other",
    message: "Couldn’t save that voice. Please try again.",
  };
}

/** Short human title per refusal kind (the inline error's headline). */
export function enrollRefusalTitle(kind: EnrollRefusal): string {
  switch (kind) {
    case "too-little-speech":
      return "Not enough of their voice here";
    case "sounds-like-someone-else":
      return "That sounds like someone else";
    case "no-audio":
      return "No audio to learn from";
    case "unavailable":
      return "Voice recognition unavailable";
    default:
      return "Couldn’t remember this voice";
  }
}
