/**
 * Pure glue between the two server hand-offs that make on-device speaker-ID
 * real — the ECAPA model (`modelDownload.ts`) and the enrolled voiceprints
 * (`GET /voice/people?include_embeddings=true`, api/liveSessions.ts) — and
 * the one-line capability status the session screen shows ("On-device:
 * Silero VAD · speaker-ID on (2 enrolled, model cached) · LLM …").
 *
 * Kept free of native imports so it is unit-tested directly; defaultDeps.ts
 * just calls it.
 */
import type { EnrolledPerson } from "./speakerId";
import type { EcapaModelResult, EcapaModelSource } from "./modelDownload";

export interface SpeakerIdCapability {
  /** Embedder + labeler are wired for this session. */
  active: boolean;
  /** Why not (when inactive), or what it is running with (when active). */
  reason: string;
  /** Voiceprints actually loaded into the labeler. */
  enrolled: number;
  /** Where the model came from this launch; null when inactive. */
  model: EcapaModelSource | null;
  /** People the server returned but that were dropped as a different
   *  model's embedding (never matched across spaces). */
  droppedForModel: number;
}

/**
 * Keep only prints from the SAME embedding space as the model this app
 * pins. A person's `model` is "<source>@<revision>" (server/speaker_id.py);
 * the comparison revision is the app's PINNED model revision
 * (api/liveSessions.ECAPA_REVISION — the download URL is a pure function
 * of it, so it is what is on disk). NEVER the download's ETag: Firebase
 * Hosting tags the file with a CONTENT HASH, which matches no print's
 * revision and silently dropped every enrolled voiceprint (2026-08-31:
 * journal said "Enroll your voice first" while enrolled 8 samples; live
 * self-ID inert). A print with no recorded model (a legacy profile) is
 * kept — the server matches with it too, and there is no evidence of a
 * mismatch. Without a revision to compare nothing is dropped.
 */
export function peopleForModel(
  people: EnrolledPerson[],
  pinnedRevision: string | null | undefined,
): { kept: EnrolledPerson[]; dropped: EnrolledPerson[] } {
  const revision = pinnedRevision?.trim() || null;
  if (!revision) return { kept: people, dropped: [] };
  const kept: EnrolledPerson[] = [];
  const dropped: EnrolledPerson[] = [];
  for (const p of people) {
    const model = p.model ?? null;
    if (model && !model.endsWith(`@${revision}`)) dropped.push(p);
    else kept.push(p);
  }
  return { kept, dropped };
}

export function describeSpeakerId(cap: SpeakerIdCapability): string {
  if (!cap.active) return `speaker-ID off (${cap.reason})`;
  const parts = [`${cap.enrolled} enrolled`];
  if (cap.model) parts.push(`model ${cap.model}`);
  if (cap.droppedForModel > 0) parts.push(`${cap.droppedForModel} skipped: other model`);
  return `speaker-ID on (${parts.join(", ")})`;
}

export function inactiveCapability(reason: string): SpeakerIdCapability {
  return { active: false, reason, enrolled: 0, model: null, droppedForModel: 0 };
}

/** The capability for a loaded model + the people that survived
 *  `peopleForModel`. `voiceprintError` (a failed people fetch) is folded
 *  into the reason so the status line says why nobody is enrolled. */
export function activeCapability(
  model: Extract<EcapaModelResult, { status: "ready" }>,
  kept: EnrolledPerson[],
  droppedForModel: number,
  voiceprintError: string | null,
): SpeakerIdCapability {
  const reason = voiceprintError
    ? `model ${model.source}; voiceprints unavailable: ${voiceprintError}`
    : `model ${model.source}`;
  return {
    active: true,
    reason,
    enrolled: kept.length,
    model: model.source,
    droppedForModel,
  };
}
