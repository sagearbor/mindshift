/**
 * src/live/speakerIdSetup.ts — the pure glue defaultDeps.ts uses to turn
 * "model resolved? voiceprints fetched?" into the labeler's people and the
 * one-line "speaker-ID on/off (…)" status the session screen shows.
 */
import {
  activeCapability,
  describeSpeakerId,
  inactiveCapability,
  peopleForModel,
} from "../src/live/speakerIdSetup";
import type { EnrolledPerson } from "../src/live/speakerId";
import { ECAPA_REVISION } from "../src/api/liveSessions";

const you: EnrolledPerson = { personId: "self", displayName: "You", isSelf: true, embedding: [1, 0], model: "speechbrain/x@rev1" };
const mom: EnrolledPerson = { personId: "mom", displayName: "Mom", isSelf: false, embedding: [0, 1], model: "speechbrain/x@rev0" };
const legacy: EnrolledPerson = { personId: "dad", displayName: "Dad", isSelf: false, embedding: [1, 1], model: null };

describe("peopleForModel", () => {
  it("keeps prints from the pinned revision and legacy prints; drops other revisions", () => {
    const { kept, dropped } = peopleForModel([you, mom, legacy], "rev1");
    expect(kept.map((p) => p.personId)).toEqual(["self", "dad"]);
    expect(dropped.map((p) => p.personId)).toEqual(["mom"]);
  });

  it("drops nothing without a revision to compare against", () => {
    expect(peopleForModel([you, mom], null).dropped).toEqual([]);
    expect(peopleForModel([you, mom], "").kept).toHaveLength(2);
  });

  it("regression 2026-08-31: the comparison is the PINNED revision, never a download ETag", () => {
    // Firebase Hosting tags the model file with a content hash, not the
    // model revision; comparing prints against that ETag dropped all three
    // enrolled voiceprints on a real device (droppedForModel: 3) and the
    // journal demanded enrollment from an enrolled user.
    const print: EnrolledPerson = { ...you, model: `speechbrain/spkrec-ecapa-voxceleb@${ECAPA_REVISION}` };
    const { kept, dropped } = peopleForModel([print], ECAPA_REVISION);
    expect(kept).toHaveLength(1);
    expect(dropped).toEqual([]);
  });
});

describe("capability + status line", () => {
  it("describes an inactive rung with its reason", () => {
    const cap = inactiveCapability("server has no ECAPA model (503)");
    expect(cap).toEqual({ active: false, reason: "server has no ECAPA model (503)", enrolled: 0, model: null, droppedForModel: 0 });
    expect(describeSpeakerId(cap)).toBe("speaker-ID off (server has no ECAPA model (503))");
  });

  it("describes an active rung: enrolled count, model source, skipped prints, voiceprint errors", () => {
    const ready = { status: "ready" as const, path: "/m/ecapa.onnx", source: "cached" as const, etag: '"rev1"' };
    const cap = activeCapability(ready, [you], 1, null);
    expect(cap).toEqual({ active: true, reason: "model cached", enrolled: 1, model: "cached", droppedForModel: 1 });
    expect(describeSpeakerId(cap)).toBe("speaker-ID on (1 enrolled, model cached, 1 skipped: other model)");

    const none = activeCapability({ ...ready, source: "downloaded" }, [], 0, "not signed in (401)");
    expect(none.reason).toBe("model downloaded; voiceprints unavailable: not signed in (401)");
    expect(describeSpeakerId(none)).toBe("speaker-ID on (0 enrolled, model downloaded)");
  });
});
