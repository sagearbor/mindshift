/**
 * replay/recordingReplay.ts — the PHONE side of the recording-replay
 * pipeline (scripts/recording_replay.py): a real recording through the real
 * fast loop (Silero + segmenter + ECAPA), enrolled with the OWNER'S OWN
 * voiceprint (a server profile JSON) instead of one pooled from the
 * recording, emitting the turn_local stream the phone would have sent and
 * when.
 *
 * The argument/profile/output pieces run everywhere; the replay itself is
 * skipped honestly without the ECAPA export, like replay.real.test.ts.
 */
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import {
  parseRecordingArgs,
  profileEnrollment,
  runRecordingReplay,
  type RecordingReplayOutput,
} from "../src/live/replay/recordingReplay";
import { enrollFromSource } from "../src/live/replay/enroll";
import { AUDIO_FIXTURES_DIR, findEcapaModel, loadModels, loadScene, DEFAULT_REPLAY_OPTIONS } from "../src/live/replay/sceneReplay";

describe("recording replay: arguments + owner profile", () => {
  it("parses the CLI the Python pipeline calls", () => {
    const a = parseRecordingArgs(["--wav", "a.wav", "--meta", "m.json", "--out", "o.json", "--mode", "speaker", "--enroll", "profile", "--profile", "p.json"]);
    expect(a).toMatchObject({ wav: "a.wav", meta: "m.json", out: "o.json", mode: "speaker", enroll: "profile", profile: "p.json" });
    expect(parseRecordingArgs(["--wav", "a.wav", "--meta", "m.json", "--out", "o.json"])).toMatchObject({ mode: "earpiece", enroll: "same", profile: null });
    expect(() => parseRecordingArgs(["--wav", "a.wav"])).toThrow(/--meta/);
    expect(() => parseRecordingArgs(["--wav", "a.wav", "--meta", "m", "--out", "o", "--enroll", "profile"])).toThrow(/--profile/);
  });

  it("--speaker-opts passes labeler tuning through as JSON (identity sweeps)", () => {
    const a = parseRecordingArgs(["--wav", "a.wav", "--meta", "m.json", "--out", "o.json", "--speaker-opts", '{"matchThreshold":0.55}']);
    expect(a.speakerOptions).toEqual({ matchThreshold: 0.55 });
    expect(parseRecordingArgs(["--wav", "a.wav", "--meta", "m.json", "--out", "o.json"]).speakerOptions).toBeNull();
    expect(() => parseRecordingArgs(["--wav", "a", "--meta", "m", "--out", "o", "--speaker-opts", "[1]"])).toThrow(/speaker-opts/);
  });

  it("turns a server voiceprint document into the owner's enrollment record", () => {
    const emb = Array.from({ length: 192 }, (_, i) => Math.sin(i));
    const rec = profileEnrollment(
      { person_id: "self", display_name: "You", is_self: true, embedding: emb, model: "speechbrain/spkrec-ecapa-voxceleb@rev", samples: [{}, {}, {}] },
      "You",
    );
    expect(rec).toMatchObject({ personId: "self", displayName: "You", isSelf: true, fromScene: "owner_profile", crossScene: true, model: "speechbrain/spkrec-ecapa-voxceleb@rev" });
    expect(Array.from(rec.embedding)).toHaveLength(192);
    expect(() => profileEnrollment({ embedding: [] }, "You")).toThrow(/embedding/);
  });
});

const ecapaPath = findEcapaModel();
const maybe = ecapaPath ? describe : describe.skip;

maybe("recording replay on a real recording (real Silero + ECAPA)", () => {
  it("family_real with an owner-profile print: same turns as same-scene enrollment, turn_locals timed", async () => {
    const models = await loadModels({ ...DEFAULT_REPLAY_OPTIONS, ortFactory: null, ecapaPath });
    // A "profile" built the way the server builds one: pooled from the owner's
    // own turns. The replay must match him with it exactly as it does when it
    // enrolls from the meta itself.
    const scene = loadScene("family_real", { selfSpeaker: "Sage" });
    const pooled = await enrollFromSource(models.embedder!, { script: scene.script, pcm: scene.pcmF32 }, "Sage", {
      personId: "self", displayName: "Sage", isSelf: true, crossScene: false, maxSeconds: 10,
    });
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "recreplay-"));
    const profile = path.join(dir, "profile.json");
    fs.writeFileSync(profile, JSON.stringify({ person_id: "self", display_name: "You", is_self: true, embedding: Array.from(pooled.embedding) }));
    const out: RecordingReplayOutput = await runRecordingReplay(
      {
        wav: path.join(AUDIO_FIXTURES_DIR, "test_recording_family_real.wav"),
        meta: path.join(AUDIO_FIXTURES_DIR, "test_recording_family_real_meta.json"),
        mode: "earpiece",
        enroll: "profile",
        profile,
        self: "Sage",
        out: path.join(dir, "out.json"),
      },
      models,
    );
    expect(out.enrollment.mode).toBe("profile");
    expect(out.enrollment.records[0]).toMatchObject({ displayName: "Sage", isSelf: true, fromScene: "owner_profile" });
    expect(out.attribution).toMatchObject({ selfCorrect: 3, selfTotal: 3 });
    expect(out.sent.length).toBe(out.turns.filter((t) => !t.beforeLoopUp).length);
    // every turn_local has the phone's send time, after its own audio ended
    for (const e of out.sent) {
      expect(e.sent_at_audio_s).toBeGreaterThanOrEqual(e.end_time);
      expect(e.type).toBe("turn_local");
    }
    expect(out.sent.filter((e) => e.is_self === true)).toHaveLength(3);
    // JSON-safe (Float32Arrays flattened): the Python side reads this file
    expect(JSON.parse(fs.readFileSync(path.join(dir, "out.json"), "utf8")).sent).toHaveLength(out.sent.length);
  }, 90_000);
});
