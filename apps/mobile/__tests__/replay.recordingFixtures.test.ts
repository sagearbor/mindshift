/**
 * Opt-in: the owner's PRIVATE real-conversation fixtures
 * (tmp/recordings/fixtures/<name>/, frozen by scripts/recording_replay.py)
 * re-run through the phone's real fast loop. The loop is deterministic on
 * a virtual clock, so the turn_local stream must match the frozen
 * phone.json exactly in speaker, is_self, span and send time — any change
 * in VAD, segmentation or speaker-ID on real family audio shows up here
 * first (the Python test then shows what it did to the coach).
 *
 * Skipped when the fixtures or the ECAPA export are absent (CI has neither).
 */
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { runRecordingReplay, type RecordingEnroll, type RecordingReplayOutput } from "../src/live/replay/recordingReplay";
import { findEcapaModel, REPO_ROOT } from "../src/live/replay/sceneReplay";

const FIXTURES = path.join(process.env.MINDSHIFT_RECORDINGS_DIR ?? path.join(REPO_ROOT, "tmp/recordings"), "fixtures");
const names = fs.existsSync(FIXTURES)
  ? fs.readdirSync(FIXTURES).filter((n) => ["audio.wav", "phone_meta.json", "phone.json", "baseline.json"].every((f) => fs.existsSync(path.join(FIXTURES, n, f))))
  : [];
const ecapa = findEcapaModel();
const maybe = names.length && ecapa ? describe : describe.skip;

function profileFor(baseline: { settings?: { profile?: string | null } }): string | null {
  const p = baseline.settings?.profile;
  if (!p) return null;
  const abs = path.isAbsolute(p) ? p : path.join(REPO_ROOT, p);
  if (fs.existsSync(abs)) return abs;
  const alt = path.join(REPO_ROOT, "tmp/private_fixtures", path.basename(p));
  return fs.existsSync(alt) ? alt : null;
}

maybe("recording fixtures: phone loop unchanged on real audio", () => {
  it.each(names)("%s", async (name) => {
    const dir = path.join(FIXTURES, name);
    const frozen = JSON.parse(fs.readFileSync(path.join(dir, "phone.json"), "utf8")) as RecordingReplayOutput;
    const baseline = JSON.parse(fs.readFileSync(path.join(dir, "baseline.json"), "utf8"));
    const enroll = (frozen.enrollment?.mode ?? "same") as RecordingEnroll;
    const profile = enroll === "profile" ? profileFor(baseline) : null;
    if (enroll === "profile" && !profile) {
      console.log(`${name}: owner profile not on this machine — skipped`);
      return;
    }
    const out = await runRecordingReplay({
      wav: path.join(dir, "audio.wav"),
      meta: path.join(dir, "phone_meta.json"),
      out: path.join(os.tmpdir(), `recfx-${name}.json`),
      mode: frozen.mode,
      enroll,
      profile,
      self: null,
    });
    const shape = (o: RecordingReplayOutput) =>
      o.sent.map((e) => ({ speaker: e.speaker, is_self: e.is_self, start: e.start_time, end: e.end_time, sent: e.sent_at_audio_s, text: e.text }));
    expect(shape(out)).toEqual(shape(frozen));
    expect(out.attribution).toEqual(frozen.attribution);
    expect(out.haptics).toEqual(frozen.haptics);
  }, 180_000);
});
