/**
 * Wearer-identity decision rule (plan 2026-11-05, Step 3):
 *   - judge only turns >= 1.5 s (shorter = abstain, never a claim);
 *   - a cluster is the wearer by SECONDS-weighted evidence, not by one turn;
 *   - overlapped turns (mixed windows) never count toward that evidence.
 * Synthetic axis vectors (the print is axis 0); the real-audio numbers come
 * from scripts/recreplay/identity_eval.py.
 */
import { ECAPA_DIM, SpeakerLabeler, type EnrolledPerson } from "../src/live/speakerId";
import { vectorAtCosine } from "../src/live/testing/synth";

const D = ECAPA_DIM;
const print = vectorAtCosine(D, 1, 0, 1);
const person: EnrolledPerson = {
  personId: "me",
  displayName: "You",
  isSelf: true,
  embedding: print,
  settings: 1,
};
/** A turn embedding at cosine `c` to the print, in the "room" direction `dir`. */
const at = (c: number, dir = 7) => vectorAtCosine(D, c, 0, dir);

describe("minJudgeSeconds", () => {
  it("a short turn that clears the print bar is still the wearer (a shout is short)", () => {
    const l = new SpeakerLabeler([person], { minJudgeSeconds: 1.5 });
    expect(l.label(at(0.9), 1.0)).toMatchObject({ isSelf: true, basis: "absolute" });
  });

  it("a short turn that does NOT clear the bar abstains: no 'not you' claim, no cluster, no vote seconds", () => {
    const l = new SpeakerLabeler([person], { minJudgeSeconds: 1.5 });
    const v = l.label(at(0.3), 1.0);
    expect(v.isSelf).toBeNull();
    expect(v.speaker).toBe("Unknown");
    expect(l.clusterCount).toBe(0);
  });

  it("a short turn never joins an existing cluster", () => {
    const l = new SpeakerLabeler([person], { minJudgeSeconds: 1.5 });
    expect(l.label(at(0.3), 3).speaker).toBe("Speaker A");
    expect(l.label(at(0.3), 1.0).speaker).toBe("Unknown");
  });

  it("judges a turn at or above the bar normally", () => {
    const l = new SpeakerLabeler([person], { minJudgeSeconds: 1.5 });
    expect(l.label(at(0.9), 1.5).isSelf).toBe(true);
    expect(l.label(vectorAtCosine(D, 0, 5, 6), 2.0).isSelf).toBe(false);
  });

  it("is OFF by default (at 1.5 s it cost replay.nudgeReport a pinned scene_meeting4 hit)", () => {
    const l = new SpeakerLabeler([person]);
    expect(l.label(at(0.3), 1.0).speaker).toBe("Unknown"); // < MIN_CLUSTER_SECONDS: unchanged
    expect(l.label(at(0.3), 3).speaker).toBe("Speaker A");
    // The old behaviour survives: a 1 s turn still joins the existing cluster.
    expect(l.label(at(0.3), 1.0).speaker).toBe("Speaker A");
  });

  it("a caller that omits seconds is not guarded", () => {
    const l = new SpeakerLabeler([person]);
    expect(l.label(at(0.9)).isSelf).toBe(true);
  });
});

describe("clusterSelfVote (seconds-weighted)", () => {
  // The wearer's voice drifts: every turn lands at ~0.50 vs the print, under
  // the 0.60 bar, but consistently. One turn is weak evidence; 8+ s of them
  // is not.
  const opts = { clusterSelfVote: { minSeconds: 8, meanScore: 0.45 } };

  it("stays not-self until enough seconds have accumulated, then flips the cluster to self", () => {
    const l = new SpeakerLabeler([person], opts);
    expect(l.label(at(0.5), 3).isSelf).toBe(false);
    expect(l.label(at(0.5), 3).isSelf).toBe(false); // 6 s < 8 s
    const third = l.label(at(0.5), 3); // 9 s
    expect(third.isSelf).toBe(true);
    expect(third.basis).toBe("session");
  });

  it("is weighted by seconds: one long turn outweighs several short ones", () => {
    const l = new SpeakerLabeler([person], opts);
    expect(l.label(at(0.5), 9).isSelf).toBe(false); // first turn founds the cluster
    expect(l.label(at(0.5), 2).isSelf).toBe(true); // 11 s of evidence >= 8
  });

  it("an abstained (short) turn adds no seconds to the vote", () => {
    const l = new SpeakerLabeler([person], opts);
    for (let k = 0; k < 10; k++) l.label(at(0.5), 1.0);
    expect(l.label(at(0.5), 3).isSelf).toBe(false);
  });

  it("a stranger's cluster (low mean score) never flips", () => {
    const l = new SpeakerLabeler([person], opts);
    for (let k = 0; k < 6; k++) {
      expect(l.label(vectorAtCosine(D, 0.1, 0, 9), 4).isSelf).toBe(false);
    }
  });

  it("is off unless configured", () => {
    const l = new SpeakerLabeler([person]);
    for (let k = 0; k < 6; k++) expect(l.label(at(0.5), 4).isSelf).toBe(false);
  });
});

describe("windowVote.vetoOtherFrac (overlap exclusion)", () => {
  const wv = (vetoOtherFrac?: number) => ({
    windowVote: { seconds: 2, selfThreshold: 0.6, otherCeiling: 0.35, minFrac: 0.6, maxWindows: 6, vetoOtherFrac },
  });
  const w = (c: number) => vectorAtCosine(D, c, 0, 7);

  it("default veto needs half the windows to be someone else", () => {
    const l = new SpeakerLabeler([person], wv());
    // full turn matches; 1 of 4 windows is clearly someone else -> still self
    expect(l.label(at(0.8), 8, [w(0.8), w(0.8), w(0.8), w(0.1)]).isSelf).toBe(true);
  });

  it("a lower vetoOtherFrac abstains when a quarter of the windows are another voice", () => {
    const l = new SpeakerLabeler([person], wv(0.25));
    expect(l.label(at(0.8), 8, [w(0.8), w(0.8), w(0.8), w(0.1)]).isSelf).toBeNull();
  });

  it("a clean turn is unaffected", () => {
    const l = new SpeakerLabeler([person], wv(0.25));
    expect(l.label(at(0.8), 8, [w(0.8), w(0.7), w(0.75), w(0.8)]).isSelf).toBe(true);
  });
});
