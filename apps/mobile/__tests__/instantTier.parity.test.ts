/**
 * PARITY — does the hand-written TypeScript extractor agree with openSMILE?
 *
 * Not numerically: the frame windows, the auditory model behind eGeMAPS
 * "loudness", the mel filterbank and the pitch tracker all differ, and
 * chasing bit-parity with a native library we do not ship would be work
 * spent on the wrong thing. What has to hold is RANK: given the same 2 s of
 * audio, the phone must order clips by anger the way the Python reference
 * does, or the corpus numbers behind the model mean nothing on-device.
 *
 * The fixture (fixtures/instantTier.parity.json, regenerate with
 * `python scripts/instant_tier_select.py fixture`) carries 147 clips spread
 * over both corpora, three emotions and as many speakers as each corpus has:
 * the reference model's P(angry) for the clip's 2 s window, and the features
 * this extractor measured on exactly those samples. A handful also carry the
 * raw PCM, so the extractor itself is re-run here rather than trusted.
 */
import fixture from "./fixtures/instantTier.parity.json";

import {
  INSTANT_FEATURE_NAMES,
  extractInstantFeatures,
  instantHeatScore,
  scoreInstantFeatures,
  type InstantFeatures,
} from "../src/live/instantTier";

interface Clip {
  key: string;
  corpus: string;
  speaker: string;
  emotion: string;
  isAngry: boolean;
  pythonScore: number;
  tsFeatures: Record<string, number | null>;
  pcmBase64?: string;
  sampleRate?: number;
}

const clips = fixture.clips as Clip[];

/** Spearman rank correlation, mid-ranks on ties. */
function spearman(a: number[], b: number[]): number {
  const rank = (xs: number[]): number[] => {
    const order = xs.map((v, i) => [v, i] as const).sort((p, q) => p[0] - q[0]);
    const out = new Array<number>(xs.length);
    let i = 0;
    while (i < order.length) {
      let j = i;
      while (j + 1 < order.length && order[j + 1][0] === order[i][0]) j++;
      const mid = (i + j) / 2 + 1;
      for (let k = i; k <= j; k++) out[order[k][1]] = mid;
      i = j + 1;
    }
    return out;
  };
  const ra = rank(a);
  const rb = rank(b);
  const n = a.length;
  const mean = (n + 1) / 2;
  let num = 0;
  let da = 0;
  let db = 0;
  for (let i = 0; i < n; i++) {
    num += (ra[i] - mean) * (rb[i] - mean);
    da += (ra[i] - mean) ** 2;
    db += (rb[i] - mean) ** 2;
  }
  return num / Math.sqrt(da * db);
}

/** AUC via the rank-sum identity. */
function auc(labels: number[], scores: number[]): number {
  const order = scores.map((s, i) => [s, i] as const).sort((p, q) => p[0] - q[0]);
  const rank = new Array<number>(scores.length);
  let i = 0;
  while (i < order.length) {
    let j = i;
    while (j + 1 < order.length && order[j + 1][0] === order[i][0]) j++;
    const mid = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) rank[order[k][1]] = mid;
    i = j + 1;
  }
  let sumPos = 0;
  let nPos = 0;
  for (let k = 0; k < labels.length; k++) {
    if (labels[k] === 1) {
      sumPos += rank[k];
      nPos++;
    }
  }
  const nNeg = labels.length - nPos;
  return (sumPos - (nPos * (nPos + 1)) / 2) / (nPos * nNeg);
}

const tsScore = (c: Clip): number => scoreInstantFeatures(c.tsFeatures as InstantFeatures);

describe("instant tier — parity with the openSMILE reference", () => {
  it("has a fixture worth testing against", () => {
    expect(clips.length).toBeGreaterThanOrEqual(30);
    expect(new Set(clips.map((c) => c.corpus))).toEqual(new Set(["cremad", "ravdess"]));
    expect(new Set(clips.map((c) => c.emotion))).toEqual(new Set(["angry", "happy", "neutral"]));
    expect(new Set(clips.map((c) => c.speaker)).size).toBeGreaterThanOrEqual(10);
    for (const c of clips) {
      expect(Object.keys(c.tsFeatures).sort()).toEqual([...INSTANT_FEATURE_NAMES].sort());
    }
  });

  it("ranks the clips the way the Python reference does", () => {
    const rho = spearman(clips.map(tsScore), clips.map((c) => c.pythonScore));
    console.log(`instant tier parity: Spearman ${rho.toFixed(3)} over ${clips.length} clips`);
    expect(rho).toBeGreaterThanOrEqual(0.9);
  });

  it("separates angry from happy on the fixture", () => {
    const ah = clips.filter((c) => c.emotion === "angry" || c.emotion === "happy");
    const a = auc(
      ah.map((c) => (c.isAngry ? 1 : 0)),
      ah.map(tsScore),
    );
    console.log(`instant tier fixture angry-vs-happy AUC ${a.toFixed(3)} over ${ah.length} clips`);
    expect(a).toBeGreaterThanOrEqual(0.75);
  });

  it("scores a neutral clip below the same speaker's angry one", () => {
    // the weaker, more human version of the AUC above: per speaker, not pooled
    const bySpeaker = new Map<string, Clip[]>();
    for (const c of clips) {
      const list = bySpeaker.get(c.speaker) ?? [];
      list.push(c);
      bySpeaker.set(c.speaker, list);
    }
    let pairs = 0;
    let right = 0;
    for (const list of bySpeaker.values()) {
      const angry = list.filter((c) => c.emotion === "angry");
      const neutral = list.filter((c) => c.emotion === "neutral");
      for (const a of angry) {
        for (const n of neutral) {
          pairs++;
          if (tsScore(a) > tsScore(n)) right++;
        }
      }
    }
    if (pairs > 0) expect(right / pairs).toBeGreaterThanOrEqual(0.75);
  });
});

describe("instant tier — end to end on real PCM", () => {
  const withAudio = clips.filter((c): c is Clip & { pcmBase64: string } => Boolean(c.pcmBase64));

  it("carries a few clips' audio so the extractor is run, not trusted", () => {
    expect(withAudio.length).toBeGreaterThanOrEqual(3);
  });

  it.each(withAudio.map((c) => [c.key, c] as const))(
    "re-derives %s's stored features from its samples",
    (_key, clip) => {
      const bytes = Buffer.from(clip.pcmBase64, "base64");
      const pcm = new Int16Array(bytes.byteLength / 2);
      for (let i = 0; i < pcm.length; i++) pcm[i] = bytes.readInt16LE(i * 2);
      // CREMA-D clips are often shorter than two seconds; the dump keeps the
      // whole clip in that case, and the extractor handles it.
      const sr = clip.sampleRate ?? 16000;
      expect(pcm.length).toBeLessThanOrEqual(sr * 2);
      expect(pcm.length).toBeGreaterThanOrEqual(sr);

      const measured = extractInstantFeatures(pcm, sr);
      for (const name of INSTANT_FEATURE_NAMES) {
        const expected = clip.tsFeatures[name];
        const got = measured[name];
        if (expected === null) {
          expect(got).toBeNull();
          continue;
        }
        expect(got).not.toBeNull();
        // the fixture stores six decimals; anything looser than this is a
        // change in the extractor, and the fixture must be regenerated with
        // `python scripts/instant_tier_select.py fixture`
        expect(got as number).toBeCloseTo(expected, 4);
      }
      // the fixture's features are rounded to six decimals, so the score
      // re-derived from the samples lands within a rounding error of the one
      // computed from the stored vector
      expect(instantHeatScore(pcm, sr).score as number).toBeCloseTo(tsScore(clip), 5);
    },
  );
});
