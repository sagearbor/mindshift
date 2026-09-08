/**
 * Parity of the phone's window-pass port with the SHIPPED numpy
 * (server/diarize_sliding_window.py), on the same window embeddings.
 * Fixtures: docs/research/2026-08-29-voice-separation/F-device-b/
 * parity_<fixture>.json (dump_parity.py) — inputs rounded to 1e-7 so both
 * sides see identical numbers; parity_rng.json pins the numpy RNG port.
 */
import * as fs from "fs";
import * as path from "path";
import {
  clusterWindows,
  eigengapK,
  refineAffinity,
  spectralLabels,
  symmetricEigen,
  windowLabelRuns,
} from "../src/live/diarizeWindows";
import { NumpyGenerator } from "../src/live/numpyRandom";

const F_DIR = path.resolve(__dirname, "../../../docs/research/2026-08-29-voice-separation/F-device-b");

interface Policy {
  max_k: number;
  min_k: number;
  k_eigengap: number;
  k: number;
  eigenvalues: number[];
  raw_labels: number[];
  labels: number[];
  segments: [number, number, number][];
}
interface Parity {
  fixture: string;
  window_s: number;
  hop_s: number;
  duration_s: number;
  starts: number[];
  embeddings: number[][];
  affinity_row_sums: number[];
  affinity_trace: number;
  policies: Record<string, Policy>;
}

function load(name: string): Parity {
  return JSON.parse(fs.readFileSync(path.join(F_DIR, `parity_${name}.json`), "utf8")) as Parity;
}

/** Fraction of windows whose labels agree under the best one-to-one mapping. */
function agreement(a: number[], b: number[]): number {
  const labs = (x: number[]) => [...new Set(x)];
  const la = labs(a);
  const lb = labs(b);
  let best = 0;
  const perm = (rest: number[], chosen: number[]) => {
    if (chosen.length === la.length || rest.length === 0) {
      const m = new Map<number, number>();
      chosen.forEach((v, i) => m.set(la[i], v));
      let hit = 0;
      for (let i = 0; i < a.length; i++) if (m.get(a[i]) === b[i]) hit++;
      best = Math.max(best, hit);
      return;
    }
    for (let i = 0; i < rest.length; i++) perm([...rest.slice(0, i), ...rest.slice(i + 1)], [...chosen, rest[i]]);
  };
  perm(lb, []);
  return best / a.length;
}

describe("numpy RNG port", () => {
  it("reproduces numpy.random.default_rng bit for bit (integers / random / choice)", () => {
    const dump = JSON.parse(fs.readFileSync(path.join(F_DIR, "parity_rng.json"), "utf8")) as Record<
      string,
      { seq: [string, number | null, number][]; choice_p: number[]; choice: number[] }
    >;
    for (const [seed, v] of Object.entries(dump)) {
      const g = new NumpyGenerator(Number(seed));
      for (const [kind, n, expected] of v.seq) {
        expect(kind === "r" ? g.random() : g.integers(n as number)).toBe(expected);
      }
      for (const expected of v.choice) expect(g.choice(v.choice_p)).toBe(expected);
    }
  });
});

describe.each(["family_real", "poker6"])("window-pass parity on %s", (name) => {
  const p = load(name);
  const embs = p.embeddings.map((e) => Float32Array.from(e));

  it("refined affinity matches numpy (row sums, trace)", () => {
    const aff = refineAffinity(embs);
    expect(aff.n).toBe(p.starts.length);
    // Row-max normalised: every row peaks at exactly 1.
    for (let i = 0; i < aff.n; i++) {
      let m = -Infinity;
      for (let j = 0; j < aff.n; j++) m = Math.max(m, aff.data[i * aff.n + j]);
      expect(m).toBeCloseTo(1, 12);
    }
    const trace = p.starts.map((_, i) => aff.data[i * aff.n + i]).reduce((s, x) => s + x, 0);
    expect(trace).toBeCloseTo(p.affinity_trace, 6);
    p.affinity_row_sums.forEach((rs, i) => {
      let s = 0;
      for (let j = 0; j < aff.n; j++) s += aff.data[i * aff.n + j];
      expect(s).toBeCloseTo(rs, 6);
    });
  });

  it("eigenvalues + eigengap k match numpy.linalg.eigvalsh", () => {
    const aff = refineAffinity(embs);
    for (const pol of Object.values(p.policies)) {
      const { k, eigenvalues } = eigengapK(aff, pol.max_k);
      expect(k).toBe(pol.k_eigengap);
      pol.eigenvalues.forEach((ev, i) => expect(eigenvalues[i]).toBeCloseTo(ev, 5));
    }
  });

  it.each(Object.keys(p.policies))("spectral labels, smoothing and runs match the server (%s)", (policy) => {
    const pol = p.policies[policy];
    const r = clusterWindows(embs, p.starts, p.hop_s, { maxSpeakers: pol.max_k, minSpeakers: pol.min_k });
    expect(r.kEigengap).toBe(pol.k_eigengap);
    expect(r.k).toBe(pol.k);
    expect(agreement(r.rawLabels, pol.raw_labels)).toBeGreaterThanOrEqual(0.99);
    expect(agreement(r.labels, pol.labels)).toBeGreaterThanOrEqual(0.99);
    // Same seeding ⇒ the very same label ids, not just the same partition.
    expect(r.rawLabels).toEqual(pol.raw_labels);
    expect(r.labels).toEqual(pol.labels);
    const segs = windowLabelRuns(r.labels, p.starts, p.window_s, 0, p.duration_s);
    expect(segs.length).toBe(pol.segments.length);
    segs.forEach(([s, e, l], i) => {
      expect(s).toBeCloseTo(pol.segments[i][0], 6);
      expect(e).toBeCloseTo(pol.segments[i][1], 6);
      expect(l).toBe(pol.segments[i][2]);
    });
  });

  it("spectralLabels at a fixed k is deterministic across calls", () => {
    const aff = refineAffinity(embs);
    expect(spectralLabels(aff, 3)).toEqual(spectralLabels(aff, 3));
    // The eigensolver itself: A·v = λ·v on the symmetrised affinity.
    const n = aff.n;
    const sym = new Float64Array(n * n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) sym[i * n + j] = (aff.data[i * n + j] + aff.data[j * n + i]) / 2;
    const eig = symmetricEigen({ n, data: sym });
    const top = n - 1;
    for (let i = 0; i < n; i++) {
      let av = 0;
      for (let j = 0; j < n; j++) av += sym[i * n + j] * eig.vectors[j][top];
      expect(av).toBeCloseTo(eig.values[top] * eig.vectors[i][top], 8);
    }
  });
});
