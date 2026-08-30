/**
 * A bit-exact port of the slice of `numpy.random.default_rng(seed)` that the
 * server's seeded k-means++ (`diarize_sliding_window._kmeans`) draws from, so
 * the phone's spectral clustering (`diarizeWindows.ts`) seeds its restarts
 * with the SAME random choices as production and the two partitions are
 * reproducible against each other:
 *
 *   - `SeedSequence(seed)` (numpy/random/_seed_sequence.pyx): the entropy
 *     pool mixing that turns an integer seed into the PCG64 state words;
 *   - `PCG64` (numpy/random/src/pcg64): the 128-bit LCG with XSL-RR output;
 *   - `Generator.integers(n)` for n < 2^32: Lemire's bounded rejection
 *     sampler over the bit generator's BUFFERED 32-bit stream (the low half
 *     of one 64-bit draw first, the high half on the next call);
 *   - `Generator.random()`: `(next64 >> 11) * 2^-53`;
 *   - `Generator.choice(n, p=...)`: cumsum / searchsorted(side="right") over
 *     one `random()` draw.
 *
 * Verified against numpy 2.5 by `__tests__/diarizeWindows.parity.test.ts`
 * (sequence dumped by docs/research/2026-08-29-voice-separation/F-device-b/
 * dump_parity.py). BigInt is only used for the 128-bit state arithmetic —
 * a handful of draws per clustering, so cost is irrelevant.
 */

const MASK32 = 0xffffffff;
const MASK64 = (1n << 64n) - 1n;
const MASK128 = (1n << 128n) - 1n;
const PCG_MULT_128 = (2549297995355413924n << 64n) | 4865540595714422341n;

// SeedSequence constants (numpy/random/_seed_sequence.pyx).
const INIT_A = 0x43b0d7e5;
const MULT_A = 0x931e8875;
const INIT_B = 0x8b51f9dd;
const MULT_B = 0x58f38ded;
const MIX_MULT_L = 0xca01f9dd;
const MIX_MULT_R = 0x4973f715;
const XSHIFT = 16;
const POOL_SIZE = 4;

function u32(x: number): number {
  return x >>> 0;
}

/** numpy's `_coerce_to_uint32_array` for a non-negative integer seed. */
function seedWords(seed: number): number[] {
  if (!Number.isInteger(seed) || seed < 0) throw new Error(`numpyRandom: seed must be a non-negative integer, got ${seed}`);
  if (seed === 0) return [0];
  const words: number[] = [];
  let n = BigInt(seed);
  while (n > 0n) {
    words.push(Number(n & 0xffffffffn));
    n >>= 32n;
  }
  return words;
}

/** `SeedSequence(seed).generate_state(nWords, np.uint32)`. */
export function seedSequenceState(seed: number, nWords: number): number[] {
  const entropy = seedWords(seed);
  const pool = new Array<number>(POOL_SIZE).fill(0);
  const hashConst = { v: INIT_A };
  const hashmix = (value: number): number => {
    let v = u32(value ^ hashConst.v);
    hashConst.v = u32(Math.imul(hashConst.v, MULT_A));
    v = u32(Math.imul(v, hashConst.v));
    v = u32(v ^ (v >>> XSHIFT));
    return v;
  };
  const mix = (x: number, y: number): number => {
    let r = u32(Math.imul(MIX_MULT_L, x) - Math.imul(MIX_MULT_R, y));
    r = u32(r ^ (r >>> XSHIFT));
    return r;
  };
  for (let i = 0; i < POOL_SIZE; i++) pool[i] = hashmix(i < entropy.length ? entropy[i] : 0);
  for (let src = 0; src < POOL_SIZE; src++) {
    for (let dst = 0; dst < POOL_SIZE; dst++) {
      if (src !== dst) pool[dst] = mix(pool[dst], hashmix(pool[src]));
    }
  }
  for (let src = POOL_SIZE; src < entropy.length; src++) {
    for (let dst = 0; dst < POOL_SIZE; dst++) pool[dst] = mix(pool[dst], hashmix(entropy[src]));
  }
  // generate_state
  const out: number[] = [];
  let hc = INIT_B;
  for (let i = 0; i < nWords; i++) {
    let data = u32(pool[i % POOL_SIZE] ^ hc);
    hc = u32(Math.imul(hc, MULT_B));
    data = u32(Math.imul(data, hc));
    data = u32(data ^ (data >>> XSHIFT));
    out.push(data);
  }
  return out;
}

/** `numpy.random.PCG64(seed)` + the `Generator` methods the clustering uses. */
export class NumpyGenerator {
  private state: bigint;
  private readonly inc: bigint;
  private hasUint32 = false;
  private uinteger = 0;

  constructor(seed: number) {
    // generate_state(4, np.uint64) → 8 uint32 words viewed little-endian.
    const w = seedSequenceState(seed, 8);
    const u64 = (lo: number, hi: number) => (BigInt(hi) << 32n) | BigInt(lo);
    const initstate = (u64(w[0], w[1]) << 64n) | u64(w[2], w[3]);
    const initseq = (u64(w[4], w[5]) << 64n) | u64(w[6], w[7]);
    this.state = 0n;
    this.inc = ((initseq << 1n) | 1n) & MASK128;
    this.step();
    this.state = (this.state + initstate) & MASK128;
    this.step();
  }

  private step(): void {
    this.state = (this.state * PCG_MULT_128 + this.inc) & MASK128;
  }

  /** One 64-bit output (XSL-RR), as a BigInt in [0, 2^64). */
  nextUint64(): bigint {
    this.step();
    const hi = this.state >> 64n;
    const lo = this.state & MASK64;
    const xsl = (hi ^ lo) & MASK64;
    const rot = Number(this.state >> 122n);
    if (rot === 0) return xsl;
    return ((xsl >> BigInt(rot)) | (xsl << BigInt(64 - rot))) & MASK64;
  }

  /** The bit generator's buffered 32-bit stream (low half first). */
  nextUint32(): number {
    if (this.hasUint32) {
      this.hasUint32 = false;
      return this.uinteger;
    }
    const next = this.nextUint64();
    this.hasUint32 = true;
    this.uinteger = Number(next >> 32n);
    return Number(next & 0xffffffffn);
  }

  /** `Generator.random()`: a float64 in [0, 1). */
  random(): number {
    return Number(this.nextUint64() >> 11n) / 9007199254740992;
  }

  /** `Generator.integers(n)` for 0 < n < 2^32 (Lemire's method). */
  integers(n: number): number {
    if (!Number.isInteger(n) || n <= 0 || n > MASK32) throw new Error(`numpyRandom.integers: unsupported n=${n}`);
    const rng = n - 1;
    if (rng === 0) return 0;
    if (rng === MASK32) return this.nextUint32();
    const rngExcl = BigInt(rng + 1);
    let m = BigInt(this.nextUint32()) * rngExcl;
    let leftover = Number(m & 0xffffffffn);
    if (leftover < rng + 1) {
      const threshold = Number((BigInt(MASK32) - BigInt(rng)) % rngExcl);
      while (leftover < threshold) {
        m = BigInt(this.nextUint32()) * rngExcl;
        leftover = Number(m & 0xffffffffn);
      }
    }
    return Number(m >> 32n);
  }

  /** `Generator.choice(len(p), p=p)`: one index drawn by cumulative weight. */
  choice(p: ArrayLike<number>): number {
    const n = p.length;
    const cdf = new Float64Array(n);
    let acc = 0;
    for (let i = 0; i < n; i++) {
      acc += p[i];
      cdf[i] = acc;
    }
    const total = cdf[n - 1];
    for (let i = 0; i < n; i++) cdf[i] /= total;
    const u = this.random();
    // searchsorted(side="right"): first index whose cdf value exceeds u.
    let lo = 0;
    let hi = n;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (cdf[mid] <= u) lo = mid + 1;
      else hi = mid;
    }
    return Math.min(lo, n - 1);
  }
}
