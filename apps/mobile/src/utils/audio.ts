/**
 * PCM helpers for the live-audio pipeline.
 *
 * Wire contract with the backend (binary WebSocket frames):
 *   raw PCM, int16 little-endian, 16,000 Hz, mono, no WAV header.
 *
 * We capture float32 from expo-audio at whatever rate/channel count the
 * hardware actually delivers, then normalise here — in JS, where every step
 * is unit-testable — before bytes ever hit the socket. Typed-array byte order
 * is the platform's native order, which is little-endian on every platform
 * React Native and browsers run on (ARM/x86), so an Int16Array's underlying
 * buffer is already wire-ready.
 */

/** Collapse interleaved multi-channel PCM ([L, R, L, R, ...]) to mono by averaging. */
export function downmixToMono(
  interleaved: Float32Array,
  channels: number,
): Float32Array {
  if (channels <= 1) {
    return interleaved;
  }
  const frames = Math.floor(interleaved.length / channels);
  const out = new Float32Array(frames);
  for (let i = 0; i < frames; i++) {
    let sum = 0;
    for (let c = 0; c < channels; c++) {
      sum += interleaved[i * channels + c];
    }
    out[i] = sum / channels;
  }
  return out;
}

/**
 * Stateful streaming resampler (linear interpolation — fine for speech).
 *
 * A stateless per-buffer resampler floors the output length of every buffer
 * and restarts its read phase at 0 each call. At non-integer ratios
 * (44.1 kHz -> 16 kHz is 2.75625) that injects a phase discontinuity at every
 * buffer boundary (~100 ms) plus cumulative drift. This class instead carries
 * the fractional read position and the unconsumed input tail across
 * `process()` calls, so a stream fed in arbitrary chunk sizes resamples
 * exactly as if it had arrived as one contiguous buffer.
 *
 * Create one instance per capture session; call `flush()` at end-of-stream to
 * collect the held-back tail samples.
 */
export class StreamingResampler {
  readonly inputRate: number;
  readonly outputRate: number;
  private readonly ratio: number;
  /** Unconsumed input samples carried over from previous process() calls. */
  private buffer: Float32Array = new Float32Array(0);
  /** Fractional read position into `buffer` for the next output sample. */
  private pos = 0;

  constructor(inputRate: number, outputRate: number) {
    if (!Number.isFinite(inputRate) || inputRate <= 0) {
      throw new Error(
        `StreamingResampler: invalid input sample rate ${inputRate}`,
      );
    }
    if (!Number.isFinite(outputRate) || outputRate <= 0) {
      throw new Error(
        `StreamingResampler: invalid output sample rate ${outputRate}`,
      );
    }
    this.inputRate = inputRate;
    this.outputRate = outputRate;
    this.ratio = inputRate / outputRate;
  }

  /** Resample the next chunk of the stream, emitting all samples that can be
   *  interpolated so far. Remaining input is held for the next call. */
  process(chunk: Float32Array): Float32Array {
    if (this.ratio === 1) {
      // Same rate: pass through, never buffers.
      return chunk;
    }
    // Append the new samples to the carried-over tail.
    if (this.buffer.length === 0) {
      this.buffer = chunk;
    } else if (chunk.length > 0) {
      const joined = new Float32Array(this.buffer.length + chunk.length);
      joined.set(this.buffer, 0);
      joined.set(chunk, this.buffer.length);
      this.buffer = joined;
    }

    const maxIndex = this.buffer.length - 1;
    // Emit every read position p = pos + k*ratio that has both neighbours
    // floor(p) and floor(p)+1 available — i.e. while p < maxIndex. Positions
    // at or past the last sample wait for the next chunk (or flush()).
    const count =
      maxIndex > this.pos ? Math.ceil((maxIndex - this.pos) / this.ratio) : 0;
    const out = new Float32Array(count);
    let p = this.pos;
    for (let i = 0; i < count; i++) {
      const i0 = Math.floor(p);
      const frac = p - i0;
      out[i] = this.buffer[i0] * (1 - frac) + this.buffer[i0 + 1] * frac;
      p += this.ratio;
    }
    // Drop fully consumed input: keep floor(p) onward (the left neighbour of
    // the next output is still needed) and carry the remainder of p. When p
    // has run past the buffer entirely, carry the overshoot into `pos`.
    const keep = Math.min(Math.floor(p), this.buffer.length);
    this.buffer = this.buffer.slice(keep);
    this.pos = p - keep;
    return out;
  }

  /** End of stream: emit the remaining read positions (clamping the right
   *  interpolation neighbour to the final sample) and reset. */
  flush(): Float32Array {
    const maxIndex = this.buffer.length - 1;
    const out: number[] = [];
    let p = this.pos;
    while (p <= maxIndex) {
      const i0 = Math.floor(p);
      const i1 = Math.min(i0 + 1, maxIndex);
      const frac = p - i0;
      out.push(this.buffer[i0] * (1 - frac) + this.buffer[i1] * frac);
      p += this.ratio;
    }
    this.buffer = new Float32Array(0);
    this.pos = 0;
    return Float32Array.from(out);
  }
}

/** Convert float32 PCM ([-1, 1]) to int16 PCM, clamping out-of-range values. */
export function float32ToInt16(input: Float32Array): Int16Array<ArrayBuffer> {
  const out = new Int16Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const clamped = Math.max(-1, Math.min(1, input[i]));
    out[i] = Math.round(clamped * 0x7fff);
  }
  return out;
}

/** Concatenate two int16 PCM buffers. */
export function concatInt16(
  a: Int16Array<ArrayBuffer>,
  b: Int16Array<ArrayBuffer>,
): Int16Array<ArrayBuffer> {
  if (a.length === 0) return b;
  if (b.length === 0) return a;
  const out = new Int16Array(a.length + b.length);
  out.set(a, 0);
  out.set(b, a.length);
  return out;
}
