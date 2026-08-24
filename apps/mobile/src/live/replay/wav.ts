/**
 * Minimal RIFF/WAVE reader for the replay harness: 16 kHz mono int16 only —
 * the ECAPA/Silero contract rate and the shape every checked-in fixture and
 * every phone capture the harness accepts must already be in. Anything else
 * is rejected with the ffmpeg one-liner that produces the right file; a
 * resampler here would only hide a capture that the phone itself would not
 * have produced.
 *
 * Node-only (fs); nothing in the app graph imports this.
 */
import * as fs from "fs";

export const REPLAY_SAMPLE_RATE = 16000;

export interface WavInfo {
  channels: number;
  sampleRate: number;
  bitsPerSample: number;
  /** PCM frames (samples per channel). */
  frames: number;
  seconds: number;
}

export interface WavPcm extends WavInfo {
  samples: Int16Array;
}

/** Parse a RIFF/WAVE buffer (PCM int16, any channel count / rate). */
export function parseWav(buf: Buffer, label = "wav"): WavPcm {
  if (buf.length < 12 || buf.toString("ascii", 0, 4) !== "RIFF" || buf.toString("ascii", 8, 12) !== "WAVE") {
    throw new Error(`${label}: not a RIFF/WAVE file`);
  }
  let offset = 12;
  let channels = 0;
  let sampleRate = 0;
  let bits = 0;
  let format = 0;
  let data: Buffer | null = null;
  while (offset + 8 <= buf.length) {
    const id = buf.toString("ascii", offset, offset + 4);
    const size = buf.readUInt32LE(offset + 4);
    const body = offset + 8;
    if (id === "fmt ") {
      format = buf.readUInt16LE(body);
      channels = buf.readUInt16LE(body + 2);
      sampleRate = buf.readUInt32LE(body + 4);
      bits = buf.readUInt16LE(body + 14);
    } else if (id === "data") {
      data = buf.subarray(body, Math.min(body + size, buf.length));
    }
    offset = body + size + (size % 2);
  }
  if (!data) throw new Error(`${label}: no data chunk`);
  if (format !== 1 || bits !== 16) {
    throw new Error(`${label}: expected PCM int16 (format ${format}, ${bits}-bit)`);
  }
  const n = Math.floor(data.length / 2);
  // Explicit little-endian reads: a byte view over the Buffer would be
  // host-endian and would also pin the array to Node's realm (see
  // testing/ortNode.ts for why the realm matters under Jest).
  const samples = new Int16Array(n);
  for (let i = 0; i < n; i++) samples[i] = data.readInt16LE(i * 2);
  const frames = channels > 0 ? Math.floor(n / channels) : 0;
  return {
    channels,
    sampleRate,
    bitsPerSample: bits,
    frames,
    seconds: sampleRate > 0 ? frames / sampleRate : 0,
    samples,
  };
}

/** Read a 16 kHz mono int16 WAV into an Int16Array; throws with a fix hint. */
export function readWav16kMono(file: string): Int16Array {
  const wav = parseWav(fs.readFileSync(file), file);
  if (wav.channels !== 1 || wav.sampleRate !== REPLAY_SAMPLE_RATE) {
    throw new Error(
      `${file}: expected 16 kHz mono int16, got ${wav.channels}ch ${wav.sampleRate} Hz ${wav.bitsPerSample}-bit. ` +
        `Convert with: ffmpeg -i <input> -ac 1 -ar 16000 -sample_fmt s16 <out>.wav`,
    );
  }
  return wav.samples;
}

/** int16 -> float32 in [-1, 1) via /32768 — the fast loop's own conversion. */
export function int16ToFloat32(samples: Int16Array): Float32Array {
  const out = new Float32Array(samples.length);
  for (let i = 0; i < samples.length; i++) out[i] = samples[i] / 32768;
  return out;
}
