/**
 * RIFF/WAVE → PCM parser for the app (no Node `Buffer`): the 16 kHz mono
 * int16 WAV the server's `GET /recordings/{id}/media?format=pcm16k`
 * transcodes a stored recording to, read back on the phone for the on-device
 * voice-separation engine (live/deviceDiarization.ts). Walks the chunk list
 * (a `LIST`/`fact` chunk before `data` is fine), rejects anything that is
 * not PCM int16, and converts with the fast loop's own int16 → float32
 * mapping (/32768).
 */

export interface ParsedWav {
  channels: number;
  sampleRate: number;
  bitsPerSample: number;
  /** Interleaved int16 samples (little-endian on the wire). */
  samples: Int16Array;
  seconds: number;
}

function ascii(view: DataView, offset: number, length: number): string {
  let s = "";
  for (let i = 0; i < length; i++) s += String.fromCharCode(view.getUint8(offset + i));
  return s;
}

export function parseWav(bytes: Uint8Array, label = "wav"): ParsedWav {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (bytes.byteLength < 12 || ascii(view, 0, 4) !== "RIFF" || ascii(view, 8, 4) !== "WAVE") {
    throw new Error(`${label}: not a RIFF/WAVE file`);
  }
  let offset = 12;
  let channels = 0;
  let sampleRate = 0;
  let bits = 0;
  let format = 0;
  let dataStart = -1;
  let dataLength = 0;
  while (offset + 8 <= bytes.byteLength) {
    const id = ascii(view, offset, 4);
    const size = view.getUint32(offset + 4, true);
    const body = offset + 8;
    if (id === "fmt ") {
      format = view.getUint16(body, true);
      channels = view.getUint16(body + 2, true);
      sampleRate = view.getUint32(body + 4, true);
      bits = view.getUint16(body + 14, true);
    } else if (id === "data") {
      dataStart = body;
      dataLength = Math.min(size, bytes.byteLength - body);
      break;
    }
    offset = body + size + (size % 2);
  }
  if (dataStart < 0) throw new Error(`${label}: no data chunk`);
  if (format !== 1 || bits !== 16) throw new Error(`${label}: expected PCM int16 (format ${format}, ${bits}-bit)`);
  const n = Math.floor(dataLength / 2);
  const samples = new Int16Array(n);
  for (let i = 0; i < n; i++) samples[i] = view.getInt16(dataStart + i * 2, true);
  const frames = channels > 0 ? Math.floor(n / channels) : 0;
  return { channels, sampleRate, bitsPerSample: bits, samples, seconds: sampleRate > 0 ? frames / sampleRate : 0 };
}

/** The engine's contract: 16 kHz mono. Anything else is an honest error. */
export function parseWav16kMono(bytes: Uint8Array, label = "wav"): Float32Array {
  const wav = parseWav(bytes, label);
  if (wav.channels !== 1 || wav.sampleRate !== 16000) {
    throw new Error(`${label}: expected 16 kHz mono, got ${wav.channels}ch ${wav.sampleRate} Hz`);
  }
  const out = new Float32Array(wav.samples.length);
  for (let i = 0; i < out.length; i++) out[i] = wav.samples[i] / 32768;
  return out;
}
