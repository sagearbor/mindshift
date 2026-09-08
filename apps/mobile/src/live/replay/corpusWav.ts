/**
 * Reading a RESEARCH CORPUS clip, as opposed to a fixture.
 *
 * `wav.ts` deliberately refuses anything that is not already 16 kHz mono: a
 * checked-in fixture that needs resampling is a capture bug, and hiding it
 * behind a converter is how a fixture stops representing what the phone
 * produces. A corpus is the opposite case — RAVDESS ships 48 kHz, that is
 * simply what it is, and refusing it would mean no corpus experiments at all.
 *
 * So this module exists to keep the two rules apart, and to keep the
 * decimation honest: an integer-factor box filter, deterministic, no
 * dependency. It is good enough for the 8-number prosody summaries and the
 * speaker embeddings these experiments compare — the training pipeline used
 * scipy's `resample_poly` on the same clips — and it is NOT good enough to
 * produce a fixture with. Node-only; nothing in the app graph imports it.
 */
import * as fs from "fs";
import { parseWav, REPLAY_SAMPLE_RATE } from "./wav";

/**
 * Read any mono 16-bit WAV down to 16 kHz float32 in [-1, 1).
 *
 * The source rate must be an integer multiple of 16 kHz (48 k and 32 k are;
 * 44.1 k is not) — a fractional resampler belongs in ffmpeg, not here, and
 * silently accepting one would put a different signal in front of the model
 * than the caller thinks.
 */
export function readCorpusWavTo16k(file: string): Float32Array {
  const { samples, sampleRate, channels } = parseWav(fs.readFileSync(file), file);
  if (channels < 1) throw new Error(`${file}: no channels`);
  // Downmix first, and do it properly. A corpus is not uniform: 5 of RAVDESS's
  // 1440 clips are stereo while the rest are mono, and a reader that assumes
  // mono reads those five as interleaved L,R,L,R — a signal that is not the
  // recording at all. Averaging the channels is the standard downmix and is
  // what the training pipeline's `soundfile` load did.
  const frames = Math.floor(samples.length / channels);
  const raw = new Float32Array(frames);
  for (let i = 0; i < frames; i++) {
    let acc = 0;
    for (let c = 0; c < channels; c++) acc += samples[i * channels + c];
    raw[i] = acc / channels / 32768;
  }
  if (sampleRate === REPLAY_SAMPLE_RATE) return raw;
  const factor = sampleRate / REPLAY_SAMPLE_RATE;
  if (!Number.isInteger(factor) || factor < 1) {
    throw new Error(
      `${file}: ${sampleRate} Hz is not an integer multiple of ${REPLAY_SAMPLE_RATE} — ` +
        `resample it with ffmpeg first (ffmpeg -i in.wav -ac 1 -ar 16000 out.wav)`,
    );
  }
  const outLen = Math.floor(raw.length / factor);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    let acc = 0;
    for (let k = 0; k < factor; k++) acc += raw[i * factor + k];
    out[i] = acc / factor;
  }
  return out;
}
