/**
 * Byte math for the guided "Train my voice" flow: a few short in-memory phrase
 * takes concatenated into ONE canonical wav for POST /voice/enroll-direct.
 *
 * Deliberately NOT the session engine: phrase takes are ~5 seconds each, so
 * crash-durable segment files, manifests and recovery would be machinery
 * without a payoff — the honest failure mode for a killed app mid-phrase is
 * simply recording the phrase again. Capture still comes through the same
 * PcmSource seam the v2 engine uses (same mic, same downmix, same honest
 * per-frame sample rate), and the wav bytes come from the same wav.ts helpers.
 */

import { WAV_HEADER_BYTES, buildWavHeader, int16ToBytes } from "./wav";

/** One phrase's captured audio: raw int16 frames at the rate the hardware
 *  actually delivered (a per-frame fact — see PcmFrame.sampleRate). */
export interface PhraseTake {
  chunks: Int16Array[];
  sampleRate: number;
}

export function takeSampleCount(take: PhraseTake): number {
  return take.chunks.reduce((sum, c) => sum + c.length, 0);
}

export function takeDurationMs(take: PhraseTake): number {
  if (take.sampleRate <= 0) return 0;
  return Math.round((takeSampleCount(take) / take.sampleRate) * 1000);
}

/**
 * Concatenate every take's samples into one canonical mono 16-bit wav whose
 * header states the takes' TRUE shared sample rate.
 *
 * Honesty over convenience: takes recorded at different hardware rates cannot
 * be byte-concatenated without silently detuning someone's voice — that is an
 * error (the flow asks the user to start over), never a resample-by-lying.
 */
export function concatTakesToWav(takes: PhraseTake[]): Uint8Array {
  const nonEmpty = takes.filter((t) => takeSampleCount(t) > 0);
  const total = nonEmpty.reduce((sum, t) => sum + takeSampleCount(t), 0);
  if (nonEmpty.length === 0 || total === 0) {
    throw new Error("no audio was captured");
  }
  const rate = nonEmpty[0].sampleRate;
  if (nonEmpty.some((t) => t.sampleRate !== rate)) {
    throw new Error(
      "takes were captured at different sample rates and cannot be merged",
    );
  }
  const out = new Uint8Array(WAV_HEADER_BYTES + total * 2);
  out.set(buildWavHeader(rate, total * 2), 0);
  let offset = WAV_HEADER_BYTES;
  for (const take of nonEmpty) {
    for (const chunk of take.chunks) {
      out.set(int16ToBytes(chunk), offset);
      offset += chunk.byteLength;
    }
  }
  return out;
}
