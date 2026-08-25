/**
 * "Remember this voice" from a LIVE session's own audio.
 *
 * The fast loop keeps the last ~20 s of each speaker's VAD-cut speech on the
 * phone (`FastLoop.speakerAudio`). When the user names a speaker mid-call
 * as a NEW person, that pooled PCM is written as a canonical 16 kHz mono
 * wav and uploaded through the EXISTING guided-enrollment endpoint
 * (`POST /voice/enroll-direct`, api/client.ts `enrollVoiceDirect`) with the
 * person id + display name, which creates the person and stores their
 * voiceprint — so the next session recognizes them from the first turn.
 *
 * The server refuses fewer than 3 s of actual speech (its
 * MIN_ENROLL_SECONDS); this module pre-checks the same floor so a too-short
 * pool never leaves the phone. Everything platform-specific is injectable
 * (`EnrollFromSessionDeps`) so the flow is unit-tested without native code.
 */
import { Platform } from "react-native";
import { enrollVoiceDirect, type DirectEnrollResult } from "../api/client";
import { buildWavHeader, int16ToBytes, WAV_HEADER_BYTES } from "../recorder/wav";

/** Same floor as the server's speaker_id.MIN_ENROLL_SECONDS. */
export const MIN_ENROLL_SECONDS = 3;
export const ENROLL_SAMPLE_RATE = 16000;

export interface EnrollFromSessionDeps {
  /** Persist the wav; returns the handle enrollVoiceDirect accepts
   *  (a file URI on native, a File on web). */
  saveWav: (bytes: Uint8Array, name: string) => Promise<string | File>;
  enroll: (
    file: string | File,
    name: string,
    person: { personId: string; displayName?: string | null },
  ) => Promise<DirectEnrollResult>;
}

export interface EnrollFromSessionResult {
  enrollCount: number;
  /** Seconds of pooled speech that were uploaded. */
  seconds: number;
}

/** float32 [-1, 1) PCM → canonical mono 16-bit wav bytes. */
export function pcmToWav(pcm: Float32Array, sampleRate = ENROLL_SAMPLE_RATE): Uint8Array {
  const int16 = new Int16Array(pcm.length);
  for (let i = 0; i < pcm.length; i++) {
    const v = Math.max(-1, Math.min(1, pcm[i]));
    int16[i] = v < 0 ? Math.round(v * 32768) : Math.round(v * 32767);
  }
  const out = new Uint8Array(WAV_HEADER_BYTES + int16.byteLength);
  out.set(buildWavHeader(sampleRate, int16.byteLength), 0);
  out.set(int16ToBytes(int16), WAV_HEADER_BYTES);
  return out;
}

export function defaultEnrollDeps(): EnrollFromSessionDeps {
  return {
    saveWav: async (bytes, name) => {
      if (Platform.OS === "web") {
        return new File([bytes as BlobPart], name, { type: "audio/wav" });
      }
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      const { File: FSFile, Paths } = require("expo-file-system") as typeof import("expo-file-system");
      const base: string = Paths.cache.uri;
      const uri = `${base.endsWith("/") ? base : `${base}/`}${name}`;
      new FSFile(uri).write(bytes);
      return uri;
    },
    enroll: (file, name, person) => enrollVoiceDirect(file, name, person),
  };
}

/**
 * Upload a speaker's pooled session audio as a voiceprint sample for
 * `person`. Throws (with the server's `.status`/detail when it answered)
 * on refusal; the caller renders the honest reason. Throws a plain Error
 * with `.status = 0`-free message when the pool is too short — checked
 * here so nothing is uploaded that the server would refuse anyway.
 */
export async function enrollSpeakerAudio(
  pcm: Float32Array,
  person: { personId: string; displayName: string },
  deps: EnrollFromSessionDeps = defaultEnrollDeps(),
  sampleRate = ENROLL_SAMPLE_RATE,
): Promise<EnrollFromSessionResult> {
  const seconds = pcm.length / sampleRate;
  if (seconds < MIN_ENROLL_SECONDS) {
    throw new Error(
      `only ${seconds.toFixed(1)} s of ${person.displayName}'s voice so far — at least ` +
        `${MIN_ENROLL_SECONDS} s is needed to remember a voice`,
    );
  }
  const name = `live-enroll-${person.personId}-${Date.now()}.wav`;
  const file = await deps.saveWav(pcmToWav(pcm, sampleRate), name);
  const result = await deps.enroll(file, name, {
    personId: person.personId,
    displayName: person.displayName,
  });
  return { enrollCount: result.enroll_count, seconds: Math.round(seconds * 10) / 10 };
}
