/**
 * src/live/enrollFromSession.ts — a speaker's pooled session audio → a
 * canonical wav → the existing guided-enrollment upload, with the server's
 * 3 s floor enforced BEFORE anything leaves the phone.
 */
import {
  ENROLL_SAMPLE_RATE,
  MIN_ENROLL_SECONDS,
  enrollSpeakerAudio,
  pcmToWav,
  type EnrollFromSessionDeps,
} from "../src/live/enrollFromSession";
import { WAV_HEADER_BYTES } from "../src/recorder/wav";

function tone(seconds: number, amp = 0.5): Float32Array {
  const n = Math.round(seconds * ENROLL_SAMPLE_RATE);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = amp * Math.sin((2 * Math.PI * 200 * i) / ENROLL_SAMPLE_RATE);
  return out;
}

describe("pcmToWav", () => {
  it("writes a canonical 16 kHz mono 16-bit wav with clipped samples", () => {
    const pcm = new Float32Array([0, 0.5, -0.5, 1.5, -1.5]);
    const wav = pcmToWav(pcm);
    expect(wav.length).toBe(WAV_HEADER_BYTES + pcm.length * 2);
    const v = new DataView(wav.buffer, wav.byteOffset, wav.byteLength);
    expect(String.fromCharCode(...wav.subarray(0, 4))).toBe("RIFF");
    expect(String.fromCharCode(...wav.subarray(8, 12))).toBe("WAVE");
    expect(v.getUint32(24, true)).toBe(16000); // sample rate
    expect(v.getUint16(22, true)).toBe(1); // mono
    expect(v.getUint16(34, true)).toBe(16); // bits
    expect(v.getUint32(40, true)).toBe(pcm.length * 2); // data bytes
    const s = new Int16Array(wav.buffer.slice(WAV_HEADER_BYTES));
    expect(Array.from(s)).toEqual([0, 16384, -16384, 32767, -32768]);
  });
});

describe("enrollSpeakerAudio", () => {
  const deps = (): EnrollFromSessionDeps & { saved: string[]; enrolled: unknown[] } => {
    const d = {
      saved: [] as string[],
      enrolled: [] as unknown[],
      saveWav: async (bytes: Uint8Array, name: string) => {
        d.saved.push(`${name}:${bytes.length}`);
        return `file:///cache/${name}`;
      },
      enroll: async (file: string | File, name: string, person: { personId: string; displayName?: string | null }) => {
        d.enrolled.push({ file, name, person });
        return { enrolled: true, enroll_count: 1, dim: 192, updated_at: "now", stored: "a numeric voice signature" };
      },
    };
    return d;
  };

  it("refuses a pool shorter than the server's floor without uploading", async () => {
    const d = deps();
    await expect(
      enrollSpeakerAudio(tone(MIN_ENROLL_SECONDS - 0.5), { personId: "mom", displayName: "Mom" }, d),
    ).rejects.toThrow(/2\.5 s of Mom's voice/);
    expect(d.saved).toHaveLength(0);
    expect(d.enrolled).toHaveLength(0);
  });

  it("saves the wav and enrolls the named person through the guided endpoint", async () => {
    const d = deps();
    const result = await enrollSpeakerAudio(tone(4.2), { personId: "mom", displayName: "Mom" }, d);
    expect(result).toEqual({ enrollCount: 1, seconds: 4.2 });
    expect(d.saved).toHaveLength(1);
    expect(d.saved[0]).toMatch(/^live-enroll-mom-\d+\.wav:\d+$/);
    expect(Number(d.saved[0].split(":")[1])).toBe(WAV_HEADER_BYTES + Math.round(4.2 * 16000) * 2);
    expect(d.enrolled[0]).toMatchObject({
      file: expect.stringMatching(/^file:\/\/\/cache\/live-enroll-mom-/),
      person: { personId: "mom", displayName: "Mom" },
    });
  });

  it("propagates the server's refusal (status + detail) untouched", async () => {
    const d = deps();
    d.enroll = async () => {
      const err = new Error("not enough speech in the clip") as Error & { status?: number };
      err.status = 422;
      throw err;
    };
    await expect(enrollSpeakerAudio(tone(5), { personId: "mom", displayName: "Mom" }, d)).rejects.toMatchObject({
      status: 422,
      message: "not enough speech in the clip",
    });
  });
});
