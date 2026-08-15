import {
  concatTakesToWav,
  takeDurationMs,
  takeSampleCount,
  type PhraseTake,
} from "../src/recorder/guidedCapture";
import { WAV_HEADER_BYTES } from "../src/recorder/wav";

function take(chunks: number[][], sampleRate = 16000): PhraseTake {
  return { chunks: chunks.map((c) => Int16Array.from(c)), sampleRate };
}

describe("guidedCapture — take math", () => {
  it("counts samples across chunks and derives duration honestly", () => {
    const t = take([[1, 2, 3], [4, 5]], 16000);
    expect(takeSampleCount(t)).toBe(5);
    // 16000 samples = 1000ms → 5 samples ≈ 0ms (rounds); use a bigger take.
    const big = { chunks: [new Int16Array(8000)], sampleRate: 16000 };
    expect(takeDurationMs(big)).toBe(500);
  });
});

describe("guidedCapture — concatTakesToWav", () => {
  it("builds one canonical wav: header rate + sizes + concatenated payload", () => {
    const wav = concatTakesToWav([
      take([[1, 2], [3]]),
      take([[4, 5, 6]]),
    ]);
    const v = new DataView(wav.buffer, wav.byteOffset, wav.byteLength);
    // RIFF/WAVE magic.
    expect(String.fromCharCode(wav[0], wav[1], wav[2], wav[3])).toBe("RIFF");
    expect(String.fromCharCode(wav[8], wav[9], wav[10], wav[11])).toBe("WAVE");
    // Header declares the takes' true rate and the exact payload size.
    expect(v.getUint32(24, true)).toBe(16000);
    expect(v.getUint32(40, true)).toBe(6 * 2);
    expect(wav.byteLength).toBe(WAV_HEADER_BYTES + 6 * 2);
    // Payload is the takes' samples in order, 16-bit little-endian.
    const payload = new Int16Array(6);
    for (let i = 0; i < 6; i++) {
      payload[i] = v.getInt16(WAV_HEADER_BYTES + i * 2, true);
    }
    expect(Array.from(payload)).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it("stamps a non-16k hardware rate honestly instead of lying at 16k", () => {
    const wav = concatTakesToWav([take([[9, 9]], 48000)]);
    const v = new DataView(wav.buffer, wav.byteOffset, wav.byteLength);
    expect(v.getUint32(24, true)).toBe(48000);
  });

  it("refuses to merge takes recorded at different sample rates", () => {
    expect(() =>
      concatTakesToWav([take([[1]], 16000), take([[2]], 48000)]),
    ).toThrow(/sample rate/i);
  });

  it("refuses to build an empty wav", () => {
    expect(() => concatTakesToWav([])).toThrow(/no audio/i);
    expect(() => concatTakesToWav([take([[]])])).toThrow(/no audio/i);
  });
});
