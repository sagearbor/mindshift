import { stitchSegments, StitchError } from "../src/recorder/stitch";

/** Build a minimal canonical WAV (16-bit little-endian mono PCM) from samples. */
function makeWav(samples: number[], sampleRate = 16000): Uint8Array {
  const dataLen = samples.length * 2;
  const buf = new ArrayBuffer(44 + dataLen);
  const v = new DataView(buf);
  const writeAscii = (offset: number, s: string) => {
    for (let i = 0; i < s.length; i++) v.setUint8(offset + i, s.charCodeAt(i));
  };
  writeAscii(0, "RIFF");
  v.setUint32(4, 36 + dataLen, true);
  writeAscii(8, "WAVE");
  writeAscii(12, "fmt ");
  v.setUint32(16, 16, true); // fmt chunk size
  v.setUint16(20, 1, true); // PCM
  v.setUint16(22, 1, true); // mono
  v.setUint32(24, sampleRate, true);
  v.setUint32(28, sampleRate * 2, true); // byte rate
  v.setUint16(32, 2, true); // block align
  v.setUint16(34, 16, true); // bits per sample
  writeAscii(36, "data");
  v.setUint32(40, dataLen, true);
  samples.forEach((s, i) => v.setInt16(44 + i * 2, s, true));
  return new Uint8Array(buf);
}

/** Read the 16-bit PCM samples out of a (single-header) WAV buffer. */
function readWavSamples(wav: Uint8Array): number[] {
  const v = new DataView(wav.buffer, wav.byteOffset, wav.byteLength);
  // Walk chunks from offset 12 to find "data".
  let offset = 12;
  while (offset + 8 <= wav.byteLength) {
    const id = String.fromCharCode(
      wav[offset],
      wav[offset + 1],
      wav[offset + 2],
      wav[offset + 3],
    );
    const size = v.getUint32(offset + 4, true);
    if (id === "data") {
      const samples: number[] = [];
      for (let i = 0; i < size; i += 2) {
        samples.push(v.getInt16(offset + 8 + i, true));
      }
      return samples;
    }
    offset += 8 + size + (size % 2);
  }
  throw new Error("no data chunk");
}

function ascii(bytes: Uint8Array, offset: number, len: number): string {
  return String.fromCharCode(...bytes.slice(offset, offset + len));
}

describe("stitchSegments — ADTS (Android AAC)", () => {
  it("concatenates segments byte-for-byte in order", () => {
    const a = new Uint8Array([1, 2, 3]);
    const b = new Uint8Array([4, 5]);
    const c = new Uint8Array([6]);
    const out = stitchSegments("adts", [a, b, c]);
    expect(Array.from(out)).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it("passes a single segment through intact", () => {
    const a = new Uint8Array([9, 8, 7]);
    expect(Array.from(stitchSegments("adts", [a]))).toEqual([9, 8, 7]);
  });
});

describe("stitchSegments — WAV (iOS linear PCM)", () => {
  it("merges two segments into one file with a single header and summed sizes", () => {
    const a = makeWav([100, 200]);
    const b = makeWav([300]);
    const out = stitchSegments("wav", [a, b]);
    // One RIFF/WAVE header only.
    expect(ascii(out, 0, 4)).toBe("RIFF");
    expect(ascii(out, 8, 4)).toBe("WAVE");
    const v = new DataView(out.buffer, out.byteOffset, out.byteLength);
    // RIFF size covers everything after the first 8 bytes.
    expect(v.getUint32(4, true)).toBe(out.byteLength - 8);
    // PCM data is both segments' samples, in order.
    expect(readWavSamples(out)).toEqual([100, 200, 300]);
  });

  it("preserves segment order across three segments", () => {
    const out = stitchSegments("wav", [
      makeWav([1]),
      makeWav([2]),
      makeWav([3]),
    ]);
    expect(readWavSamples(out)).toEqual([1, 2, 3]);
  });

  it("passes a single WAV segment through intact", () => {
    const a = makeWav([42, -42]);
    const out = stitchSegments("wav", [a]);
    expect(readWavSamples(out)).toEqual([42, -42]);
  });

  it("handles a segment with an extra chunk before data", () => {
    // Insert a 4-byte "LIST" chunk between fmt and data.
    const base = makeWav([7]);
    const withList = new Uint8Array(base.byteLength + 12);
    withList.set(base.slice(0, 36), 0); // through fmt chunk
    const v = new DataView(withList.buffer);
    // LIST chunk: id + size(4) + payload
    [..."LIST"].forEach((ch, i) => v.setUint8(36 + i, ch.charCodeAt(0)));
    v.setUint32(40, 4, true);
    v.setUint32(44, 0, true); // payload
    withList.set(base.slice(36), 48); // data chunk
    // Patch RIFF size for the added 12 bytes.
    v.setUint32(4, withList.byteLength - 8, true);

    const out = stitchSegments("wav", [withList, makeWav([8])]);
    expect(readWavSamples(out)).toEqual([7, 8]);
  });

  it("refuses mismatched sample rates with StitchError", () => {
    const a = makeWav([1], 16000);
    const b = makeWav([2], 44100);
    expect(() => stitchSegments("wav", [a, b])).toThrow(StitchError);
  });

  it("refuses a corrupt segment with StitchError", () => {
    const junk = new Uint8Array([0, 1, 2, 3, 4, 5]);
    expect(() => stitchSegments("wav", [makeWav([1]), junk])).toThrow(
      StitchError,
    );
  });

  it("refuses an empty segment list with StitchError", () => {
    expect(() => stitchSegments("wav", [])).toThrow(StitchError);
  });
});
