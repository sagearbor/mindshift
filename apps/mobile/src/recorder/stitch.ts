import type { SegmentContainer } from "./types";

/** A segment could not be combined — corrupt data or incompatible formats.
 *  Callers surface this honestly; segments stay on disk for another attempt. */
export class StitchError extends Error {}

interface WavInfo {
  /** The full fmt chunk payload (must match across segments). */
  fmt: Uint8Array;
  /** The raw PCM payload of the data chunk. */
  data: Uint8Array;
}

function ascii(bytes: Uint8Array, offset: number): string {
  return String.fromCharCode(
    bytes[offset],
    bytes[offset + 1],
    bytes[offset + 2],
    bytes[offset + 3],
  );
}

/** Walk a WAV file's chunks and pull out fmt + data. Throws StitchError on
 *  anything that isn't a readable RIFF/WAVE file. */
function parseWav(bytes: Uint8Array): WavInfo {
  if (bytes.byteLength < 44) throw new StitchError("WAV segment too short");
  if (ascii(bytes, 0) !== "RIFF" || ascii(bytes, 8) !== "WAVE") {
    throw new StitchError("Not a WAV segment");
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let fmt: Uint8Array | null = null;
  let data: Uint8Array | null = null;
  let offset = 12;
  while (offset + 8 <= bytes.byteLength) {
    const id = ascii(bytes, offset);
    const size = view.getUint32(offset + 4, true);
    const payloadStart = offset + 8;
    // A truncated final chunk is corruption, except we tolerate a data chunk
    // whose declared size overruns the file (a crash mid-write) by clamping.
    const available = bytes.byteLength - payloadStart;
    if (id === "fmt ") {
      if (size > available) throw new StitchError("Truncated fmt chunk");
      fmt = bytes.slice(payloadStart, payloadStart + size);
    } else if (id === "data") {
      data = bytes.slice(payloadStart, payloadStart + Math.min(size, available));
    }
    offset = payloadStart + size + (size % 2);
  }
  if (!fmt || !data) throw new StitchError("WAV segment missing fmt/data");
  return { fmt, data };
}

function sameBytes(a: Uint8Array, b: Uint8Array): boolean {
  if (a.byteLength !== b.byteLength) return false;
  for (let i = 0; i < a.byteLength; i++) if (a[i] !== b[i]) return false;
  return true;
}

function writeAscii(view: DataView, offset: number, s: string): void {
  for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i));
}

/** Rebuild a single canonical WAV from a shared fmt chunk + PCM payloads. */
function buildWav(fmt: Uint8Array, chunks: Uint8Array[]): Uint8Array {
  const dataLen = chunks.reduce((sum, c) => sum + c.byteLength, 0);
  const headerLen = 12 + 8 + fmt.byteLength + 8;
  const out = new Uint8Array(headerLen + dataLen);
  const view = new DataView(out.buffer);
  writeAscii(view, 0, "RIFF");
  view.setUint32(4, out.byteLength - 8, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, fmt.byteLength, true);
  out.set(fmt, 20);
  const dataOffset = 20 + fmt.byteLength;
  writeAscii(view, dataOffset, "data");
  view.setUint32(dataOffset + 4, dataLen, true);
  let cursor = dataOffset + 8;
  for (const c of chunks) {
    out.set(c, cursor);
    cursor += c.byteLength;
  }
  return out;
}

/**
 * Combine finished segments into ONE playable/uploadable audio file.
 *
 * - `adts`: AAC-ADTS streams are self-synchronizing — plain byte concatenation
 *   is a valid stream.
 * - `wav`: strip each segment's header, join the PCM, emit a single header
 *   with patched sizes. Refuses (StitchError) if segments disagree on format —
 *   silently mixing sample rates would corrupt the audio.
 */
export function stitchSegments(
  format: SegmentContainer,
  segments: Uint8Array[],
): Uint8Array {
  if (segments.length === 0) {
    throw new StitchError("No segments to stitch");
  }
  if (format === "adts") {
    const total = segments.reduce((sum, s) => sum + s.byteLength, 0);
    const out = new Uint8Array(total);
    let cursor = 0;
    for (const s of segments) {
      out.set(s, cursor);
      cursor += s.byteLength;
    }
    return out;
  }
  const parsed = segments.map(parseWav);
  const fmt = parsed[0].fmt;
  for (const p of parsed) {
    if (!sameBytes(p.fmt, fmt)) {
      throw new StitchError("WAV segments have mismatched formats");
    }
  }
  return buildWav(
    fmt,
    parsed.map((p) => p.data),
  );
}
