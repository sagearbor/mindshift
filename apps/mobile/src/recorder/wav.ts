/**
 * Minimal canonical WAV (RIFF/PCM) byte helpers for the v2 stream engine.
 *
 * Everything here writes the strict 44-byte canonical header (16-byte fmt
 * chunk, PCM, mono, 16-bit little-endian) — the exact shape stitch.ts parses
 * and the server's native format. No other chunk layout is ever produced.
 */

/** Canonical header: RIFF(12) + fmt(8+16) + data chunk header(8). */
export const WAV_HEADER_BYTES = 44;

/** Byte offsets of the two size fields patched after every append. */
export const RIFF_SIZE_OFFSET = 4;
export const DATA_SIZE_OFFSET = 40;

function u32le(value: number): Uint8Array {
  const out = new Uint8Array(4);
  new DataView(out.buffer).setUint32(0, value, true);
  return out;
}

/** RIFF chunk size for a canonical file with `dataBytes` of PCM payload. */
export function riffSizeBytes(dataBytes: number): Uint8Array {
  return u32le(WAV_HEADER_BYTES - 8 + dataBytes);
}

export function dataSizeBytes(dataBytes: number): Uint8Array {
  return u32le(dataBytes);
}

/** Build the full 44-byte header for mono 16-bit PCM at `sampleRate`. */
export function buildWavHeader(
  sampleRate: number,
  dataBytes: number,
): Uint8Array {
  const out = new Uint8Array(WAV_HEADER_BYTES);
  const v = new DataView(out.buffer);
  const ascii = (offset: number, s: string) => {
    for (let i = 0; i < s.length; i++) v.setUint8(offset + i, s.charCodeAt(i));
  };
  ascii(0, "RIFF");
  v.setUint32(4, WAV_HEADER_BYTES - 8 + dataBytes, true);
  ascii(8, "WAVE");
  ascii(12, "fmt ");
  v.setUint32(16, 16, true); // fmt chunk size
  v.setUint16(20, 1, true); // PCM
  v.setUint16(22, 1, true); // mono
  v.setUint32(24, sampleRate, true);
  v.setUint32(28, sampleRate * 2, true); // byte rate (16-bit mono)
  v.setUint16(32, 2, true); // block align
  v.setUint16(34, 16, true); // bits per sample
  ascii(36, "data");
  v.setUint32(40, dataBytes, true);
  return out;
}

/**
 * View int16 samples as wire bytes (s16 little-endian). Typed arrays use the
 * platform's native byte order, which is little-endian on every platform this
 * app runs on (ARM/x86) — the same fact the live-coach WebSocket path already
 * relies on when it sends `Int16Array.buffer` frames.
 */
export function int16ToBytes(samples: Int16Array): Uint8Array {
  return new Uint8Array(samples.buffer, samples.byteOffset, samples.byteLength);
}
