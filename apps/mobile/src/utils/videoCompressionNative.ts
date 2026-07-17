import { Video } from "react-native-compressor";
import { File as FSFile } from "expo-file-system";
import type { CompressionResult, CompressOptions } from "./videoCompression";

/**
 * Native on-device video compression, backed by `react-native-compressor`.
 *
 * This module imports the native compressor and MUST NOT be imported on web —
 * it's reached only through `videoCompression.compressVideo`, which `require()`s
 * it lazily off-web (see that file's doc comment). `compressionMethod: "auto"`
 * lets the library pick a target bitrate/resolution that typically cuts a phone
 * video by ~80–90% while staying watchable for transcription; the callback
 * reports a 0→1 progress fraction.
 */
export async function compressVideo(
  uri: string,
  options?: CompressOptions,
): Promise<CompressionResult> {
  const outUri = await Video.compress(
    uri,
    { compressionMethod: "auto" },
    (progress) => {
      options?.onProgress?.(progress);
    },
  );
  // Best-effort size so the caller can route direct-vs-chunked honestly; a stat
  // failure just leaves it undefined (caller falls back to the direct path).
  let size: number | undefined;
  try {
    const statted = new FSFile(outUri).size;
    if (typeof statted === "number" && statted > 0) size = statted;
  } catch {
    // ignore — undefined size is handled by the caller
  }
  return { uri: outUri, size };
}
