import { Platform } from "react-native";

/**
 * On-device video compression, platform-gated exactly like RecordScreen.
 *
 * `react-native-compressor` is a NATIVE module with no web implementation —
 * importing it in a web bundle pulls in native bindings that don't exist there
 * and can blank the app at load (the same failure class as the RecordScreen web
 * blank-screen bug). So the real implementation lives in
 * `videoCompressionNative.ts` and is `require()`d ONLY off-web, at call time;
 * the web build never executes that module. This wrapper imports nothing but
 * `react-native`, so it is always safe to import (incl. in web bundles and
 * under jest).
 */

/** Outcome of a successful compression: the compressed file's local URI and,
 *  best-effort, its size in bytes (undefined when it couldn't be stat'd). The
 *  size drives the caller's direct-vs-chunked upload routing. */
export interface CompressionResult {
  uri: string;
  size?: number;
}

export interface CompressOptions {
  /** Called with a 0→1 fraction as compression advances, for a "Compressing…
   *  n%" progress UI. */
  onProgress?: (fraction: number) => void;
}

/**
 * Whether on-device compression is available on this platform. Web has no
 * native compressor, so it is always false there — callers upload the original
 * unchanged on web. On native it's available (the module is bundled via EAS
 * builds, not Expo Go).
 */
export function isCompressionAvailable(): boolean {
  return Platform.OS !== "web";
}

/**
 * Compress a local video file, returning the compressed copy's URI (and size
 * when known). Throws on web (guard with {@link isCompressionAvailable} first)
 * and rethrows any native compression failure so the caller can offer an honest
 * fallback (upload the original) rather than silently dropping the upload.
 *
 * The native module is resolved lazily via `require()` so it never lands in a
 * web bundle (see the module doc comment).
 */
export async function compressVideo(
  uri: string,
  options?: CompressOptions,
): Promise<CompressionResult> {
  if (Platform.OS === "web") {
    throw new Error("Video compression is not available on web.");
  }
  // Lazy, native-only module resolution (see doc comment above).
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const native = require("./videoCompressionNative") as {
    compressVideo: (
      uri: string,
      options?: CompressOptions,
    ) => Promise<CompressionResult>;
  };
  return native.compressVideo(uri, options);
}
