import { Platform } from "react-native";
import {
  isCompressionAvailable,
  compressVideo,
} from "../src/utils/videoCompression";

// The native compressor + expo-file-system are mocked in jest-setup.ts:
// `Video.compress` reports full progress and returns `${uri}.compressed.mp4`.

const originalOS = Platform.OS;

function setOS(os: string) {
  Object.defineProperty(Platform, "OS", { value: os, configurable: true });
}

afterEach(() => {
  Object.defineProperty(Platform, "OS", {
    value: originalOS,
    configurable: true,
  });
});

describe("videoCompression platform gate", () => {
  it("is available on native (ios/android)", () => {
    setOS("ios");
    expect(isCompressionAvailable()).toBe(true);
    setOS("android");
    expect(isCompressionAvailable()).toBe(true);
  });

  it("is unavailable on web, and compressVideo refuses there (module gated out)", async () => {
    setOS("web");
    expect(isCompressionAvailable()).toBe(false);
    await expect(compressVideo("file:///x.mp4")).rejects.toThrow(
      /not available on web/,
    );
  });
});

describe("videoCompression native path", () => {
  it("compresses via the native module, forwarding progress and returning the compressed uri", async () => {
    setOS("ios");
    const onProgress = jest.fn();
    const result = await compressVideo("file:///clip.mp4", { onProgress });
    // The mocked native compressor returns a derived compressed uri.
    expect(result.uri).toBe("file:///clip.mp4.compressed.mp4");
    // Progress was forwarded to the caller (mock reports completion).
    expect(onProgress).toHaveBeenCalledWith(1);
  });
});
