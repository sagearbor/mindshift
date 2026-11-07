/**
 * The one {@link EmotionDevHub} the live-session hook and the Settings panel
 * share. The heavy native wiring (onnxruntime-react-native, the file system,
 * the watch link) is loaded lazily, only once the mode is switched on — so
 * with the mode off nothing beyond this file is even evaluated, and the
 * existing jest suites that import the live hook never pull in native ORT.
 */
import { EmotionDevHub } from "./devHub";
import type { EmotionDevStore } from "./devStore";

export const emotionDevHub = new EmotionDevHub(
  async (settings) => {
    const { buildNativeRuntime } = await import("./nativeRuntime");
    return buildNativeRuntime(settings);
  },
  (): EmotionDevStore | null => {
    try {
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      return (require("./nativeRuntime") as typeof import("./nativeRuntime")).openNativeStore();
    } catch {
      return null;
    }
  },
);
