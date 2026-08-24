// Metro config: the default Expo config plus `.onnx` as a bundled asset so
// the on-device fast loop's Silero VAD model (assets/models/silero_vad.onnx)
// ships inside the app and resolves through expo-asset at runtime
// (see src/live/ortNative.ts).
const { getDefaultConfig } = require("expo/metro-config");

const config = getDefaultConfig(__dirname);

if (!config.resolver.assetExts.includes("onnx")) {
  config.resolver.assetExts.push("onnx");
}

module.exports = config;
