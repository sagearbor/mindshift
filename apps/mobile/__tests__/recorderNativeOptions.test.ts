/**
 * The native AudioRecorder expects FLATTENED options — expo-audio's
 * useAudioRecorder hoists the platform sub-object to the top level via
 * createRecordingOptions before constructing. Passing the nested
 * RecordingOptions shape straight through made prepare throw on-device
 * ("Couldn't start the microphone", v1.16.0's first field bug). These tests
 * pin the flattening our adapter must perform.
 */
import {
  flattenForNative,
  formatForPlatform,
  recordingOptionsFor,
} from "../src/recorder/expoRecorderPort";

const androidOptions = recordingOptionsFor(formatForPlatform("android"));
const iosOptions = recordingOptionsFor(formatForPlatform("ios"));

describe("flattenForNative", () => {
  it("hoists android keys to the top level for the native constructor", () => {
    const flat = flattenForNative(androidOptions, "android");
    expect(flat.outputFormat).toBe("aac_adts");
    expect(flat.audioEncoder).toBe("aac");
    expect(flat.extension).toBe(".aac");
    expect(flat.sampleRate).toBe(16000);
    expect(flat.numberOfChannels).toBe(1);
    // The nested platform objects must NOT survive into the native dict.
    expect(flat.android).toBeUndefined();
    expect(flat.ios).toBeUndefined();
  });

  it("hoists ios keys to the top level for the native constructor", () => {
    const flat = flattenForNative(iosOptions, "ios");
    expect(flat.outputFormat).toBe("lpcm");
    expect(flat.linearPCMBitDepth).toBe(16);
    expect(flat.extension).toBe(".wav");
    expect(flat.android).toBeUndefined();
    expect(flat.ios).toBeUndefined();
  });

  it("defaults isMeteringEnabled false, mirroring createRecordingOptions", () => {
    expect(flattenForNative(androidOptions, "android").isMeteringEnabled).toBe(
      false,
    );
  });
});
