import { Platform } from "react-native";
import { AudioModule } from "expo-audio";
import type { RecordingOptions } from "expo-audio";
import type { RecorderPort, SegmentFormat } from "./types";

/**
 * Per-platform segment format. The choice is driven by STITCHABILITY — every
 * session (not just crashed ones) ends by concatenating its segments into one
 * uploadable file, so the container must support lossless client-side joins:
 *
 * - Android: AAC in an ADTS stream (MediaRecorder `aac_adts`) — ADTS frames
 *   are self-synchronizing, so segments byte-concatenate into a valid stream.
 *   64 kbps mono ≈ 8 KB/s → a 1-hour session is ~29 MB.
 * - iOS: 16 kHz mono 16-bit linear PCM in WAV — AVAudioRecorder can't write
 *   ADTS, but WAV joins are a trivial header patch. ~32 KB/s → a 1-hour
 *   session is ~115 MB (chunked upload path handles it; 16 kHz is the same
 *   rate the live-coach transcription pipeline uses).
 */
export function formatForPlatform(os: string): SegmentFormat {
  if (os === "android") {
    return {
      format: "adts",
      extension: ".aac",
      mimeType: "audio/aac",
      bytesPerSecond: 8500, // 64 kbps AAC + ADTS overhead
    };
  }
  return {
    format: "wav",
    extension: ".wav",
    mimeType: "audio/wav",
    bytesPerSecond: 32100, // 16 kHz * 2 bytes mono + headers
  };
}

/** expo-audio RecordingOptions for one segment in the given format. */
export function recordingOptionsFor(format: SegmentFormat): RecordingOptions {
  return {
    extension: format.extension,
    sampleRate: 16000,
    numberOfChannels: 1,
    bitRate: 64000,
    android: {
      extension: ".aac",
      outputFormat: "aac_adts",
      audioEncoder: "aac",
    },
    ios: {
      extension: ".wav",
      outputFormat: "lpcm", // IOSOutputFormat.LINEARPCM
      audioQuality: 96, // AudioQuality.HIGH
      linearPCMBitDepth: 16,
      linearPCMIsBigEndian: false,
      linearPCMIsFloat: false,
    },
    // The recorder itself is native-only (the screen shows an honest note on
    // web), but the type requires the key.
    web: {},
  };
}

/**
 * The native AudioRecorder expects FLAT options: expo-audio's useAudioRecorder
 * hoists the platform sub-object to the top level (createRecordingOptions)
 * before constructing. Passing the nested RecordingOptions shape straight to
 * the native layer made prepare throw on-device ("Couldn't start the
 * microphone" — v1.16.0's first field bug), while jest fakes never noticed.
 */
export function flattenForNative(
  options: RecordingOptions,
  os: string,
): Record<string, unknown> {
  const common = {
    extension: options.extension,
    sampleRate: options.sampleRate,
    numberOfChannels: options.numberOfChannels,
    bitRate: options.bitRate,
    isMeteringEnabled: options.isMeteringEnabled ?? false,
  };
  if (os === "ios") return { ...common, ...options.ios };
  if (os === "android") return { ...common, ...options.android };
  return { ...common, ...options.web };
}

/**
 * RecorderPort over expo-audio's imperative AudioRecorder. One instance per
 * segment: prepare() names a fresh cache file, stop() finalizes it and hands
 * back the uri for the store to move into durable session storage.
 */
export class ExpoAudioRecorderPort implements RecorderPort {
  private recorder: InstanceType<typeof AudioModule.AudioRecorder>;
  private nativeOptions: Record<string, unknown>;

  constructor(options: RecordingOptions) {
    this.nativeOptions = flattenForNative(options, Platform.OS);
    this.recorder = new AudioModule.AudioRecorder(this.nativeOptions);
  }

  async prepare(): Promise<void> {
    await this.recorder.prepareToRecordAsync(this.nativeOptions);
  }

  start(): void {
    this.recorder.record();
  }

  async stop(): Promise<{ uri: string | null }> {
    await this.recorder.stop();
    return { uri: this.recorder.uri };
  }

  isRecording(): boolean {
    return this.recorder.isRecording;
  }

  release(): void {
    try {
      this.recorder.release();
    } catch {
      // A recorder invalidated by the OS (media-services reset) may throw on
      // release; there is nothing further to free.
    }
  }
}

export function makeExpoRecorderFactory(): () => RecorderPort {
  const format = formatForPlatform(Platform.OS);
  const options = recordingOptionsFor(format);
  return () => new ExpoAudioRecorderPort(options);
}
