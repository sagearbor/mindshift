import { Platform } from "react-native";
import { setAudioModeAsync } from "expo-audio";
import { setRecordingMode, setPlaybackMode } from "../src/utils/audioMode";

// Minimal expo-audio mock: we only care that the right session config is
// requested (or, on web, that no native call is made at all).
jest.mock("expo-audio", () => ({
  __esModule: true,
  setAudioModeAsync: jest.fn().mockResolvedValue(undefined),
}));

const mockSetAudioMode = setAudioModeAsync as jest.Mock;
const realOS = Platform.OS;

function forceOS(os: string) {
  (Platform as { OS: string }).OS = os;
}

afterEach(() => {
  forceOS(realOS);
  mockSetAudioMode.mockClear();
});

describe("audioMode", () => {
  it("setRecordingMode requests a recording-oriented session on native", async () => {
    forceOS("ios");
    await setRecordingMode();
    expect(mockSetAudioMode).toHaveBeenCalledTimes(1);
    expect(mockSetAudioMode).toHaveBeenCalledWith(
      expect.objectContaining({
        allowsRecording: true,
        playsInSilentMode: true,
      }),
    );
  });

  it("setPlaybackMode requests a playback-oriented session on native (allowsRecording off)", async () => {
    forceOS("android");
    await setPlaybackMode();
    expect(mockSetAudioMode).toHaveBeenCalledTimes(1);
    expect(mockSetAudioMode).toHaveBeenCalledWith(
      expect.objectContaining({
        allowsRecording: false,
        playsInSilentMode: true,
      }),
    );
  });

  it("both are no-ops on web (there is no configurable native audio session)", async () => {
    forceOS("web");
    await setRecordingMode();
    await setPlaybackMode();
    expect(mockSetAudioMode).not.toHaveBeenCalled();
  });
});
