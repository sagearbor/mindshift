import { classifyInputs, createDefaultAudioRouteProbe, PRIVATE_INPUT_TYPES } from "../src/live/audioRoute";

describe("audioRoute (earpiece privacy probe)", () => {
  it("a Bluetooth hands-free or wired headset input means a private route", () => {
    expect(classifyInputs([{ type: "MicrophoneBuiltIn" }, { type: "BluetoothSCO" }])).toBe("private");
    expect(classifyInputs([{ type: "MicrophoneWired" }])).toBe("private");
    expect(classifyInputs([{ type: "BluetoothHFP" }])).toBe("private"); // iOS
    expect(classifyInputs([{ type: "HeadsetMic" }])).toBe("private"); // iOS
  });

  it("only the phone's own mic — or a car kit, which is a loudspeaker — is public", () => {
    expect(classifyInputs([{ type: "MicrophoneBuiltIn" }])).toBe("public");
    expect(classifyInputs([])).toBe("public");
    expect(classifyInputs([{ type: "CarAudio" }, { type: "MicrophoneBuiltIn" }])).toBe("public");
    expect(PRIVATE_INPUT_TYPES).not.toContain("CarAudio");
    expect(PRIVATE_INPUT_TYPES).not.toContain("USBAudio");
  });

  it("an unreadable answer is unknown, never private", () => {
    expect(classifyInputs(null)).toBe("unknown");
    expect(classifyInputs(undefined)).toBe("unknown");
  });

  it("the probe follows the device list live and fails safe when the native call throws", () => {
    let inputs = [{ type: "BluetoothSCO" }];
    const release = jest.fn();
    const probe = createDefaultAudioRouteProbe(() => ({ getAvailableInputs: () => inputs, release }));
    expect(probe.check()).toBe("private");
    inputs = [{ type: "MicrophoneBuiltIn" }];
    expect(probe.check()).toBe("public");
    probe.dispose();
    expect(release).toHaveBeenCalled();

    const broken = createDefaultAudioRouteProbe(() => {
      throw new Error("no native module");
    });
    expect(broken.check()).toBe("unknown");
    expect(broken.check()).toBe("unknown");
  });
});
