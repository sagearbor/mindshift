import {
  classifyInputs,
  classifyOutputRoute,
  createDefaultAudioRouteProbe,
  PRIVATE_INPUT_TYPES,
  PRIVATE_OUTPUT_TYPES,
  type NativeAudioRouteModule,
  type NativeRouteSnapshot,
} from "../src/live/audioRoute";

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

// ---------------------------------------------------------------------------
// Native OUTPUT-route module (modules/audio-route, runtime 1.19.0+)
// ---------------------------------------------------------------------------

/** A fake of the native module: tests flip `snap` and fire route events. */
function fakeNative(initial: NativeRouteSnapshot) {
  const listeners = new Set<(s: NativeRouteSnapshot) => void>();
  const fake: {
    snap: NativeRouteSnapshot;
    module: NativeAudioRouteModule;
    emit(s: NativeRouteSnapshot): void;
    listenerCount(): number;
  } = {
    snap: initial,
    module: {
      getRoute: (): NativeRouteSnapshot => fake.snap,
      addListener: (_name: "onRouteChange", fn: (s: NativeRouteSnapshot) => void) => {
        listeners.add(fn);
        return { remove: () => void listeners.delete(fn) };
      },
    } as NativeAudioRouteModule,
    emit(s: NativeRouteSnapshot) {
      fake.snap = s;
      for (const fn of [...listeners]) fn(s);
    },
    listenerCount: () => listeners.size,
  };
  return fake;
}

const androidRoute = (
  active: string[] | null,
  outputs: string[] = active ?? [],
  extra: Partial<NativeRouteSnapshot> = {},
): NativeRouteSnapshot => ({
  platform: "android",
  outputs: ["BUILTIN_EARPIECE", "BUILTIN_SPEAKER", ...outputs],
  active,
  communication: null,
  mode: "normal",
  ...extra,
});
const iosRoute = (outputs: string[]): NativeRouteSnapshot => ({
  platform: "ios",
  outputs,
  active: outputs,
  communication: null,
  mode: null,
});

describe("audioRoute — native OUTPUT route (A2DP / LE Audio aware)", () => {
  const PRIVATE_ANDROID = [
    "BLUETOOTH_A2DP", // headphones with no mic — invisible to the input probe
    "BLUETOOTH_SCO",
    "BLE_HEADSET", // LE Audio earbuds (Pixel Buds on a Pixel 10)
    "WIRED_HEADSET",
    "WIRED_HEADPHONES",
    "USB_HEADSET",
    "HEARING_AID",
  ];
  const PUBLIC_ANDROID = [
    "BUILTIN_SPEAKER",
    "BUILTIN_EARPIECE",
    "BLE_SPEAKER", // a Bluetooth LE speaker is a loudspeaker
    "BLE_BROADCAST", // Auracast: anyone tuned in hears it
    "USB_DEVICE",
    "USB_ACCESSORY",
    "HDMI",
    "LINE_ANALOG",
    "DOCK",
    "BUS",
    "UNKNOWN",
  ];
  const PRIVATE_IOS = ["Headphones", "BluetoothA2DPOutput", "BluetoothHFP", "BluetoothLE", "USBAudio"];
  const PUBLIC_IOS = ["Speaker", "Receiver", "AirPlay", "CarAudio", "HDMI", "LineOut"];

  it.each(PRIVATE_ANDROID)("Android %s as the media route is private", (type) => {
    expect(classifyOutputRoute(androidRoute([type]))).toBe("private");
    expect(PRIVATE_OUTPUT_TYPES).toContain(type);
  });

  it.each(PUBLIC_ANDROID)("Android %s as the media route is public", (type) => {
    expect(classifyOutputRoute(androidRoute([type]))).toBe("public");
    expect(PRIVATE_OUTPUT_TYPES).not.toContain(type);
  });

  it.each(PRIVATE_IOS)("iOS %s as the current route output is private", (type) => {
    expect(classifyOutputRoute(iosRoute([type]))).toBe("private");
  });

  it.each(PUBLIC_IOS)("iOS %s as the current route output is public", (type) => {
    expect(classifyOutputRoute(iosRoute([type]))).toBe("public");
    expect(PRIVATE_OUTPUT_TYPES).not.toContain(type);
  });

  it("the ACTIVE media route decides: a connected headset that media is not routed to is not private", () => {
    // LE Audio buds connected, but media is on the loudspeaker (Android 13+ answer).
    expect(classifyOutputRoute(androidRoute(["BUILTIN_SPEAKER"], ["BLE_HEADSET"]))).toBe("public");
    // Mixed active route (speaker + headset duplicated) is not private.
    expect(classifyOutputRoute(androidRoute(["BLE_HEADSET", "BUILTIN_SPEAKER"]))).toBe("public");
    expect(classifyOutputRoute(iosRoute(["Headphones", "AirPlay"]))).toBe("public");
  });

  it("Android < 13 (no active route): any connected private output counts; only the phone's own is public", () => {
    expect(classifyOutputRoute(androidRoute(null, ["BLUETOOTH_A2DP"]))).toBe("private");
    expect(classifyOutputRoute(androidRoute([], ["BLE_HEADSET"]))).toBe("private");
    expect(classifyOutputRoute(androidRoute(null, []))).toBe("public");
    expect(classifyOutputRoute(androidRoute(null, ["BLE_SPEAKER"]))).toBe("public");
  });

  it("in a call/communication mode, a non-private communication device wins over the media route", () => {
    const snap = androidRoute(["BLE_HEADSET"], ["BLE_HEADSET"], { mode: "communication", communication: "BUILTIN_SPEAKER" });
    expect(classifyOutputRoute(snap)).toBe("public");
    const ok = androidRoute(["BLE_HEADSET"], ["BLE_HEADSET"], { mode: "communication", communication: "BLE_HEADSET" });
    expect(classifyOutputRoute(ok)).toBe("private");
    // Outside a call the communication device is irrelevant.
    const normal = androidRoute(["BLE_HEADSET"], ["BLE_HEADSET"], { mode: "normal", communication: "BUILTIN_EARPIECE" });
    expect(classifyOutputRoute(normal)).toBe("private");
  });

  it("an unreadable snapshot is unknown, never private", () => {
    expect(classifyOutputRoute(null)).toBe("unknown");
    expect(classifyOutputRoute(undefined)).toBe("unknown");
    expect(classifyOutputRoute({} as NativeRouteSnapshot)).toBe("unknown");
    expect(classifyOutputRoute({ outputs: "x" } as unknown as NativeRouteSnapshot)).toBe("unknown");
  });

  it("the probe prefers the native module over the input list when it is present", () => {
    const native = fakeNative(androidRoute(["BLUETOOTH_A2DP"]));
    const lister = jest.fn(() => ({ getAvailableInputs: () => [{ type: "MicrophoneBuiltIn" }] }));
    const probe = createDefaultAudioRouteProbe(lister, () => native.module);
    expect(probe.check()).toBe("private"); // A2DP-only headphones: the input list says "public"
    expect(probe.source?.()).toBe("native");
    expect(lister).not.toHaveBeenCalled();
    native.snap = androidRoute(["BUILTIN_SPEAKER"]);
    expect(probe.check()).toBe("public");
  });

  it("a missing native module (web, or an older 1.18.0 binary on an OTA) falls back to the input list", () => {
    const probe = createDefaultAudioRouteProbe(() => ({ getAvailableInputs: () => [{ type: "BluetoothSCO" }] }), () => null);
    expect(probe.check()).toBe("private");
    expect(probe.source?.()).toBe("inputs");
    const throwing = createDefaultAudioRouteProbe(
      () => ({ getAvailableInputs: () => [{ type: "MicrophoneBuiltIn" }] }),
      () => {
        throw new Error("Cannot find native module");
      },
    );
    expect(throwing.check()).toBe("public");
    expect(throwing.source?.()).toBe("inputs");
  });

  it("a native call that throws fails safe to unknown (never private)", () => {
    const native = fakeNative(androidRoute(["BLE_HEADSET"]));
    native.module.getRoute = () => {
      throw new Error("AudioManager died");
    };
    const probe = createDefaultAudioRouteProbe(undefined, () => native.module);
    expect(probe.check()).toBe("unknown");
  });

  it("route-change events reach subscribers immediately, and unsubscribe/dispose remove the native listener", () => {
    const native = fakeNative(androidRoute(["BLE_HEADSET"]));
    const probe = createDefaultAudioRouteProbe(undefined, () => native.module);
    expect(probe.check()).toBe("private");
    const seen: string[] = [];
    const unsubscribe = probe.subscribe!((state) => seen.push(state));
    expect(native.listenerCount()).toBe(1);
    native.emit(androidRoute(["BUILTIN_SPEAKER"], [])); // earbuds disconnected
    expect(seen).toEqual(["public"]);
    expect(probe.check()).toBe("public");
    unsubscribe();
    expect(native.listenerCount()).toBe(0);
    probe.subscribe!(() => {});
    probe.dispose();
    expect(native.listenerCount()).toBe(0);
  });

  it("without the native module there are no events: subscribe is a no-op and the poll stays the only signal", () => {
    const probe = createDefaultAudioRouteProbe(() => ({ getAvailableInputs: () => [] }), () => null);
    const fn = jest.fn();
    const unsubscribe = probe.subscribe!(fn);
    expect(typeof unsubscribe).toBe("function");
    unsubscribe();
    expect(fn).not.toHaveBeenCalled();
  });
});
