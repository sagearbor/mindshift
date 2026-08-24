/**
 * src/live/capability.ts on the web build: capable only when the browser
 * has BOTH the Web Speech API and getUserMedia + AudioWorklet.
 */
import { Platform } from "react-native";
import { detectLiveCapability } from "../src/live/capability";

const originalOS = Platform.OS;
const g = globalThis as Record<string, unknown>;
const saved: Record<string, unknown> = {};
const KEYS = ["webkitSpeechRecognition", "SpeechRecognition", "navigator", "AudioWorkletNode", "AudioContext"];

beforeEach(() => {
  Object.defineProperty(Platform, "OS", { value: "web", configurable: true });
  for (const k of KEYS) {
    saved[k] = g[k];
    delete g[k];
  }
});

afterEach(() => {
  Object.defineProperty(Platform, "OS", { value: originalOS, configurable: true });
  for (const k of KEYS) {
    if (saved[k] === undefined) delete g[k];
    else g[k] = saved[k];
  }
});

function installMic() {
  g.navigator = { mediaDevices: { getUserMedia: () => Promise.resolve() } };
  g.AudioWorkletNode = class {};
  g.AudioContext = class {};
}

describe("detectLiveCapability (web)", () => {
  it("is not capable in a browser without a microphone API", () => {
    g.webkitSpeechRecognition = class {};
    const cap = detectLiveCapability();
    expect(cap.capable).toBe(false);
    expect(cap.reason).toMatch(/microphone/);
  });

  it("is not capable in a browser without speech recognition (e.g. Firefox)", () => {
    installMic();
    const cap = detectLiveCapability();
    expect(cap.capable).toBe(false);
    expect(cap.reason).toMatch(/no speech recognition/);
  });

  it("is capable in iOS Safari / Chrome (webkitSpeechRecognition + Web Audio)", () => {
    installMic();
    g.webkitSpeechRecognition = class {};
    const cap = detectLiveCapability();
    expect(cap.capable).toBe(true);
    expect(cap.reason).toMatch(/browser speech recognition/);
  });
});
