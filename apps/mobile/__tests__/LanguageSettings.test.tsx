import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import LanguageSettings, { VOICES_SHOWN } from "../src/components/LanguageSettings";
import { setLanguagePrefs } from "../src/live/languagePrefs";

let mockVoices: { identifier: string; name: string; language: string; quality?: string }[] = [];
jest.mock("expo-speech", () => ({
  getAvailableVoicesAsync: () => Promise.resolve(mockVoices),
  speak: jest.fn(),
  stop: jest.fn(),
}));

const flush = () => new Promise((r) => setTimeout(r, 0));
const byId = (root: ReactTestInstance, id: string) => root.findAll((n) => n.props.testID === id);
const press = (root: ReactTestInstance, id: string) =>
  act(() => {
    byId(root, id).find((n) => typeof n.props.onPress === "function")!.props.onPress();
  });
const texts = (root: ReactTestInstance) =>
  root.findAll((n) => (n.type as unknown) === "Text").map((n) => [n.props.children].flat().join(""));

async function mount() {
  let c!: renderer.ReactTestRenderer;
  await act(async () => {
    c = renderer.create(<LanguageSettings userId="u" />);
    await flush();
  });
  return c;
}

beforeEach(() => {
  mockVoices = [];
  setLanguagePrefs({ coachLanguage: "auto", conversationLanguage: "auto", coachVoice: null });
});

describe("LanguageSettings accordion", () => {
  it("is collapsed by default; headers show current values", async () => {
    setLanguagePrefs({ coachLanguage: "es-ES", conversationLanguage: "auto", coachVoice: null });
    mockVoices = [{ identifier: "m", name: "Monica", language: "es-ES" }];
    const c = await mount();
    expect(texts(c.root)).toEqual(["Language", "▼"]);
    expect(byId(c.root, "coach-language")).toHaveLength(0);

    press(c.root, "language-group-header");
    const t = texts(c.root);
    expect(t).toContain("Coach language: Spanish");
    expect(t).toContain("Conversation language: Auto");
    expect(t).toContain("Coach voice: Default");
    // inner rows are still closed
    expect(byId(c.root, "coach-language")).toHaveLength(0);
    press(c.root, "coach-language-acc-header");
    expect(byId(c.root, "coach-language").length).toBeGreaterThan(0);
  });

  it("lists only matching-language voices, caps at 6 with Show more", async () => {
    mockVoices = [
      ...Array.from({ length: 8 }, (_, i) => ({ identifier: `e${i}`, name: "", language: "en-US" })),
      { identifier: "au", name: "Karen", language: "en-AU" },
      { identifier: "es", name: "Monica", language: "es-ES" },
    ];
    const c = await mount();
    press(c.root, "language-group-header");
    press(c.root, "coach-voice-row-header");
    const shown = () => texts(c.root).filter((s) => /voice \d/.test(s));
    expect(shown()).toHaveLength(VOICES_SHOWN);
    expect(texts(c.root).join("|")).not.toMatch(/Karen|Monica/);
    press(c.root, "coach-voice-more");
    expect(shown()).toHaveLength(8);
  });
});
