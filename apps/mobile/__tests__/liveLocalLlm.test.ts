/**
 * src/live/localLlm.ts — prompt, defensive parsing, and the provider chain's
 * fall-through (unavailable / refused / unparseable / error → next; cloud
 * terminal). Fakes only; the real expo-ai-kit is behind `ExpoAiKitLike`.
 */
import {
  buildPrompt,
  bundledModelProvider,
  cloudProvider,
  isRefusal,
  osModelProvider,
  parseSuggestionJson,
  ProviderChain,
  type ExpoAiKitLike,
  type SuggestInput,
  type SuggestionProvider,
} from "../src/live/localLlm";

const input: SuggestInput = {
  text: "You never call me anymore.",
  speaker: "Mom",
  isSelf: false,
  empathy: 70,
  context: [{ speaker: "You", text: "Hi Mom." }],
  prosodyHint: "quiet",
  mode: "speaker",
};

const GOOD =
  '{"suggestion": "I hear you — I miss our calls too.", "tone": {"warmth": 20, "defensiveness": 10, "sarcasm": 5, "sadness": 70, "frustration": 40, "label": "hurt"}}';

function provider(name: string, impl: Partial<SuggestionProvider>): SuggestionProvider {
  return {
    name,
    isAvailable: async () => true,
    suggest: async () => parseSuggestionJson(GOOD),
    ...impl,
  };
}

describe("buildPrompt", () => {
  it("asks for a response to the other person in the empathy stance", () => {
    const { system, user } = buildPrompt(input);
    expect(system).toMatch(/JSON/);
    expect(user).toContain("Earlier:\nYou: Hi Mom.");
    expect(user).toContain('Latest turn from Mom: "You never call me anymore."');
    expect(user).toContain("warm and empathetic");
    expect(user).toContain("Delivery cue: quiet");
  });

  it("asks for a short nudge on the coached person's own turn", () => {
    const { user } = buildPrompt({ ...input, isSelf: true, speaker: "You", prosodyHint: undefined });
    expect(user).toContain("delivery nudge");
    expect(user).toContain("the coached person (YOU)");
    expect(user).not.toContain("Delivery cue");
  });
});

describe("parseSuggestionJson", () => {
  it("parses the canonical shape and clamps scores", () => {
    const out = parseSuggestionJson('{"suggestion":"ok","tone":{"warmth":150,"frustration":"33","label":" tense "}}');
    expect(out).toEqual({
      suggestion: "ok",
      textTone: { warmth: 100, defensiveness: null, sarcasm: null, sadness: null, frustration: 33, label: "tense" },
    });
  });

  it("tolerates fences, prose around the object, and a suggestions[] array", () => {
    expect(parseSuggestionJson("Sure!\n```json\n" + GOOD + "\n```")?.suggestion).toMatch(/miss our calls/);
    expect(parseSuggestionJson('{"suggestions": ["first", "second"]}')?.suggestion).toBe("first");
  });

  it("returns null for junk, empty suggestions, and non-objects", () => {
    expect(parseSuggestionJson("I cannot help with that.")).toBeNull();
    expect(parseSuggestionJson('{"suggestion": ""}')).toBeNull();
    expect(parseSuggestionJson("[1,2]")).toBeNull();
    expect(parseSuggestionJson("{not json")).toBeNull();
  });
});

describe("isRefusal", () => {
  it("recognises common refusal / guardrail phrasings only", () => {
    expect(isRefusal("I can't help with that request.")).toBe(true);
    expect(isRefusal("guardrailViolation: unsafe content")).toBe(true);
    expect(isRefusal("Try saying: I hear you.")).toBe(false);
  });
});

describe("ProviderChain", () => {
  it("returns the first provider that answers, in the configured order", async () => {
    const calls: string[] = [];
    const chain = new ProviderChain(
      [
        provider("bundled", { suggest: async () => (calls.push("bundled"), parseSuggestionJson(GOOD)) }),
        provider("os", { suggest: async () => (calls.push("os"), parseSuggestionJson(GOOD)) }),
        cloudProvider(),
      ],
      ["os", "bundled", "cloud"],
    );
    expect(chain.providerNames).toEqual(["os", "bundled", "cloud"]);
    const r = await chain.suggest(input);
    expect(r.provider).toBe("os");
    expect(calls).toEqual(["os"]);
    expect(r.attempts).toEqual([expect.objectContaining({ provider: "os", outcome: "ok" })]);
  });

  it("falls through unavailable → refused → unparseable → error to the cloud", async () => {
    const chain = new ProviderChain([
      provider("os", { isAvailable: async () => false }),
      provider("bundled", { suggest: async () => ({ suggestion: "I can't help with that.", textTone: parseSuggestionJson(GOOD)!.textTone }) }),
      provider("third", { suggest: async () => { throw new Error("unparseable: <html>"); } }),
      provider("fourth", { suggest: async () => { throw new Error("INFERENCE_OOM:x:out of memory"); } }),
      provider("fifth", { isAvailable: async () => { throw new Error("boom"); } }),
      cloudProvider(),
    ], ["os", "bundled", "third", "fourth", "fifth", "cloud"]);
    const r = await chain.suggest(input);
    expect(r.output).toBeNull();
    expect(r.provider).toBe("cloud");
    expect(r.attempts.map((a) => `${a.provider}:${a.outcome}`)).toEqual([
      "os:unavailable",
      "bundled:refused",
      "third:unparseable",
      "fourth:error",
      "fifth:unavailable",
      "cloud:cloud",
    ]);
  });

  it("providers not named in the order still run, after the named ones", async () => {
    const chain = new ProviderChain([provider("extra", {}), provider("os", { isAvailable: async () => false })], ["os"]);
    expect(chain.providerNames).toEqual(["os", "extra"]);
    expect((await chain.suggest(input)).provider).toBe("extra");
  });

  it("reports provider 'none' when nothing (not even cloud) is configured", async () => {
    const r = await new ProviderChain([provider("os", { isAvailable: async () => false })]).suggest(input);
    expect(r).toMatchObject({ output: null, provider: "none" });
  });

  it("a thrown guardrail error counts as a refusal", async () => {
    const r = await new ProviderChain([
      provider("os", { suggest: async () => { throw new Error("guardrailViolation"); } }),
      cloudProvider(),
    ]).suggest(input);
    expect(r.attempts[0].outcome).toBe("refused");
    expect(r.provider).toBe("cloud");
  });
});

type FakeAi = ExpoAiKitLike & { calls: string[]; active: string };

function fakeAi(overrides: Partial<ExpoAiKitLike> = {}): FakeAi {
  const ai: FakeAi = {
    calls: [] as string[],
    active: "",
    isAvailable: async () => true,
    prepareBuiltInModel: async () => {
      ai.calls.push("prepare");
    },
    sendMessage: async () => {
      ai.calls.push("send");
      return { text: GOOD };
    },
    setModel: async (id: string) => {
      ai.calls.push(`setModel:${id}`);
      ai.active = id;
    },
    getActiveModel: (): string => ai.active,
    getRecommendedModel: async () => ({ id: "qwen-0.6b", status: "downloaded" }),
    getDownloadedModels: async () => [{ id: "qwen-0.6b", status: "downloaded" }],
    ...overrides,
  };
  return ai;
}

describe("expo-ai-kit providers", () => {
  it("os provider prepares once, activates the built-in model, and parses", async () => {
    const ai = fakeAi();
    const p = osModelProvider(ai, "mlkit");
    expect(await p.isAvailable()).toBe(true);
    expect(await p.isAvailable()).toBe(true);
    const out = await p.suggest(input);
    expect(out?.suggestion).toMatch(/miss our calls/);
    await p.suggest(input);
    expect(ai.calls).toEqual(["prepare", "setModel:mlkit", "send", "send"]);
  });

  it("os provider is unavailable when the OS says no or prepare fails", async () => {
    expect(await osModelProvider(fakeAi({ isAvailable: async () => false }), "apple-fm").isAvailable()).toBe(false);
    expect(
      await osModelProvider(fakeAi({ prepareBuiltInModel: async () => { throw new Error("MODEL_NOT_DOWNLOADED"); } }), "mlkit").isAvailable(),
    ).toBe(false);
  });

  it("os provider surfaces a refusal / junk reply as a typed error for the chain", async () => {
    await expect(osModelProvider(fakeAi({ sendMessage: async () => ({ text: "I can't help with that." }) }), "mlkit").suggest(input)).rejects.toThrow(/^refused/);
    await expect(osModelProvider(fakeAi({ sendMessage: async () => ({ text: "???" }) }), "mlkit").suggest(input)).rejects.toThrow(/^unparseable/);
  });

  it("bundled provider is available only once the recommended model is on disk", async () => {
    const ai = fakeAi();
    const p = bundledModelProvider(ai);
    expect(await p.isAvailable()).toBe(true);
    await p.suggest(input);
    expect(ai.calls).toEqual(["setModel:qwen-0.6b", "send"]);
    const none = bundledModelProvider(fakeAi({ getDownloadedModels: async () => [] }));
    expect(await none.isAvailable()).toBe(false);
    const explicit = bundledModelProvider(fakeAi(), "gemma-e2b");
    expect(await explicit.isAvailable()).toBe(false);
  });
});
