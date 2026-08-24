/**
 * On-device coaching suggestions: one prompt, several model providers, a
 * fall-through chain.
 *
 * Providers (all behind `SuggestionProvider`, so the chain and its tests
 * never touch a native module):
 *
 * - `os`      — the platform's built-in model via expo-ai-kit (Gemini Nano
 *               on Pixel 9/10 through ML Kit GenAI; Apple Foundation Models
 *               on iOS 26 / A17 Pro+).
 * - `bundled` — a downloadable LiteRT-LM model via expo-ai-kit (the
 *               registry's recommended sub-GB model for this device; see the
 *               PR body for why not Gemma 3 1B — it is gated on HF).
 * - `cloud`   — do nothing locally: return null and let the server's
 *               `suggestion` event (suggestion_source "cloud") speak.
 *
 * Both on-device families have NON-configurable safety filters whose
 * behaviour on relationship-coaching text is unknown until measured on a
 * device (docs/research/on-device-stack-2026-08-24.md), so a refusal,
 * guardrail error, or unparseable reply falls through to the next provider
 * instead of surfacing as a broken suggestion.
 */

export interface TextTone {
  warmth: number | null;
  defensiveness: number | null;
  sarcasm: number | null;
  sadness: number | null;
  frustration: number | null;
  label: string | null;
}

export type LiveMode = "earpiece" | "speaker" | "therapist";

export interface SuggestInput {
  text: string;
  speaker: string;
  /** null = unknown; true = the coached user's own turn (nudge, not response). */
  isSelf: boolean | null;
  /** Empathy slider 0–100. */
  empathy: number;
  /** Prior turns, oldest first, already trimmed by the caller. */
  context: { speaker: string; text: string }[];
  /** One-line delivery cue from prosody, e.g. "loud, fast". Optional. */
  prosodyHint?: string;
  mode: LiveMode;
}

export interface SuggestOutput {
  suggestion: string;
  textTone: TextTone;
}

export interface SuggestionProvider {
  readonly name: string;
  isAvailable(): Promise<boolean>;
  /** A parsed suggestion; `null` means "nothing local — the cloud will
   *  answer" (only the cloud provider does that). Throws on model failure. */
  suggest(input: SuggestInput): Promise<SuggestOutput | null>;
}

// ---------------------------------------------------------------------------
// The one prompt template every provider shares.
// ---------------------------------------------------------------------------

function stance(empathy: number): string {
  if (empathy <= 20) return "assertive and direct";
  if (empathy <= 50) return "balanced";
  if (empathy <= 80) return "warm and empathetic";
  return "validating and gentle";
}

export const SUGGESTION_SYSTEM_PROMPT =
  "You are a discreet real-time conversation coach whispering to one person " +
  "during a conversation. Reply with ONLY a JSON object, no prose, no markdown: " +
  '{"suggestion": string, "tone": {"warmth": 0-100, "defensiveness": 0-100, ' +
  '"sarcasm": 0-100, "sadness": 0-100, "frustration": 0-100, "label": string}}. ' +
  '"tone" scores the turn you were given. Keep "suggestion" under 18 words.';

export function buildPrompt(input: SuggestInput): { system: string; user: string } {
  const history = input.context
    .map((t) => `${t.speaker}: ${t.text}`)
    .join("\n");
  const who = input.isSelf ? "the coached person (YOU)" : input.speaker;
  const task = input.isSelf
    ? "The coached person just said this. Give a single delivery nudge for them " +
      "(6 words or fewer, e.g. \"ease up\", \"let them finish\")."
    : "Suggest what the coached person should say next to " +
      `${input.speaker}, in a ${stance(input.empathy)} stance.`;
  const cue = input.prosodyHint ? `\nDelivery cue: ${input.prosodyHint}.` : "";
  const user =
    (history ? `Earlier:\n${history}\n\n` : "") +
    `Latest turn from ${who}: "${input.text}"${cue}\n\n${task}`;
  return { system: SUGGESTION_SYSTEM_PROMPT, user };
}

// ---------------------------------------------------------------------------
// Defensive parsing
// ---------------------------------------------------------------------------

function clampScore(v: unknown): number | null {
  const n = typeof v === "string" ? Number(v) : v;
  if (typeof n !== "number" || !Number.isFinite(n)) return null;
  return Math.max(0, Math.min(100, Math.round(n)));
}

export const EMPTY_TONE: TextTone = {
  warmth: null,
  defensiveness: null,
  sarcasm: null,
  sadness: null,
  frustration: null,
  label: null,
};

/** Parse a model reply into a suggestion; null when it isn't one. Tolerates
 *  markdown fences, leading prose, and missing/odd tone fields. */
export function parseSuggestionJson(raw: string): SuggestOutput | null {
  if (typeof raw !== "string") return null;
  const text = raw.replace(/```(?:json)?/gi, "").trim();
  const first = text.indexOf("{");
  const last = text.lastIndexOf("}");
  if (first < 0 || last <= first) return null;
  let obj: unknown;
  try {
    obj = JSON.parse(text.slice(first, last + 1));
  } catch {
    return null;
  }
  if (!obj || typeof obj !== "object") return null;
  const o = obj as Record<string, unknown>;
  const suggestion =
    typeof o.suggestion === "string"
      ? o.suggestion.trim()
      : Array.isArray(o.suggestions) && typeof o.suggestions[0] === "string"
        ? (o.suggestions[0] as string).trim()
        : "";
  if (!suggestion) return null;
  const tone = (o.tone && typeof o.tone === "object" ? o.tone : {}) as Record<string, unknown>;
  return {
    suggestion,
    textTone: {
      warmth: clampScore(tone.warmth),
      defensiveness: clampScore(tone.defensiveness),
      sarcasm: clampScore(tone.sarcasm),
      sadness: clampScore(tone.sadness),
      frustration: clampScore(tone.frustration),
      label: typeof tone.label === "string" && tone.label.trim() ? tone.label.trim() : null,
    },
  };
}

const REFUSAL_PATTERNS = [
  /\bI can(?:'|no)t (?:help|assist|provide|do that)\b/i,
  /\bI(?:'m| am) (?:not able|unable) to\b/i,
  /\bguardrail/i,
  /\bsafety (?:policy|guidelines|filter)/i,
  /\bcontent (?:policy|filter)/i,
];

/** A reply (or thrown error message) that reads as a safety refusal. */
export function isRefusal(text: string): boolean {
  return REFUSAL_PATTERNS.some((p) => p.test(text));
}

// ---------------------------------------------------------------------------
// The chain
// ---------------------------------------------------------------------------

export interface ChainAttempt {
  provider: string;
  outcome: "ok" | "unavailable" | "refused" | "unparseable" | "error" | "cloud";
  ms: number;
  detail?: string;
}

export interface ChainResult {
  output: SuggestOutput | null;
  /** Which provider answered (or "cloud" / "none"). */
  provider: string;
  attempts: ChainAttempt[];
}

export type ProviderName = "os" | "bundled" | "cloud";
export const DEFAULT_PROVIDER_ORDER: ProviderName[] = ["os", "bundled", "cloud"];

export class ProviderChain {
  private readonly ordered: SuggestionProvider[];

  constructor(
    providers: SuggestionProvider[],
    order: string[] = DEFAULT_PROVIDER_ORDER,
    private readonly now: () => number = () =>
      typeof performance !== "undefined" ? performance.now() : Date.now(),
  ) {
    const byName = new Map(providers.map((p) => [p.name, p]));
    this.ordered = order
      .map((n) => byName.get(n))
      .filter((p): p is SuggestionProvider => Boolean(p));
    // Anything not named in `order` still runs, after the named ones.
    for (const p of providers) if (!order.includes(p.name)) this.ordered.push(p);
  }

  get providerNames(): string[] {
    return this.ordered.map((p) => p.name);
  }

  async suggest(input: SuggestInput): Promise<ChainResult> {
    const attempts: ChainAttempt[] = [];
    for (const p of this.ordered) {
      const t0 = this.now();
      let available = false;
      try {
        available = await p.isAvailable();
      } catch {
        available = false;
      }
      if (!available) {
        attempts.push({ provider: p.name, outcome: "unavailable", ms: this.now() - t0 });
        continue;
      }
      try {
        const out = await p.suggest(input);
        const ms = this.now() - t0;
        if (out === null) {
          attempts.push({ provider: p.name, outcome: "cloud", ms });
          return { output: null, provider: p.name, attempts };
        }
        if (isRefusal(out.suggestion)) {
          attempts.push({ provider: p.name, outcome: "refused", ms, detail: out.suggestion });
          continue;
        }
        attempts.push({ provider: p.name, outcome: "ok", ms });
        return { output: out, provider: p.name, attempts };
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        attempts.push({
          provider: p.name,
          outcome: isRefusal(msg) ? "refused" : msg.startsWith("unparseable") ? "unparseable" : "error",
          ms: this.now() - t0,
          detail: msg,
        });
      }
    }
    return { output: null, provider: "none", attempts };
  }
}

// ---------------------------------------------------------------------------
// Providers
// ---------------------------------------------------------------------------

/** The slice of expo-ai-kit the providers use — injected so tests fake it. */
export interface ExpoAiKitLike {
  isAvailable(): Promise<boolean>;
  prepareBuiltInModel(): Promise<void>;
  sendMessage(
    messages: { role: "system" | "user" | "assistant"; content: string }[],
    options?: { systemPrompt?: string; signal?: AbortSignal },
  ): Promise<{ text: string }>;
  setModel(modelId: string, options?: { backend?: "auto" | "gpu" | "cpu" }): Promise<void>;
  getActiveModel(): string;
  getRecommendedModel(): Promise<{ id: string; status: string } | null>;
  getDownloadedModels(): Promise<{ id: string; status: string }[]>;
}

function messagesFor(input: SuggestInput) {
  const { system, user } = buildPrompt(input);
  return [
    { role: "system" as const, content: system },
    { role: "user" as const, content: user },
  ];
}

async function sendAndParse(ai: ExpoAiKitLike, input: SuggestInput): Promise<SuggestOutput> {
  const res = await ai.sendMessage(messagesFor(input));
  const parsed = parseSuggestionJson(res.text);
  if (!parsed) {
    if (isRefusal(res.text)) throw new Error(`refused: ${res.text.slice(0, 120)}`);
    throw new Error(`unparseable: ${res.text.slice(0, 120)}`);
  }
  return parsed;
}

/** Gemini Nano (Android) / Apple Foundation Models (iOS) — whichever the OS has. */
export function osModelProvider(ai: ExpoAiKitLike, builtInId: "mlkit" | "apple-fm"): SuggestionProvider {
  let prepared: Promise<boolean> | null = null;
  return {
    name: "os",
    isAvailable() {
      if (!prepared) {
        prepared = (async () => {
          if (!(await ai.isAvailable())) return false;
          // Android: downloads the AICore-managed model on first use; iOS: a
          // readiness check. Either failing means "not here" — not a crash.
          await ai.prepareBuiltInModel();
          return true;
        })().catch(() => false);
      }
      return prepared;
    },
    async suggest(input) {
      if (ai.getActiveModel() !== builtInId) await ai.setModel(builtInId);
      return sendAndParse(ai, input);
    },
  };
}

/** A downloaded LiteRT-LM model; unavailable until it's on disk. */
export function bundledModelProvider(ai: ExpoAiKitLike, modelId?: string): SuggestionProvider {
  let resolvedId: string | null = modelId ?? null;
  return {
    name: "bundled",
    async isAvailable() {
      try {
        const downloaded = await ai.getDownloadedModels();
        if (resolvedId) return downloaded.some((m) => m.id === resolvedId);
        const rec = await ai.getRecommendedModel();
        if (rec && downloaded.some((m) => m.id === rec.id)) {
          resolvedId = rec.id;
          return true;
        }
        return false;
      } catch {
        return false;
      }
    },
    async suggest(input) {
      if (!resolvedId) throw new Error("bundled model not resolved");
      if (ai.getActiveModel() !== resolvedId) await ai.setModel(resolvedId);
      return sendAndParse(ai, input);
    },
  };
}

/** Terminal provider: never fails, never answers — the server does. */
export function cloudProvider(): SuggestionProvider {
  return {
    name: "cloud",
    async isAvailable() {
      return true;
    },
    async suggest() {
      return null;
    },
  };
}
