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

/**
 * The live session's shape. `speaker` is shown as "In person" (both people
 * in the room, one mic) — the wire value stays `speaker` so stored episodes
 * and per-account prefs from before the rename keep working. `call` is an
 * in-app WebRTC call: only the user's own voice is on this mic; the other
 * side's turns arrive from the server (src/live/call/).
 */
export type LiveMode = "earpiece" | "speaker" | "therapist" | "call";

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
  /** Optional per-provider `suggest()` budget (ms). On-device inference
   *  (Gemini Nano) is far slower than a cloud call, so `os` overrides the
   *  chain default here; unset providers use ChainTimeouts.suggestMs. */
  readonly suggestTimeoutMs?: number;
}

/** Verbose on-device-LLM tracing to the console (adb logcat), gated so it is
 *  fully off in normal builds. Flip on by baking EXPO_PUBLIC_DEBUG_LLM=1 into
 *  the OTA/build env, then read it with `adb logcat | grep "[llm]"`. */
export const LLM_DEBUG = process.env.EXPO_PUBLIC_DEBUG_LLM === "1";

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
  outcome: "ok" | "unavailable" | "refused" | "unparseable" | "error" | "cloud" | "timeout";
  ms: number;
  detail?: string;
}

/**
 * Per-provider deadlines. The chain runs on the fast loop's serial turn
 * queue: a provider that hangs (Gemini Nano's first-use AICore download
 * inside `prepareBuiltInModel`, a stalled inference) would otherwise block
 * every later turn's transcript line and turn_local, and the UI would sit
 * waiting for a local suggestion that never comes. Past the deadline the
 * attempt is logged as "timeout" and the next rung runs — the memoized
 * preparation keeps going in the background for the next turn.
 */
export interface ChainTimeouts {
  /** Budget for `isAvailable()` (includes model preparation). */
  availabilityMs: number;
  /** Budget for `suggest()`. Stale advice is worse than none. */
  suggestMs: number;
}

export const DEFAULT_CHAIN_TIMEOUTS: ChainTimeouts = {
  availabilityMs: 1500,
  suggestMs: 4000,
};

class ChainTimeoutError extends Error {
  constructor(readonly stage: "availability" | "suggest", ms: number) {
    super(`${stage} timed out after ${ms} ms`);
  }
}

function withTimeout<T>(
  work: Promise<T>,
  ms: number,
  stage: "availability" | "suggest",
): Promise<T> {
  if (!Number.isFinite(ms) || ms <= 0) return work;
  let timer: ReturnType<typeof setTimeout> | null = null;
  const deadline = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new ChainTimeoutError(stage, ms)), ms);
  });
  return Promise.race([work, deadline]).finally(() => {
    if (timer !== null) clearTimeout(timer);
  });
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
  private readonly timeouts: ChainTimeouts;

  constructor(
    providers: SuggestionProvider[],
    order: string[] = DEFAULT_PROVIDER_ORDER,
    private readonly now: () => number = () =>
      typeof performance !== "undefined" ? performance.now() : Date.now(),
    timeouts: Partial<ChainTimeouts> = {},
  ) {
    this.timeouts = { ...DEFAULT_CHAIN_TIMEOUTS, ...timeouts };
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

  /**
   * Kick each on-device provider's `isAvailable()` — which for the `os` rung
   * triggers Android's first-use AICore download of Gemini Nano — WITHOUT
   * blocking. Call it when Live Coach mounts so the model is downloading while
   * the user reads the pre-flight, instead of starting only on the first
   * suggestion mid-session (which made the first ~9 turns fall through to cloud
   * at ~9 s latency, then Gemini Nano fired once at the end — dx-6CY7-R9B4,
   * 2026-08-26). Each provider memoizes its preparation, so `suggest()` reuses
   * whatever this started. The `cloud` rung has a trivial `isAvailable()`, so
   * skip it. Fire-and-forget: failures are the chain's problem at suggest time,
   * not here.
   */
  prewarm(): void {
    for (const p of this.ordered) {
      if (p.name === "cloud") continue;
      void Promise.resolve()
        .then(() => p.isAvailable())
        .catch(() => {});
    }
  }

  async suggest(input: SuggestInput): Promise<ChainResult> {
    const attempts: ChainAttempt[] = [];
    for (const p of this.ordered) {
      const t0 = this.now();
      let available = false;
      try {
        available = await withTimeout(p.isAvailable(), this.timeouts.availabilityMs, "availability");
      } catch (err) {
        if (err instanceof ChainTimeoutError) {
          attempts.push({ provider: p.name, outcome: "timeout", ms: this.now() - t0, detail: err.message });
          continue;
        }
        available = false;
      }
      if (!available) {
        attempts.push({ provider: p.name, outcome: "unavailable", ms: this.now() - t0 });
        continue;
      }
      try {
        const budget = p.suggestTimeoutMs ?? this.timeouts.suggestMs;
        const out = await withTimeout(p.suggest(input), budget, "suggest");
        const ms = this.now() - t0;
        if (out === null) {
          attempts.push({ provider: p.name, outcome: "cloud", ms });
          if (LLM_DEBUG) console.log(`[llm] ${p.name}: cloud (${Math.round(ms)}ms)`);
          return { output: null, provider: p.name, attempts };
        }
        if (isRefusal(out.suggestion)) {
          attempts.push({ provider: p.name, outcome: "refused", ms, detail: out.suggestion });
          if (LLM_DEBUG) console.log(`[llm] ${p.name}: refused (${Math.round(ms)}ms) ${out.suggestion.slice(0, 120)}`);
          continue;
        }
        attempts.push({ provider: p.name, outcome: "ok", ms });
        if (LLM_DEBUG) console.log(`[llm] ${p.name}: ok (${Math.round(ms)}ms)`);
        return { output: out, provider: p.name, attempts };
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        const outcome =
          err instanceof ChainTimeoutError
            ? "timeout"
            : isRefusal(msg)
              ? "refused"
              : msg.startsWith("unparseable")
                ? "unparseable"
                : "error";
        attempts.push({ provider: p.name, outcome, ms: this.now() - t0, detail: msg });
        if (LLM_DEBUG) console.log(`[llm] ${p.name}: ${outcome} (${Math.round(this.now() - t0)}ms) ${msg.slice(0, 200)}`);
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
    // On-device inference (Gemini Nano / Apple FM) is much slower than a cloud
    // call. A real Pixel 10 hit the old 4 s budget every turn (os:timeout),
    // never answering; give it 8 s so a genuinely-working-but-slow model can
    // land before we fall through to cloud (dx-7XJB-GDR9, 2026-08-26).
    suggestTimeoutMs: 8000,
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
