/**
 * On-device store of the Emotion dev tester: an append-only JSONL log of
 * records (+ flag updates) and a folder of short WAV clips, both bounded and
 * both deletable in one call. NOTHING here uploads anything; export is a
 * separate, explicit user action (EmotionDevPanel "Export log").
 *
 * Built on the recorder's RecorderFs seam so it is unit-tested over MemoryFs.
 */
import type { RecorderFs } from "../recorder/types";
import { buildWavHeader, int16ToBytes } from "../recorder/wav";
import { EMOTION_DEV_DEFAULTS, type EmotionDevConfig, type Probs } from "./config";
import type { RecordContext } from "./context";

export type FlagValue = "bad" | "good" | "ignore" | "unrated";
export type RecordKind = "window" | "nudge" | "disagree" | "scripted" | "saved" | "mark" | "watch" | "neutral-sample";

/** One classified window inside an episode clip (raw probabilities of both models). */
export interface EpisodeWindow {
  t: number;
  tiny: number[];
  small: number[] | null;
  /** Why this window counted: "disagree" | "heard" | "nudge"; empty = a lead/tail window that did not trigger. */
  triggers: string[];
  nudged: boolean;
}

/** An episode clip: consecutive triggered windows merged into one recording. Additive field. */
export interface EpisodeInfo {
  startT: number;
  endT: number;
  /** Seconds of audio in the clip (as intended; the file is the truth). */
  clipS: number;
  windows: EpisodeWindow[];
  peakTiny: { label: string; prob: number };
  peakSmall: { label: string; prob: number } | null;
  /** A nudge fired while this episode was open (the emotion buzz, not Live Coach advice). */
  nudge: boolean;
  nudgeIds: string[];
  triggers: string[];
}

export type CapState = "ok" | "warn" | "full";

export interface EmotionRecord {
  id: string;
  /** Epoch ms. */
  t: number;
  kind: RecordKind;
  /** RAW model probabilities (before any offsets), 6 classes, 4 decimals. */
  tiny: number[];
  small: number[] | null;
  /** Probabilities after the (default-off) logit offsets, only when they differ. */
  tinyAdjusted?: number[];
  nudged: boolean;
  nudgeReason?: string;
  flag: FlagValue | null;
  clip?: string;
  /** Review mode: what the listener says it really was (optional). */
  truth?: string;
  /** Scripted test only: the emotion the owner was asked to act, and the round. */
  expect?: string;
  round?: number;
  /** Episode clips only (kind saved/disagree): the merged windows, peaks and nudge flag. */
  episode?: EpisodeInfo;
  /** What the Live Coach session knew at that moment (talk share, pace, mode, coaching line). */
  context?: RecordContext;
  /** Set when the clip could NOT be saved (storage full, write error): the reason. */
  clipSkipped?: string;
}

type LogLine =
  | ({ op: "rec" } & Omit<EmotionRecord, "flag"> & { flag: FlagValue | null })
  | { op: "flag"; id: string; flag: FlagValue; t: number }
  | { op: "clip"; id: string; clip: string }
  | { op: "truth"; id: string; truth: string }
  | { op: "skip"; id: string; reason: string; t: number; bytes: number };

const enc = new TextEncoder();
const round4 = (p: Probs | null): number[] | null => (p ? Array.from(p, (v) => Math.round(v * 1e4) / 1e4) : null);

export interface ReviewItem {
  id: string;
  clip: string;
  uri: string;
  t: number;
  kind: RecordKind;
  tiny: number[];
  small: number[] | null;
}

export class EmotionDevStore {
  private capBytes: number;
  private readonly logUri: string;
  private readonly windowLogUri: string;
  private readonly clipDir: string;

  constructor(
    private readonly fs: RecorderFs,
    readonly dir: string,
    private readonly cfg: EmotionDevConfig = EMOTION_DEV_DEFAULTS,
    private readonly now: () => number = Date.now,
  ) {
    this.logUri = `${dir}/emotion-log.jsonl`;
    // Every classified window (rolling, trimmed) lives apart from the log of
    // nudges/flags/disagreements, so trimming one can never eat the other.
    this.windowLogUri = `${dir}/emotion-windows.jsonl`;
    this.clipDir = `${dir}/clips`;
    this.capBytes = cfg.maxClipBytes;
  }

  // -- log ------------------------------------------------------------------

  private append(line: LogLine, uri: string = this.logUri): void {
    this.fs.ensureDir(this.dir);
    this.fs.appendBytes(uri, enc.encode(JSON.stringify(line) + "\n"));
    // Only the rolling per-window log is ever trimmed. The main log holds the
    // records, labels and clip links the owner has yet to review: losing the
    // oldest half of it would silently orphan unreviewed clips.
    if (uri === this.windowLogUri) this.trimLogIfNeeded(uri);
  }

  private trimLogIfNeeded(uri: string): void {
    const size = this.fs.sizeOf(uri);
    if (size === null || size <= this.cfg.maxLogBytes) return;
    // Drop the oldest half (whole lines) — bounded, never silent growth.
    const lines = this.fs.readText(uri).split("\n").filter(Boolean);
    const keep = lines.slice(Math.floor(lines.length / 2));
    this.fs.writeText(uri, keep.join("\n") + "\n");
  }

  /** One classified window, raw probabilities, in the rolling window log. */
  addWindow(w: { id: string; t: number; tiny: Probs; small: Probs | null }): void {
    this.append(
      { op: "rec", id: w.id, t: w.t, kind: "window", tiny: round4(w.tiny)!, small: round4(w.small), nudged: false, flag: null },
      this.windowLogUri,
    );
  }

  readWindows(): EmotionRecord[] {
    return this.parse(this.windowLogUri);
  }
  addRecord(r: {
    id: string;
    t: number;
    kind: RecordKind;
    tiny: Probs;
    small: Probs | null;
    tinyAdjusted?: Probs;
    nudged: boolean;
    nudgeReason?: string;
    expect?: string;
    round?: number;
    episode?: EpisodeInfo;
    context?: RecordContext | null;
  }): void {
    this.append({
      op: "rec",
      id: r.id,
      t: r.t,
      kind: r.kind,
      tiny: round4(r.tiny)!,
      small: round4(r.small),
      ...(r.tinyAdjusted ? { tinyAdjusted: round4(r.tinyAdjusted)! } : {}),
      nudged: r.nudged,
      ...(r.nudgeReason ? { nudgeReason: r.nudgeReason } : {}),
      ...(r.expect ? { expect: r.expect, round: r.round } : {}),
      ...(r.episode ? { episode: r.episode } : {}),
      ...(r.context ? { context: r.context } : {}),
      // A nudge is "unrated" until the owner says otherwise (✓ is optional).
      flag: r.nudged ? "unrated" : null,
    });
  }

  setFlag(id: string, flag: FlagValue): void {
    this.append({ op: "flag", id, flag, t: this.now() });
  }

  /** Review mode: "what was it really?" (one of the six emotion labels). */
  setTruth(id: string, truth: string): void {
    this.append({ op: "truth", id, truth });
  }

  /** Saved clips nobody has rated yet (no thumbs, no flag), oldest first.
   *  A clip whose file was evicted is skipped. */
  reviewQueue(): ReviewItem[] {
    const out: ReviewItem[] = [];
    for (const r of this.readRecords()) {
      if (!r.clip || (r.flag !== null && r.flag !== "unrated")) continue;
      const uri = this.clipUri(r.clip);
      if (!this.fs.exists(uri)) continue;
      out.push({ id: r.id, clip: r.clip, uri, t: r.t, kind: r.kind, tiny: r.tiny, small: r.small });
    }
    return out;
  }

  clipUri(name: string): string {
    return `${this.clipDir}/${name}`;
  }

  /** Raise/lower the clip storage cap. Only RATED clips are ever evicted. */
  setClipCapBytes(bytes: number): void {
    this.capBytes = Math.max(1, bytes);
    this.evictRated(0, this.capBytes);
  }

  /** Adopt the cap for reporting/saving WITHOUT evicting anything (a freshly opened store defaults to 150 MB). */
  useCapBytes(bytes: number): void {
    this.capBytes = Math.max(1, bytes);
  }

  get clipCapBytes(): number {
    return this.capBytes;
  }

  /** Records with later flag/clip lines folded in, oldest first. */
  readRecords(): EmotionRecord[] {
    return this.parse(this.logUri);
  }

  private parse(uri: string): EmotionRecord[] {
    if (!this.fs.exists(uri)) return [];
    const byId = new Map<string, EmotionRecord>();
    const order: string[] = [];
    for (const raw of this.fs.readText(uri).split("\n")) {
      if (!raw) continue;
      let line: LogLine;
      try {
        line = JSON.parse(raw) as LogLine;
      } catch {
        continue; // a torn last line after a crash
      }
      if (line.op === "rec") {
        const { op: _op, ...rec } = line;
        byId.set(rec.id, rec);
        order.push(rec.id);
      } else if (line.op === "flag") {
        const r = byId.get(line.id);
        if (r) r.flag = line.flag;
      } else if (line.op === "truth") {
        const r = byId.get(line.id);
        if (r) r.truth = line.truth;
      } else if (line.op === "clip") {
        const r = byId.get(line.id);
        if (r) r.clip = line.clip;
      } else if (line.op === "skip") {
        const r = byId.get(line.id);
        if (r) r.clipSkipped = line.reason;
      }
    }
    return order.map((id) => byId.get(id)!).filter(Boolean);
  }

  /** The explicit-export payload: the log as JSON Lines text (no audio). */
  exportLogText(): string {
    const main = this.fs.exists(this.logUri) ? this.fs.readText(this.logUri) : "";
    // Include every classified window too, so a scripted test (say each
    // emotion in turn) can be scored; before this only nudges, disagreements
    // and flags were exported and a quiet run came out nearly empty.
    const wins = this.fs.exists(this.windowLogUri) ? this.fs.readText(this.windowLogUri) : "";
    return main + wins;
  }

  // -- clips ----------------------------------------------------------------

  /** Save a clip as `<epoch>-<id>-<reason>.wav`.
   *  Storage rule: at the cap, RATED clips are evicted oldest-first to make room;
   *  UNRATED and marked clips are never deleted. If there is still no room the clip is
   *  NOT saved: returns null and appends a `skip` line (with the reason) so the loss is
   *  on record. A user-initiated clip (mark/flag) may exceed the cap by `capUserOverageFraction`. */
  saveClip(p: { id: string; reason: string; atMs: number; pcm: Int16Array; sampleRate?: number; userInitiated?: boolean }): string | null {
    const sr = p.sampleRate ?? 16000;
    const bytes = 44 + p.pcm.length * 2;
    const limit = Math.floor(this.capBytes * (1 + (p.userInitiated ? this.cfg.capUserOverageFraction : 0)));
    if (bytes > limit) return this.skip(p.id, "clip-larger-than-cap", bytes);
    if (!this.evictRated(bytes, limit)) return this.skip(p.id, "storage-full", bytes);
    this.fs.ensureDir(this.clipDir);
    const safe = (s: string) => s.replace(/[^A-Za-z0-9_-]/g, "_");
    const name = `${String(p.atMs).padStart(13, "0")}-${safe(p.id)}-${safe(p.reason)}.wav`;
    const uri = `${this.clipDir}/${name}`;
    const header = buildWavHeader(sr, p.pcm.length * 2);
    const data = int16ToBytes(p.pcm);
    const all = new Uint8Array(header.length + data.length);
    all.set(header, 0);
    all.set(data, header.length);
    this.fs.writeBytes(uri, all);
    this.append({ op: "clip", id: p.id, clip: name });
    return name;
  }

  /** Why the most recent save was refused (null = none since this store opened). */
  lastSkipReason: string | null = null;

  private skip(id: string, reason: string, bytes: number): null {
    this.lastSkipReason = reason;
    try {
      this.append({ op: "skip", id, reason, t: this.now(), bytes });
    } catch {
      // The skip note is best effort; the null return is what callers act on.
    }
    return null;
  }

  listClips(): { name: string; bytes: number }[] {
    if (!this.fs.exists(this.clipDir)) return [];
    return this.fs
      .listFileNames(this.clipDir)
      .filter((n) => n.endsWith(".wav"))
      .sort()
      .map((name) => ({ name, bytes: this.fs.sizeOf(`${this.clipDir}/${name}`) ?? 0 }));
  }

  usage(): {
    clips: number;
    clipBytes: number;
    logBytes: number;
    capBytes: number;
    state: CapState;
    /** Clips with no rating yet (they are never auto-deleted). */
    unrated: number;
    /** Saves refused so far because storage was full (counted from the log). */
    skipped: number;
  } {
    const clips = this.listClips();
    const clipBytes = clips.reduce((a, c) => a + c.bytes, 0);
    const { unrated, skipped } = this.scanLog(clips.length);
    return {
      clips: clips.length,
      clipBytes,
      logBytes: (this.fs.sizeOf(this.logUri) ?? 0) + (this.fs.sizeOf(this.windowLogUri) ?? 0),
      capBytes: this.capBytes,
      state: this.capStateOf(clipBytes),
      unrated,
      skipped,
    };
  }

  /** ok / warn (>= 80% of the cap) / full (>= 98%, new clips will not fit). */
  capState(): CapState {
    return this.capStateOf(this.listClips().reduce((a, c) => a + c.bytes, 0));
  }

  private capStateOf(clipBytes: number): CapState {
    const f = clipBytes / this.capBytes;
    return f >= 0.98 ? "full" : f >= this.cfg.capWarnFraction ? "warn" : "ok";
  }

  private scanCache: { key: string; unrated: number; skipped: number } | null = null;
  private scanLog(clipCount: number): { unrated: number; skipped: number } {
    const key = `${this.fs.sizeOf(this.logUri) ?? 0}:${clipCount}`;
    if (this.scanCache?.key === key) return this.scanCache;
    let unrated = 0;
    for (const r of this.readRecords()) {
      if (r.clip && (r.flag === null || r.flag === "unrated") && this.fs.exists(this.clipUri(r.clip))) unrated++;
    }
    const text = this.fs.exists(this.logUri) ? this.fs.readText(this.logUri) : "";
    const skipped = text.split('"op":"skip"').length - 1;
    this.scanCache = { key, unrated, skipped };
    return this.scanCache;
  }

  /** Make `needed` bytes of room under `limit` by deleting the oldest RATED clips
   *  (their labels and log rows stay). Never touches unrated or marked clips.
   *  Returns whether `needed` now fits. */
  private evictRated(needed: number, limit: number): boolean {
    const clips = this.listClips(); // oldest first (epoch-prefixed names)
    let total = clips.reduce((a, c) => a + c.bytes, 0);
    if (total + needed <= limit) return true;
    const evictable = new Set<string>();
    for (const r of this.readRecords()) {
      if (r.clip && r.kind !== "mark" && r.flag !== null && r.flag !== "unrated") evictable.add(r.clip);
    }
    for (const c of clips) {
      if (total + needed <= limit) break;
      if (!evictable.has(c.name)) continue;
      try {
        this.fs.deleteRecursive(`${this.clipDir}/${c.name}`);
        total -= c.bytes;
      } catch {
        break;
      }
    }
    return total + needed <= limit;
  }

  /** The "delete all" control: every clip and the whole log. */
  deleteAll(): void {
    for (const uri of [this.clipDir, this.logUri, this.windowLogUri]) {
      if (this.fs.exists(uri)) this.fs.deleteRecursive(uri);
    }
  }
}
