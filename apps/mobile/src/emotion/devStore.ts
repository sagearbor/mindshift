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

export type FlagValue = "bad" | "good" | "ignore" | "unrated";
export type RecordKind = "window" | "nudge" | "disagree" | "scripted" | "saved" | "mark" | "watch";

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
}

type LogLine =
  | ({ op: "rec" } & Omit<EmotionRecord, "flag"> & { flag: FlagValue | null })
  | { op: "flag"; id: string; flag: FlagValue; t: number }
  | { op: "clip"; id: string; clip: string }
  | { op: "truth"; id: string; truth: string };

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
    this.trimLogIfNeeded(uri);
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

  /** Raise/lower the clip storage cap (save-audio mode); evicts at once if now over. */
  setClipCapBytes(bytes: number): void {
    this.capBytes = Math.max(1, bytes);
    this.evictClips();
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

  /** Save a clip as `<epoch>-<id>-<reason>.wav`; evicts oldest clips past the byte cap.
   *  Returns the file name, or null when a single clip alone exceeds the cap. */
  saveClip(p: { id: string; reason: string; atMs: number; pcm: Int16Array; sampleRate?: number }): string | null {
    const sr = p.sampleRate ?? 16000;
    const bytes = 44 + p.pcm.length * 2;
    if (bytes > this.capBytes) return null;
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
    this.evictClips();
    this.append({ op: "clip", id: p.id, clip: name });
    return name;
  }

  listClips(): { name: string; bytes: number }[] {
    if (!this.fs.exists(this.clipDir)) return [];
    return this.fs
      .listFileNames(this.clipDir)
      .filter((n) => n.endsWith(".wav"))
      .sort()
      .map((name) => ({ name, bytes: this.fs.sizeOf(`${this.clipDir}/${name}`) ?? 0 }));
  }

  usage(): { clips: number; clipBytes: number; logBytes: number } {
    const clips = this.listClips();
    return {
      clips: clips.length,
      clipBytes: clips.reduce((a, c) => a + c.bytes, 0),
      logBytes: (this.fs.sizeOf(this.logUri) ?? 0) + (this.fs.sizeOf(this.windowLogUri) ?? 0),
    };
  }

  /** Over the cap: delete oldest UNRATED clips first, rated ones only if that is not enough. */
  private evictClips(): void {
    const clips = this.listClips(); // oldest first (epoch-prefixed names)
    let total = clips.reduce((a, c) => a + c.bytes, 0);
    if (total <= this.capBytes) return;
    const rated = new Set<string>();
    for (const r of this.readRecords()) if (r.clip && r.flag !== null && r.flag !== "unrated") rated.add(r.clip);
    const order = [...clips.filter((c) => !rated.has(c.name)), ...clips.filter((c) => rated.has(c.name))];
    for (const c of order) {
      if (total <= this.capBytes) break;
      try {
        this.fs.deleteRecursive(`${this.clipDir}/${c.name}`);
        total -= c.bytes;
      } catch {
        break;
      }
    }
  }

  /** The "delete all" control: every clip and the whole log. */
  deleteAll(): void {
    for (const uri of [this.clipDir, this.logUri, this.windowLogUri]) {
      if (this.fs.exists(uri)) this.fs.deleteRecursive(uri);
    }
  }
}
