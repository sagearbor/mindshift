/**
 * Process-wide switchboard between the live mic path, the Settings panel and
 * the tester runtime. The live-session hook calls {@link EmotionDevHub.pushPcm}
 * with every frame it already has (a no-op while the mode is off); the panel
 * subscribes to {@link EmotionDevHub.status}. The runtime (engines + store +
 * link + control poller) is built lazily on the first frame after the mode is
 * enabled and torn down when it is disabled — so an off mode costs one
 * boolean check per audio frame and zero memory.
 */
import type { DevSnapshot, DevFlag, EmotionDevController } from "./devController";
import type { EmotionDevStore, ReviewItem } from "./devStore";
import type { RecordContext } from "./context";
import { DEFAULT_USER_SETTINGS, type EmotionDevUserSettings, type EmotionLabel } from "./config";

export interface EmotionRuntime {
  controller: EmotionDevController;
  store: EmotionDevStore;
  /** Which models actually loaded (the panel shows this honestly). */
  models: { tiny: boolean; small: boolean; smallReason: string | null };
  /** Stop pollers, release ONNX sessions. */
  dispose(): Promise<void>;
}

export type RuntimeBuild =
  | { ok: true; runtime: EmotionRuntime }
  | { ok: false; code: "models-missing" | "error"; message: string };

export type DevStatus =
  | { state: "off" }
  | { state: "starting" }
  | { state: "waiting-for-audio" }
  | { state: "models-missing"; message: string }
  | { state: "error"; message: string }
  | { state: "running"; models: EmotionRuntime["models"]; snapshot: DevSnapshot };

export interface RecorderLine {
  text: string;
  /** ok = recording; warn = recording but needs attention; error = NOT recording. */
  level: "ok" | "warn" | "error";
}

/**
 * The one-line recorder status for the Live Coach default view. A failed start
 * is never silent: every reason the tester is not keeping clips says so.
 * Returns null when the tester is off (nothing to report).
 */
export function describeRecorder(status: DevStatus, settings: EmotionDevUserSettings): RecorderLine | null {
  if (status.state === "off" || !settings.enabled) return null;
  if (status.state === "models-missing") return { level: "error", text: `Not recording: ${status.message}` };
  if (status.state === "error") return { level: "error", text: `Not recording: ${status.message}` };
  if (!settings.saveAudio) return { level: "error", text: "Not recording: Save audio is off (the tester is on, no clips are kept)" };
  if (status.state === "starting") return { level: "warn", text: "Starting the emotion recorder…" };
  if (status.state === "waiting-for-audio") return { level: "warn", text: "Emotion recorder ready, waiting for microphone audio" };
  const sn = status.snapshot;
  if (sn.paused) return { level: "warn", text: `Emotion recorder paused (${sn.clipsSaved} saved)` };
  if (sn.storage === "full" || (sn.skippedSaves > 0 && sn.lastSkipReason === "storage-full")) {
    return { level: "error", text: `Not recording: storage full, ${sn.skippedSaves} clip${sn.skippedSaves === 1 ? "" : "s"} skipped. Review and rate clips to free space.` };
  }
  if (sn.skippedSaves > 0 && sn.lastSkipReason === "write-error") {
    return { level: "error", text: `Recording problem: ${sn.skippedSaves} clip(s) failed to save${sn.lastError ? ` (${sn.lastError})` : ""}` };
  }
  if (sn.storage === "warn") return { level: "warn", text: `Recording emotion clips: ${sn.clipsSaved} saved. Storage is over 80% full.` };
  return { level: "ok", text: `Recording emotion clips: ${sn.clipsSaved} saved` };
}

export class EmotionDevHub {
  private settings: EmotionDevUserSettings = DEFAULT_USER_SETTINGS;
  private runtime: EmotionRuntime | null = null;
  private building = false;
  private failed: DevStatus | null = null;
  private listeners = new Set<(s: DevStatus) => void>();
  private contextProvider: (() => RecordContext | null) | null = null;

  constructor(
    private readonly build: (settings: EmotionDevUserSettings) => Promise<RuntimeBuild>,
    /** Opens the on-device store without loading any model, so "delete all"
     *  and usage work while the mode is off or models are missing. */
    private readonly openStore: () => EmotionDevStore | null = () => null,
  ) {}

  private store(): EmotionDevStore | null {
    const st = this.runtime?.store ?? this.openStore();
    // A store opened without the runtime knows only the default cap; report against the saved one.
    st?.useCapBytes?.(this.settings.storageCapMb * 1024 * 1024);
    return st;
  }

  subscribe(fn: (s: DevStatus) => void): () => void {
    this.listeners.add(fn);
    fn(this.status());
    return () => this.listeners.delete(fn);
  }

  private emit(): void {
    const s = this.status();
    this.listeners.forEach((l) => l(s));
  }

  status(): DevStatus {
    if (!this.settings.enabled) return { state: "off" };
    if (this.failed) return this.failed;
    if (this.building) return { state: "starting" };
    if (!this.runtime) return { state: "waiting-for-audio" };
    return { state: "running", models: this.runtime.models, snapshot: this.runtime.controller.snapshot() };
  }

  /** The tester settings currently applied (read-only view for the UI). */
  get userSettings(): EmotionDevUserSettings {
    return this.settings;
  }

  get enabled(): boolean {
    return this.settings.enabled;
  }

  async configure(next: EmotionDevUserSettings): Promise<void> {
    const wasEnabled = this.settings.enabled;
    this.settings = next;
    this.runtime?.controller.applySettings(next);
    if (wasEnabled && !next.enabled) await this.teardown();
    if (!wasEnabled && next.enabled) this.failed = null;
    this.emit();
  }

  /** Called with every live-session mic frame (16 kHz mono int16). */
  pushPcm(pcm: Int16Array): void {
    if (!this.settings.enabled || this.failed) return;
    const rt = this.runtime;
    if (rt) {
      rt.controller.pushPcm(pcm);
      return;
    }
    if (!this.building) void this.start();
  }

  /** Live Coach hands the tester what it knows (talk share, pace, mode); null clears it. */
  setContextProvider(fn: (() => RecordContext | null) | null): void {
    this.contextProvider = fn;
  }

  /** A Live Coach session is starting: build the runtime NOW (retrying a past failure) so a
   *  missing model shows on the status line immediately instead of at the first audio frame. */
  sessionStarting(): void {
    if (!this.settings.enabled) return;
    void this.ensureRuntime().then(() => this.emit());
  }

  /** Retry after fixing a "models missing" state (e.g. the owner pushed the files). */
  retry(): void {
    this.failed = null;
    this.emit();
  }

  private async start(): Promise<void> {
    this.building = true;
    this.emit();
    try {
      const r = await this.build(this.settings);
      if (!this.settings.enabled) {
        if (r.ok) await r.runtime.dispose();
        return;
      }
      if (r.ok) {
        this.runtime = r.runtime;
        r.runtime.controller.applySettings(this.settings);
        r.runtime.controller.setContextProvider(() => this.contextProvider?.() ?? null);
        r.runtime.controller.setSnapshotListener(() => this.emit());
      } else {
        this.failed = r.code === "models-missing" ? { state: "models-missing", message: r.message } : { state: "error", message: r.message };
      }
    } catch (e) {
      this.failed = { state: "error", message: e instanceof Error ? e.message : String(e) };
    } finally {
      this.building = false;
      this.emit();
    }
  }

  /** Build the runtime without waiting for Live Coach audio (scripted test). */
  async ensureRuntime(): Promise<boolean> {
    if (!this.settings.enabled) return false;
    if (!this.runtime && !this.building) {
      this.failed = null;
      await this.start();
    }
    return this.runtime !== null;
  }

  async classifyClip(pcm: Int16Array, expect: string, round: number) {
    return (await this.ensureRuntime()) ? this.runtime!.controller.classifyClip(pcm, expect, round) : null;
  }

  /** A live session ended: finish pending flag clips with what exists. */
  sessionEnded(): void {
    this.runtime?.controller.stop();
    this.emit();
  }

  private async teardown(): Promise<void> {
    const rt = this.runtime;
    this.runtime = null;
    this.failed = null;
    if (rt) {
      rt.controller.stop();
      await rt.dispose();
    }
  }

  // -- controls the panel uses ------------------------------------------------

  setPaused(paused: boolean): void {
    this.runtime?.controller.setPaused(paused);
    this.emit();
  }
  flag(flag: DevFlag): string | null {
    const id = this.runtime?.controller.flag(flag) ?? null;
    this.emit();
    return id;
  }
  mark(): string | null {
    const id = this.runtime?.controller.mark() ?? null;
    this.emit();
    return id;
  }
  /** Unrated saved clips for Review mode (works with the tester off: it only reads the store). */
  reviewQueue(): ReviewItem[] {
    return this.store()?.reviewQueue() ?? [];
  }
  rate(id: string, rating: "good" | "bad", truth?: EmotionLabel | null): void {
    const st = this.store();
    if (!st) return;
    if (truth) st.setTruth(id, truth);
    st.setFlag(id, rating);
    this.emit();
  }
  usage() {
    return this.store()?.usage() ?? null;
  }
  deleteAll(): void {
    this.store()?.deleteAll();
    this.emit();
  }
  exportLogText(): string | null {
    return this.store()?.exportLogText() ?? null;
  }
}
