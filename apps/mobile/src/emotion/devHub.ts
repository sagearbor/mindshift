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
import type { EmotionDevStore } from "./devStore";
import { DEFAULT_USER_SETTINGS, type EmotionDevUserSettings } from "./config";

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

export class EmotionDevHub {
  private settings: EmotionDevUserSettings = DEFAULT_USER_SETTINGS;
  private runtime: EmotionRuntime | null = null;
  private building = false;
  private failed: DevStatus | null = null;
  private listeners = new Set<(s: DevStatus) => void>();

  constructor(
    private readonly build: (settings: EmotionDevUserSettings) => Promise<RuntimeBuild>,
    /** Opens the on-device store without loading any model, so "delete all"
     *  and usage work while the mode is off or models are missing. */
    private readonly openStore: () => EmotionDevStore | null = () => null,
  ) {}

  private store(): EmotionDevStore | null {
    return this.runtime?.store ?? this.openStore();
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

  get enabled(): boolean {
    return this.settings.enabled;
  }

  async configure(next: EmotionDevUserSettings): Promise<void> {
    const wasEnabled = this.settings.enabled;
    this.settings = next;
    this.runtime?.controller.setMaxNudgesPerHour(next.maxNudgesPerHour);
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
        r.runtime.controller.setMaxNudgesPerHour(this.settings.maxNudgesPerHour);
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
