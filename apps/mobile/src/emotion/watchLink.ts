/**
 * Phone <-> watch protocol of the Emotion dev tester, and the transport seam.
 *
 * Transport (phase 1): the EXISTING server websocket the watch's companion
 * mode already holds open all day. Phone -> watch is `POST /emotion/event`
 * (server pushes an `emotion` frame down the watch's live socket); watch ->
 * phone is an `emotion_ctl` frame up that socket, queued by the server and
 * drained by the phone with `GET /emotion/ctl?after=<seq>`. Why not the Wear
 * Data Layer is argued in docs/plans/2026-11-07-emotion-dev-mode.md (a phone
 * native module would change the native fingerprint and force a store
 * build; this path ships over the air). The controller only sees
 * {@link WatchLink}, so a Data Layer implementation can replace
 * {@link ServerWatchLink} without touching any logic.
 */
import type { FlagValue } from "./devStore";

/** Phone -> watch. Mirrors server/watch/emotion_bridge.py + the watch's EmotionFrame. */
export interface WatchEmotionState {
  v: 1;
  id: string;
  t: number;
  /** TINY's top class (the one that nudges). */
  label: string;
  prob: number;
  /** True when this state came with a buzz. */
  nudge: boolean;
  paused: boolean;
  /** Optional one-line suggestion (only when the tester enabled it). */
  suggestion: string | null;
}

export type WatchControlAction = "pause" | "resume" | "flag" | "mark";

/** Watch -> phone. */
export interface WatchControl {
  seq: number;
  action: WatchControlAction;
  flag?: Exclude<FlagValue, "unrated">;
  nudgeId?: string | null;
  t: number;
}

export interface WatchLink {
  /** Deliver a state to the watch. Resolves false (never throws) when it could not be delivered. */
  sendState(state: WatchEmotionState): Promise<boolean>;
  /** Controls newer than `afterSeq`, oldest first. Never throws: [] on failure. */
  pollControls(afterSeq: number): Promise<WatchControl[]>;
}

export type Fetcher = (url: string, init?: { method?: string; headers?: Record<string, string>; body?: string }) => Promise<{
  ok: boolean;
  status: number;
  json(): Promise<unknown>;
}>;

export class ServerWatchLink implements WatchLink {
  constructor(
    private readonly baseUrl: string,
    private readonly headers: () => Promise<Record<string, string>>,
    private readonly fetcher: Fetcher,
  ) {}

  async sendState(state: WatchEmotionState): Promise<boolean> {
    try {
      const r = await this.fetcher(`${this.baseUrl}/emotion/event`, {
        method: "POST",
        headers: await this.headers(),
        body: JSON.stringify(state),
      });
      if (!r.ok) return false;
      const body = (await r.json()) as { delivered?: boolean };
      return body.delivered === true;
    } catch {
      return false;
    }
  }

  async pollControls(afterSeq: number): Promise<WatchControl[]> {
    try {
      const r = await this.fetcher(`${this.baseUrl}/emotion/ctl?after=${afterSeq}`, {
        method: "GET",
        headers: await this.headers(),
      });
      if (!r.ok) return [];
      const body = (await r.json()) as { items?: Array<Record<string, unknown>> };
      const out: WatchControl[] = [];
      for (const it of body.items ?? []) {
        const action = it.action;
        if ((action !== "pause" && action !== "resume" && action !== "flag") || typeof it.seq !== "number") continue;
        const flag = it.flag;
        out.push({
          seq: it.seq,
          action,
          flag: flag === "bad" || flag === "good" || flag === "ignore" ? flag : undefined,
          nudgeId: typeof it.nudge_id === "string" ? it.nudge_id : null,
          t: typeof it.t === "number" ? it.t : 0,
        });
      }
      return out.sort((a, b) => a.seq - b.seq);
    } catch {
      return [];
    }
  }
}

/** In-memory link for tests: records sent states, serves queued controls. */
export class FakeWatchLink implements WatchLink {
  sent: WatchEmotionState[] = [];
  queue: WatchControl[] = [];
  deliver = true;
  async sendState(s: WatchEmotionState): Promise<boolean> {
    this.sent.push(s);
    return this.deliver;
  }
  async pollControls(afterSeq: number): Promise<WatchControl[]> {
    return this.queue.filter((c) => c.seq > afterSeq);
  }
}

/** Drains the watch's controls into a handler, remembering the last seq. */
export class ControlPoller {
  private seq = 0;
  constructor(private readonly link: WatchLink, private readonly handle: (c: WatchControl) => void) {}

  async pollOnce(): Promise<number> {
    const items = await this.link.pollControls(this.seq);
    for (const c of items) {
      if (c.seq <= this.seq) continue;
      this.seq = c.seq;
      try {
        this.handle(c);
      } catch {
        // one bad control must not stall the queue
      }
    }
    return items.length;
  }
}
