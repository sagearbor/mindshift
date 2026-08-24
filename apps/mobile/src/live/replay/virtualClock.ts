/**
 * A discrete-event virtual clock for replaying a recording through the fast
 * loop without real-time sleeps.
 *
 * The harness owns time: audio frames are "delivered" at their end time, and
 * every stage that would wait on a phone (STT finalization latency, the LLM
 * round trip, the loop's own STT poll / hold timers) waits on `sleep()` —
 * a timer on THIS clock. `advanceTo()` moves time forward firing timers in
 * due order, yielding between them so the awaiting code runs at the right
 * virtual instant. Real asynchronous work (ONNX inference) takes wall time
 * but zero virtual time; the driver calls `quiesce` before every advance so
 * a result that is still in flight on a native thread can never be
 * "overtaken" by the clock — which is what makes replays deterministic.
 */

interface Timer {
  due: number;
  seq: number;
  resolve: () => void;
}

/** One macrotask turn: lets every settled promise's continuation run. */
export function flushMacrotask(): Promise<void> {
  return new Promise((r) => setTimeout(r, 0));
}

export class VirtualClock {
  private t = 0;
  private seq = 0;
  private timers: Timer[] = [];

  /** Virtual milliseconds since the replay started. */
  now(): number {
    return this.t;
  }

  get pendingTimers(): number {
    return this.timers.length;
  }

  /** Resolve when the clock reaches now + ms (never earlier, never by itself). */
  sleep(ms: number): Promise<void> {
    return new Promise<void>((resolve) => {
      this.timers.push({ due: this.t + Math.max(0, ms), seq: this.seq++, resolve });
    });
  }

  /**
   * Move to `target`, firing due timers in (due, registration) order. After
   * each timer the awaiting code gets a macrotask to run and `quiesce` (if
   * given) waits for any real async work it kicked off, so a timer that
   * schedules another timer before `target` still fires at the right time.
   */
  async advanceTo(target: number, quiesce?: () => Promise<void>): Promise<void> {
    if (target < this.t) throw new Error(`VirtualClock: cannot go back (${this.t} -> ${target})`);
    for (;;) {
      this.timers.sort((a, b) => a.due - b.due || a.seq - b.seq);
      const next = this.timers[0];
      if (!next || next.due > target) break;
      this.timers.shift();
      this.t = Math.max(this.t, next.due);
      next.resolve();
      await flushMacrotask();
      if (quiesce) await quiesce();
    }
    this.t = target;
  }

  async advanceBy(ms: number, quiesce?: () => Promise<void>): Promise<void> {
    await this.advanceTo(this.t + ms, quiesce);
  }
}

/**
 * Tracks real (wall-time) asynchronous work so the driver can wait for it
 * before moving virtual time. Wrap only the native call itself — never a
 * virtual `sleep`, or the driver and the sleeper would wait on each other.
 */
export class InflightTracker {
  private readonly inflight = new Set<Promise<unknown>>();
  /** Wall-clock ms spent inside tracked work, by label. */
  readonly wallMs: Record<string, number> = {};
  readonly calls: Record<string, number> = {};

  track<T>(label: string, p: Promise<T>): Promise<T> {
    const t0 = performance.now();
    const wrapped = p.finally(() => {
      this.inflight.delete(wrapped);
      this.wallMs[label] = (this.wallMs[label] ?? 0) + (performance.now() - t0);
      this.calls[label] = (this.calls[label] ?? 0) + 1;
    });
    this.inflight.add(wrapped);
    return wrapped;
  }

  /** Wait until no tracked work is in flight and nothing new was started
   *  by its continuations. */
  async quiesce(): Promise<void> {
    for (;;) {
      if (this.inflight.size === 0) {
        await flushMacrotask();
        if (this.inflight.size === 0) return;
      }
      await Promise.allSettled([...this.inflight]);
      await flushMacrotask();
    }
  }
}
