/**
 * Session start reuses what the pre-flight probe already built (Silero VAD,
 * ECAPA + voiceprints) instead of rebuilding it — part of the ≈8.5 s cold
 * start measured on the Pixel 10 (2026-10-07) — and times each step.
 */
import {
  createDefaultFastLoop,
  probeFastLoopCapabilities,
  WarmBuildCache,
  WARM_MAX_AGE_MS,
  type SpeakerIdBuild,
} from "../src/live/defaultDeps";
import { EnergyVad } from "../src/live/vad";
import type { FastLoopHandlers } from "../src/live/defaultDeps";

const handlers = { send: () => {}, speak: () => {}, onTurn: () => {} } as unknown as FastLoopHandlers;

function seams(active = true) {
  const calls = { vad: 0, speaker: 0 };
  const speaker: SpeakerIdBuild = {
    embedder: null,
    labeler: null,
    capability: { active, reason: active ? "ok" : "offline", enrolled: active ? 3 : 0, model: null, droppedForModel: 0 },
  };
  let clock = 1_000;
  return {
    calls,
    advance: (ms: number) => (clock += ms),
    options: {
      warm: new WarmBuildCache(),
      now: () => clock,
      builders: {
        buildVad: async () => {
          calls.vad += 1;
          return { vad: new EnergyVad(), name: "energy VAD" };
        },
        buildSpeakerId: async () => {
          calls.speaker += 1;
          return speaker;
        },
      },
    },
  };
}

let logSpy: jest.SpyInstance;
beforeEach(() => {
  logSpy = jest.spyOn(console, "log").mockImplementation(() => {});
});
afterEach(() => logSpy.mockRestore());

describe("pre-flight → start: reuse, don't rebuild", () => {
  it("the start takes the probe's build (once) and says so in its timings", async () => {
    const s = seams();
    await probeFastLoopCapabilities(s.options);
    expect(s.calls).toEqual({ vad: 1, speaker: 1 });
    const first = await createDefaultFastLoop(handlers, s.options);
    expect(s.calls).toEqual({ vad: 1, speaker: 1 }); // nothing rebuilt
    expect(first.timings).toMatchObject({ reusedWarm: true, vadMs: 0, speakerIdMs: 0 });
    expect(first.status).toMatch(/speaker-ID/);
    // Single use: the next start builds its own.
    const second = await createDefaultFastLoop(handlers, s.options);
    expect(s.calls).toEqual({ vad: 2, speaker: 2 });
    expect(second.timings?.reusedWarm).toBe(false);
  });

  it("a stale build (voiceprints may have changed) or an inactive speaker-ID is rebuilt", async () => {
    const stale = seams();
    await probeFastLoopCapabilities(stale.options);
    stale.advance(WARM_MAX_AGE_MS + 1);
    expect((await createDefaultFastLoop(handlers, stale.options)).timings?.reusedWarm).toBe(false);
    expect(stale.calls.speaker).toBe(2);

    const offline = seams(false);
    await probeFastLoopCapabilities(offline.options);
    expect((await createDefaultFastLoop(handlers, offline.options)).timings?.reusedWarm).toBe(false);
    expect(offline.calls.speaker).toBe(2);
  });
});
