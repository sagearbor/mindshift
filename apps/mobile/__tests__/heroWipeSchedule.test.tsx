import React from "react";
import renderer, { act } from "react-test-renderer";
import {
  createHeroWipeState,
  currentEffect,
  currentImageIndex,
  DEFAULT_HERO_WIPE_CONFIG,
  HERO_WIPE_EFFECTS,
  nextImageIndex,
  shuffleIndices,
  tickHeroWipeState,
  useHeroWipeSchedule,
  wipeProgress,
  type HeroWipeConfig,
  type HeroWipeSnapshot,
} from "../src/utils/heroWipeSchedule";

/**
 * Minimal hook harness via plain react-test-renderer (not
 * @testing-library/react-native's async `renderHook`, whose internal
 * settling relies on real timers and stalls if fake timers are already
 * active at mount time). This matches the pattern already used by
 * AudioRecordScreen/recorderStreamScreen's tests for continuously-ticking
 * timers: enable fake timers BEFORE mounting, drive with plain `act`.
 */
function renderHeroWipeSchedule(
  imageCount: number,
  config: HeroWipeConfig,
  paused: boolean,
): { current: () => HeroWipeSnapshot } {
  let latest!: HeroWipeSnapshot;
  function Harness() {
    latest = useHeroWipeSchedule(imageCount, config, paused);
    return null;
  }
  act(() => {
    renderer.create(<Harness />);
  });
  return { current: () => latest };
}

// Deterministic "rng" that always returns 0 — Fisher-Yates with rng()=>0
// always swaps element i with element 0, producing a fixed, testable order.
const rngZero = () => 0;

const CONFIG: HeroWipeConfig = {
  holdMs: 1000,
  transitionMs: 500,
  effects: HERO_WIPE_EFFECTS,
};

describe("shuffleIndices", () => {
  it("returns a permutation of [0, count)", () => {
    const order = shuffleIndices(6);
    expect(order.slice().sort((a, b) => a - b)).toEqual([0, 1, 2, 3, 4, 5]);
  });

  it("is deterministic for a given rng", () => {
    expect(shuffleIndices(6, rngZero)).toEqual(shuffleIndices(6, rngZero));
  });
});

describe("createHeroWipeState", () => {
  it("starts holding on position 0 with no elapsed time or transitions", () => {
    const state = createHeroWipeState(6, rngZero);
    expect(state.position).toBe(0);
    expect(state.phase).toBe("hold");
    expect(state.phaseElapsedMs).toBe(0);
    expect(state.transitionCount).toBe(0);
    expect(state.order).toHaveLength(6);
  });
});

describe("tickHeroWipeState", () => {
  it("accumulates elapsed time within the hold phase without switching", () => {
    const state = createHeroWipeState(6, rngZero);
    const next = tickHeroWipeState(state, 400, CONFIG);
    expect(next.phase).toBe("hold");
    expect(next.phaseElapsedMs).toBe(400);
    expect(next.transitionCount).toBe(0);
  });

  it("crosses the hold→wipe boundary and carries the remainder", () => {
    const state = createHeroWipeState(6, rngZero);
    const next = tickHeroWipeState(state, 1200, CONFIG); // 1000 hold + 200 into wipe
    expect(next.phase).toBe("wipe");
    expect(next.phaseElapsedMs).toBe(200);
    expect(next.position).toBe(0); // still showing the same image mid-wipe
  });

  it("completes a wipe: advances position, bumps transitionCount, returns to hold", () => {
    const state = createHeroWipeState(6, rngZero);
    const afterHold = tickHeroWipeState(state, 1000, CONFIG);
    const afterWipe = tickHeroWipeState(afterHold, 500, CONFIG);
    expect(afterWipe.phase).toBe("hold");
    expect(afterWipe.phaseElapsedMs).toBe(0);
    expect(afterWipe.position).toBe(1);
    expect(afterWipe.transitionCount).toBe(1);
  });

  it("handles a delta spanning multiple phase boundaries in one tick", () => {
    const state = createHeroWipeState(6, rngZero);
    // hold(1000) + wipe(500, completes transition #1) + hold(1000) +
    // wipe(200, incomplete) = 2700ms
    const next = tickHeroWipeState(state, 2700, CONFIG);
    expect(next.transitionCount).toBe(1);
    expect(next.phase).toBe("wipe");
    expect(next.phaseElapsedMs).toBe(200);
    expect(next.position).toBe(1);
  });

  it("completes two full transitions when the delta covers both", () => {
    const state = createHeroWipeState(6, rngZero);
    // hold(1000) + wipe(500) + hold(1000) + wipe(500) = 3000ms, both complete.
    const next = tickHeroWipeState(state, 3000, CONFIG);
    expect(next.transitionCount).toBe(2);
    expect(next.phase).toBe("hold");
    expect(next.phaseElapsedMs).toBe(0);
    expect(next.position).toBe(2);
  });

  it("reshuffles and avoids repeating the last image once the order is exhausted", () => {
    let state = createHeroWipeState(3, rngZero);
    const originalOrder = state.order;
    for (let i = 0; i < 3; i++) {
      state = tickHeroWipeState(state, CONFIG.holdMs, CONFIG);
      state = tickHeroWipeState(state, CONFIG.transitionMs, CONFIG);
    }
    // Exhausted the 3-image order and reshuffled (still via rngZero).
    expect(state.position).toBe(0);
    expect(state.order).toHaveLength(3);
    const lastShown = originalOrder[originalOrder.length - 1];
    expect(state.order[0]).not.toBe(lastShown);
  });

  it("is a pure function: same inputs produce the same output", () => {
    const state = createHeroWipeState(6, rngZero);
    const a = tickHeroWipeState(state, 1300, CONFIG, rngZero);
    const b = tickHeroWipeState(state, 1300, CONFIG, rngZero);
    expect(a).toEqual(b);
    // ...and the input was not mutated.
    expect(state.phaseElapsedMs).toBe(0);
  });

  it("stays put but still tracks elapsed time when there's only one image", () => {
    const state = createHeroWipeState(1, rngZero);
    const next = tickHeroWipeState(state, 5000, CONFIG);
    expect(next.phase).toBe("hold");
    expect(next.position).toBe(0);
    expect(next.phaseElapsedMs).toBe(5000);
  });
});

describe("wipeProgress", () => {
  it("is 0 while holding", () => {
    const state = createHeroWipeState(6, rngZero);
    expect(wipeProgress(state, CONFIG)).toBe(0);
  });

  it("is a 0..1 fraction of the transition while wiping", () => {
    const state = createHeroWipeState(6, rngZero);
    const holding = tickHeroWipeState(state, CONFIG.holdMs, CONFIG);
    const quarterWiped = tickHeroWipeState(holding, CONFIG.transitionMs / 4, CONFIG);
    expect(wipeProgress(quarterWiped, CONFIG)).toBeCloseTo(0.25);
  });

  it("clamps at 1 even for an out-of-range elapsed value", () => {
    const state = createHeroWipeState(6, rngZero);
    // wipeProgress is defensively clamped on its own — tickHeroWipeState
    // would never produce phaseElapsedMs > transitionMs, but the reader
    // shouldn't trust that blindly.
    const overWiped = { ...state, phase: "wipe" as const, phaseElapsedMs: CONFIG.transitionMs * 3 };
    expect(wipeProgress(overWiped, CONFIG)).toBe(1);
  });
});

describe("currentEffect", () => {
  it("rotates through the registry in order as transitions complete", () => {
    let state = createHeroWipeState(6, rngZero);
    const seen: string[] = [currentEffect(state, CONFIG)];
    for (let i = 0; i < HERO_WIPE_EFFECTS.length; i++) {
      state = tickHeroWipeState(state, CONFIG.holdMs, CONFIG);
      state = tickHeroWipeState(state, CONFIG.transitionMs, CONFIG);
      seen.push(currentEffect(state, CONFIG));
    }
    // Rotates through all three and wraps back to the first.
    expect(seen).toEqual([
      HERO_WIPE_EFFECTS[0],
      HERO_WIPE_EFFECTS[1],
      HERO_WIPE_EFFECTS[2],
      HERO_WIPE_EFFECTS[0],
    ]);
  });
});

describe("currentImageIndex / nextImageIndex", () => {
  it("current is order[position], next wraps around the order", () => {
    const state = createHeroWipeState(6, rngZero);
    expect(currentImageIndex(state)).toBe(state.order[0]);
    expect(nextImageIndex(state)).toBe(state.order[1]);

    const lastPositionState = { ...state, position: state.order.length - 1 };
    expect(nextImageIndex(lastPositionState)).toBe(state.order[0]); // wraps
  });
});

describe("useHeroWipeSchedule", () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it("stays in the hold phase before the hold duration elapses", () => {
    const hook = renderHeroWipeSchedule(6, DEFAULT_HERO_WIPE_CONFIG, false);
    act(() => {
      jest.advanceTimersByTime(1000);
    });
    expect(hook.current().phase).toBe("hold");
  });

  it("advances to wipe and eventually to the next image over time", () => {
    const hook = renderHeroWipeSchedule(6, DEFAULT_HERO_WIPE_CONFIG, false);
    const firstIndex = hook.current().currentIndex;

    act(() => {
      jest.advanceTimersByTime(DEFAULT_HERO_WIPE_CONFIG.holdMs + 100);
    });
    expect(hook.current().phase).toBe("wipe");

    act(() => {
      jest.advanceTimersByTime(DEFAULT_HERO_WIPE_CONFIG.transitionMs + 100);
    });
    expect(hook.current().phase).toBe("hold");
    expect(hook.current().currentIndex).not.toBe(firstIndex);
  });

  it("never advances while paused (reduced motion)", () => {
    const hook = renderHeroWipeSchedule(6, DEFAULT_HERO_WIPE_CONFIG, true);
    const snapshot = { ...hook.current() };
    act(() => {
      jest.advanceTimersByTime(60000);
    });
    expect(hook.current()).toEqual(snapshot);
  });

  it("does not run a timer for a single image", () => {
    const hook = renderHeroWipeSchedule(1, DEFAULT_HERO_WIPE_CONFIG, false);
    act(() => {
      jest.advanceTimersByTime(60000);
    });
    expect(hook.current().phase).toBe("hold");
    expect(hook.current().currentIndex).toBe(hook.current().nextIndex);
  });
});
