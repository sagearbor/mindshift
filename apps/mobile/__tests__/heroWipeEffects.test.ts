import { HERO_WIPE_EFFECT_REGISTRY } from "../src/utils/heroWipeEffects";
import { HERO_WIPE_EFFECTS } from "../src/utils/heroWipeSchedule";

/**
 * The particle internals (embers/ice/bubbles) are exempt from unit testing
 * per the task spec — but their *absence* must never crash the app: no
 * canvas 2D context (unsupported browser, or this jest/node test
 * environment), no `document`, no rAF. Every runner is expected to fail
 * soft: return a callable no-op cleanup instead of throwing.
 */
describe("HERO_WIPE_EFFECT_REGISTRY", () => {
  it("has a runner for every effect in the rotation", () => {
    for (const id of HERO_WIPE_EFFECTS) {
      expect(typeof HERO_WIPE_EFFECT_REGISTRY[id]).toBe("function");
    }
  });

  it("never throws when getContext returns null", () => {
    const fakeCanvas = {
      getContext: () => null,
    } as unknown as HTMLCanvasElement;

    for (const id of HERO_WIPE_EFFECTS) {
      const runner = HERO_WIPE_EFFECT_REGISTRY[id];
      let cleanup!: () => void;
      expect(() => {
        cleanup = runner({ canvas: fakeCanvas, width: 60, height: 100 });
      }).not.toThrow();
      expect(typeof cleanup).toBe("function");
      expect(() => cleanup()).not.toThrow();
    }
  });

  it("never throws when getContext itself throws", () => {
    const fakeCanvas = {
      getContext: () => {
        throw new Error("canvas unsupported in this environment");
      },
    } as unknown as HTMLCanvasElement;

    for (const id of HERO_WIPE_EFFECTS) {
      const runner = HERO_WIPE_EFFECT_REGISTRY[id];
      let cleanup!: () => void;
      expect(() => {
        cleanup = runner({ canvas: fakeCanvas, width: 60, height: 100 });
      }).not.toThrow();
      expect(() => cleanup()).not.toThrow();
    }
  });

  it("never throws when the canvas has no getContext at all", () => {
    const fakeCanvas = {} as unknown as HTMLCanvasElement;

    for (const id of HERO_WIPE_EFFECTS) {
      const runner = HERO_WIPE_EFFECT_REGISTRY[id];
      let cleanup!: () => void;
      expect(() => {
        cleanup = runner({ canvas: fakeCanvas, width: 60, height: 100 });
      }).not.toThrow();
      expect(() => cleanup()).not.toThrow();
    }
  });

  it("runs and cleans up without throwing given a real-ish 2D context", () => {
    const calls: string[] = [];
    const fakeCtx = {
      clearRect: () => calls.push("clearRect"),
      createRadialGradient: () => ({ addColorStop: () => {} }),
      beginPath: () => {},
      arc: () => {},
      fill: () => {},
      stroke: () => {},
      moveTo: () => {},
      lineTo: () => {},
      fillStyle: "",
      strokeStyle: "",
      lineWidth: 1,
      globalAlpha: 1,
    };
    const fakeCanvas = {
      getContext: () => fakeCtx,
    } as unknown as HTMLCanvasElement;

    const originalRaf = global.requestAnimationFrame;
    const originalCaf = global.cancelAnimationFrame;
    // Fires the callback exactly once per runner (so we exercise one real
    // draw pass) without recursing forever on the runner's next rAF request.
    let rafId = 0;
    let invocationsLeft = 0;
    global.requestAnimationFrame = ((cb: FrameRequestCallback) => {
      rafId += 1;
      if (invocationsLeft > 0) {
        invocationsLeft -= 1;
        cb(0);
      }
      return rafId;
    }) as typeof requestAnimationFrame;
    global.cancelAnimationFrame = (() => {}) as typeof cancelAnimationFrame;

    try {
      for (const id of Object.keys(
        HERO_WIPE_EFFECT_REGISTRY,
      ) as (keyof typeof HERO_WIPE_EFFECT_REGISTRY)[]) {
        invocationsLeft = 1;
        const runner = HERO_WIPE_EFFECT_REGISTRY[id];
        const cleanup = runner({ canvas: fakeCanvas, width: 60, height: 100 });
        expect(typeof cleanup).toBe("function");
        expect(() => cleanup()).not.toThrow();
      }
      expect(calls).toContain("clearRect");
    } finally {
      global.requestAnimationFrame = originalRaf;
      global.cancelAnimationFrame = originalCaf;
    }
  });
});
