import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import { Image } from "react-native";
import { usePrefersReducedMotion } from "../src/hooks/usePrefersReducedMotion";
import HeroWipeWeb from "../src/components/HeroWipe.web";

/**
 * Direct test of the web implementation (Task P3-4b, review round 1). The
 * default jest environment resolves the "native" haste platform for a bare
 * `../components/HeroWipe` import (see HeroWipe.native.tsx's doc comment),
 * so — deliberately — this file imports `HeroWipe.web` by its explicit
 * filename to reach the real implementation directly, bypassing platform
 * resolution the same way a native `.ios.tsx` test would.
 */
jest.mock("../src/hooks/usePrefersReducedMotion");

const mockReducedMotion = usePrefersReducedMotion as jest.Mock;

function queryAll(comp: renderer.ReactTestRenderer, id: string): ReactTestInstance[] {
  return comp.root.findAll((n) => n.props?.testID === id);
}

// Predicate-based findAll (not findAllByType(Image)) sidesteps a generic
// ElementType/ReactNode structural mismatch under this project's React 19
// types (the same class of friction documented in heroWipeSchedule.test.tsx
// — react-test-renderer's typings don't line up with React 19's ReactNode
// including `bigint`). The `as unknown` cast is needed even for the plain
// equality check, for the same reason.
function queryImages(comp: renderer.ReactTestRenderer): ReactTestInstance[] {
  return comp.root.findAll((n) => (n.type as unknown) === Image);
}

describe("HeroWipe.web", () => {
  afterEach(() => {
    mockReducedMotion.mockReset();
  });

  it("renders the hero container with pointerEvents none (never traps clicks)", () => {
    mockReducedMotion.mockReturnValue(false);
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<HeroWipeWeb />);
    });
    const hero = queryAll(comp, "hero-wipe")[0];
    expect(hero).toBeTruthy();
    expect(hero.props.pointerEvents).toBe("none");
    act(() => comp.unmount());
  });

  it("renders two stacked images (current + next) when motion is allowed", () => {
    mockReducedMotion.mockReturnValue(false);
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<HeroWipeWeb />);
    });
    expect(queryImages(comp).length).toBe(2);
    act(() => comp.unmount());
  });

  describe("prefers-reduced-motion", () => {
    it("renders a single static image, no effect strip, and starts no wipe timer", () => {
      mockReducedMotion.mockReturnValue(true);
      const setIntervalSpy = jest.spyOn(global, "setInterval");

      let comp!: renderer.ReactTestRenderer;
      act(() => {
        comp = renderer.create(<HeroWipeWeb />);
      });

      // Single image only — no wipe, no clipped top layer, no next-image base.
      expect(queryImages(comp).length).toBe(1);
      // No effect strip is ever mounted under reduced motion.
      expect(queryAll(comp, "hero-wipe-effect-strip").length).toBe(0);
      // useHeroWipeSchedule is called with paused=true under reduced motion,
      // which skips the setInterval clock entirely — not just a frozen one.
      expect(setIntervalSpy).not.toHaveBeenCalled();

      setIntervalSpy.mockRestore();
      act(() => comp.unmount());
    });

    it("still sets pointerEvents none on the static banner", () => {
      mockReducedMotion.mockReturnValue(true);
      let comp!: renderer.ReactTestRenderer;
      act(() => {
        comp = renderer.create(<HeroWipeWeb />);
      });
      const hero = queryAll(comp, "hero-wipe")[0];
      expect(hero.props.pointerEvents).toBe("none");
      act(() => comp.unmount());
    });
  });
});
