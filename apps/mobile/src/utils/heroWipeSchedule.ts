/**
 * Pure scheduling engine for the web hero wipe-reveal (Task P3-4b). Owns
 * nothing but data: which image is showing, whether we're holding or mid-wipe,
 * how far the wipe has traveled, and which effect rides the current transition.
 * No timers, no DOM, no React — that lives in `useHeroWipeSchedule` below —
 * so the rotation/scheduling/effect-cycling logic is fully unit-testable with
 * plain deltas instead of fake-timer choreography.
 */
import { useEffect, useRef, useState } from "react";

export type HeroWipeEffectId = "embers" | "ice" | "bubbles";

/** Small, extendable registry order — add a new effect id here (and a runner
 * in heroWipeEffects.ts) to fold it into the rotation. */
export const HERO_WIPE_EFFECTS: readonly HeroWipeEffectId[] = [
  "embers",
  "ice",
  "bubbles",
];

export const HERO_WIPE_HOLD_MS = 7000;
export const HERO_WIPE_TRANSITION_MS = 2500;

export type HeroWipePhase = "hold" | "wipe";

export interface HeroWipeConfig {
  holdMs: number;
  transitionMs: number;
  effects: readonly HeroWipeEffectId[];
}

export const DEFAULT_HERO_WIPE_CONFIG: HeroWipeConfig = {
  holdMs: HERO_WIPE_HOLD_MS,
  transitionMs: HERO_WIPE_TRANSITION_MS,
  effects: HERO_WIPE_EFFECTS,
};

export interface HeroWipeState {
  /** Shuffled playback order — a permutation of [0, imageCount). */
  order: number[];
  /** Index into `order` for the image currently on top. */
  position: number;
  phase: HeroWipePhase;
  /** Milliseconds elapsed in the current phase. */
  phaseElapsedMs: number;
  /** Total completed wipes — drives effect rotation. */
  transitionCount: number;
}

/** Fisher-Yates with an injectable RNG so tests get deterministic orders. */
export function shuffleIndices(
  count: number,
  rng: () => number = Math.random,
): number[] {
  const arr = Array.from({ length: count }, (_, i) => i);
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

export function createHeroWipeState(
  imageCount: number,
  rng: () => number = Math.random,
): HeroWipeState {
  return {
    order: shuffleIndices(Math.max(imageCount, 0), rng),
    position: 0,
    phase: "hold",
    phaseElapsedMs: 0,
    transitionCount: 0,
  };
}

const MAX_TICK_ITERATIONS = 1000; // safety cap against zero/negative durations

/**
 * Advance the schedule by `deltaMs`. Pure: same inputs, same output, no
 * globals touched. Handles deltas larger than a single phase (e.g. a big
 * fake-timer jump) by looping through as many phase boundaries as the delta
 * crosses, so the engine never "sticks" mid-phase.
 */
export function tickHeroWipeState(
  state: HeroWipeState,
  deltaMs: number,
  config: HeroWipeConfig = DEFAULT_HERO_WIPE_CONFIG,
  rng: () => number = Math.random,
): HeroWipeState {
  if (state.order.length <= 1) {
    // Nothing to cycle between — stay put, but still track elapsed time.
    return { ...state, phaseElapsedMs: state.phaseElapsedMs + Math.max(0, deltaMs) };
  }

  let { order, position, phase, phaseElapsedMs, transitionCount } = state;
  let remaining = Math.max(0, deltaMs);
  let iterations = 0;

  while (remaining > 0 && iterations < MAX_TICK_ITERATIONS) {
    iterations++;
    const phaseDuration = phase === "hold" ? config.holdMs : config.transitionMs;
    const roomLeft = phaseDuration - phaseElapsedMs;

    if (remaining < roomLeft) {
      phaseElapsedMs += remaining;
      remaining = 0;
      break;
    }

    // Crosses (or exactly meets) the phase boundary — consume it and flip.
    remaining -= roomLeft;
    phaseElapsedMs = 0;

    if (phase === "hold") {
      phase = "wipe";
    } else {
      phase = "hold";
      transitionCount += 1;
      position += 1;
      if (position >= order.length) {
        position = 0;
        const next = shuffleIndices(order.length, rng);
        // Avoid an immediate repeat of the image we just finished showing.
        if (next[0] === order[order.length - 1] && next.length > 1) {
          [next[0], next[1]] = [next[1], next[0]];
        }
        order = next;
      }
    }
  }

  return { order, position, phase, phaseElapsedMs, transitionCount };
}

export function currentImageIndex(state: HeroWipeState): number {
  return state.order[state.position] ?? 0;
}

export function nextImageIndex(state: HeroWipeState): number {
  if (state.order.length === 0) return 0;
  const nextPos = (state.position + 1) % state.order.length;
  return state.order[nextPos];
}

/** 0 while holding; 0..1 fraction of the transition while wiping. */
export function wipeProgress(
  state: HeroWipeState,
  config: HeroWipeConfig = DEFAULT_HERO_WIPE_CONFIG,
): number {
  if (state.phase !== "wipe" || config.transitionMs <= 0) return 0;
  return Math.min(1, Math.max(0, state.phaseElapsedMs / config.transitionMs));
}

/** The effect assigned to the transition currently in progress (or the one
 * that just completed, while holding) — rotates through the registry. */
export function currentEffect(
  state: HeroWipeState,
  config: HeroWipeConfig = DEFAULT_HERO_WIPE_CONFIG,
): HeroWipeEffectId {
  if (config.effects.length === 0) return HERO_WIPE_EFFECTS[0];
  const idx = state.transitionCount % config.effects.length;
  return config.effects[idx];
}

export interface HeroWipeSnapshot {
  currentIndex: number;
  nextIndex: number;
  phase: HeroWipePhase;
  progress: number;
  effect: HeroWipeEffectId;
}

function snapshotOf(
  state: HeroWipeState,
  config: HeroWipeConfig,
): HeroWipeSnapshot {
  return {
    currentIndex: currentImageIndex(state),
    nextIndex: nextImageIndex(state),
    phase: state.phase,
    progress: wipeProgress(state, config),
    effect: currentEffect(state, config),
  };
}

const TICK_MS = 100;

/**
 * React hook wrapping the pure engine with a setInterval clock. `paused`
 * stops the clock entirely (used for prefers-reduced-motion, where we want a
 * single static image and no timers running at all).
 */
export function useHeroWipeSchedule(
  imageCount: number,
  config: HeroWipeConfig = DEFAULT_HERO_WIPE_CONFIG,
  paused = false,
): HeroWipeSnapshot {
  const [state, setState] = useState(() => createHeroWipeState(imageCount));
  const lastRef = useRef<number>(Date.now());
  const imageCountRef = useRef(imageCount);
  imageCountRef.current = imageCount;

  // Re-seed if the image count changes (e.g. images finish loading later).
  useEffect(() => {
    setState((prev) =>
      prev.order.length === imageCount ? prev : createHeroWipeState(imageCount),
    );
  }, [imageCount]);

  useEffect(() => {
    if (paused || imageCount <= 1) return;
    lastRef.current = Date.now();
    const id = setInterval(() => {
      const now = Date.now();
      const delta = now - lastRef.current;
      lastRef.current = now;
      setState((prev) => tickHeroWipeState(prev, delta, config));
    }, TICK_MS);
    return () => clearInterval(id);
    // config is a stable module constant in normal use; re-running the
    // interval on identity change is intentionally avoided to keep the clock
    // steady across re-renders.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [paused, imageCount]);

  return snapshotOf(state, config);
}
