/**
 * Effect registry for the hero wipe's traveling edge (Task P3-4b). Each
 * effect owns a narrow canvas strip anchored to the wipe line and draws a
 * cheap, restrained particle treatment — never a spectacle. Particle counts
 * are small and 60fps is not a goal; "graceful" is.
 *
 * Deliberately isolated from React: every runner takes a plain canvas + size
 * and returns a cleanup function. `HeroWipe.tsx` is responsible for calling
 * these only on web, only outside prefers-reduced-motion, and only while a
 * canvas 2D context is actually available — but each runner ALSO guards
 * itself (no context, canvas absent, rAF absent) so its absence — in a test
 * environment, under reduced motion, on an unsupported browser — can never
 * throw or leave a dangling timer.
 *
 * To add an effect: write a runner matching HeroWipeEffectRunner, add it to
 * HERO_WIPE_EFFECT_REGISTRY under a new HeroWipeEffectId (see
 * heroWipeSchedule.ts), and append that id to HERO_WIPE_EFFECTS to fold it
 * into the rotation.
 */
import type { HeroWipeEffectId } from "./heroWipeSchedule";

export interface HeroWipeEffectContext {
  canvas: HTMLCanvasElement;
  width: number;
  height: number;
}

export type HeroWipeCleanup = () => void;
export type HeroWipeEffectRunner = (ctx: HeroWipeEffectContext) => HeroWipeCleanup;

const NOOP: HeroWipeCleanup = () => {};

interface Particle {
  x: number; // 0..width
  y: number; // 0..height, may drift outside — wrapped
  vx: number;
  vy: number;
  r: number;
  seed: number;
}

function makeParticles(count: number, width: number, height: number): Particle[] {
  const particles: Particle[] = [];
  for (let i = 0; i < count; i++) {
    particles.push({
      x: Math.random() * width,
      y: Math.random() * height,
      vx: (Math.random() - 0.5) * 0.15,
      vy: 0,
      r: 1 + Math.random() * 2,
      seed: Math.random() * Math.PI * 2,
    });
  }
  return particles;
}

/** Wraps a raw draw loop with the shared guard: no context / rAF → no-op. */
function safeRunner(
  build: (
    ctx2d: CanvasRenderingContext2D,
    width: number,
    height: number,
  ) => HeroWipeCleanup,
): HeroWipeEffectRunner {
  return ({ canvas, width, height }) => {
    try {
      if (typeof canvas.getContext !== "function") return NOOP;
      const ctx2d = canvas.getContext("2d");
      if (!ctx2d) return NOOP;
      if (
        typeof requestAnimationFrame !== "function" ||
        typeof cancelAnimationFrame !== "function"
      ) {
        return NOOP;
      }
      return build(ctx2d, width, height);
    } catch {
      return NOOP;
    }
  };
}

/** Warm embers drifting upward off the wipe line, flickering out. */
const runEmbers: HeroWipeEffectRunner = safeRunner((ctx2d, width, height) => {
  const particles = makeParticles(14, width, height);
  let frameId = 0;
  const tick = () => {
    ctx2d.clearRect(0, 0, width, height);
    const t = Date.now() / 1000;
    for (const p of particles) {
      p.y -= 0.35 + Math.abs(p.vx) * 2;
      p.x += Math.sin(t + p.seed) * 0.3;
      if (p.y < -4) {
        p.y = height + 4;
        p.x = Math.random() * width;
      }
      const flicker = 0.4 + 0.6 * Math.abs(Math.sin(t * 3 + p.seed));
      const grad = ctx2d.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.r * 3);
      grad.addColorStop(0, `rgba(255,180,80,${0.85 * flicker})`);
      grad.addColorStop(1, "rgba(255,80,20,0)");
      ctx2d.fillStyle = grad;
      ctx2d.beginPath();
      ctx2d.arc(p.x, p.y, p.r * 2.5, 0, Math.PI * 2);
      ctx2d.fill();
    }
    frameId = requestAnimationFrame(tick);
  };
  frameId = requestAnimationFrame(tick);
  return () => cancelAnimationFrame(frameId);
});

/** Pale blue jagged flicker — a cold crackle along the edge. */
const runIce: HeroWipeEffectRunner = safeRunner((ctx2d, width, height) => {
  const shards = 10;
  let frameId = 0;
  const tick = () => {
    ctx2d.clearRect(0, 0, width, height);
    const t = Date.now() / 1000;
    ctx2d.strokeStyle = "rgba(190,230,255,0.8)";
    ctx2d.lineWidth = 1;
    for (let i = 0; i < shards; i++) {
      const y = (i / shards) * height + Math.sin(t * 4 + i) * 3;
      const jag = 6 + Math.abs(Math.sin(t * 5 + i * 1.7)) * 10;
      const cx = width / 2;
      ctx2d.globalAlpha = 0.3 + 0.5 * Math.abs(Math.sin(t * 6 + i));
      ctx2d.beginPath();
      ctx2d.moveTo(cx - jag, y);
      ctx2d.lineTo(cx + jag, y + 2);
      ctx2d.stroke();
    }
    ctx2d.globalAlpha = 1;
    frameId = requestAnimationFrame(tick);
  };
  frameId = requestAnimationFrame(tick);
  return () => cancelAnimationFrame(frameId);
});

/** Small circles rising off the line, like bubbles through water. */
const runBubbles: HeroWipeEffectRunner = safeRunner((ctx2d, width, height) => {
  const particles = makeParticles(10, width, height);
  let frameId = 0;
  const tick = () => {
    ctx2d.clearRect(0, 0, width, height);
    const t = Date.now() / 1000;
    for (const p of particles) {
      p.y -= 0.5;
      p.x += Math.sin(t * 2 + p.seed) * 0.4;
      if (p.y < -4) {
        p.y = height + 4;
        p.x = Math.random() * width;
      }
      ctx2d.strokeStyle = "rgba(150,210,255,0.7)";
      ctx2d.lineWidth = 1;
      ctx2d.beginPath();
      ctx2d.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx2d.stroke();
    }
    frameId = requestAnimationFrame(tick);
  };
  frameId = requestAnimationFrame(tick);
  return () => cancelAnimationFrame(frameId);
});

export const HERO_WIPE_EFFECT_REGISTRY: Record<HeroWipeEffectId, HeroWipeEffectRunner> = {
  embers: runEmbers,
  ice: runIce,
  bubbles: runBubbles,
};
