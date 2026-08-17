import React, { useEffect, useRef, useState } from "react";
import { Image, Platform, StyleSheet, View, type LayoutChangeEvent } from "react-native";

import { usePrefersReducedMotion } from "../hooks/usePrefersReducedMotion";
import { HERO_IMAGES } from "../assets/heroImages";
import { HERO_WIPE_EFFECT_REGISTRY } from "../utils/heroWipeEffects";
import { useHeroWipeSchedule } from "../utils/heroWipeSchedule";

const EFFECT_STRIP_WIDTH = 60;

/**
 * The web home screen's hero banner (Task P3-4b): six owner-curated images,
 * each held ~7s then wiped away left→right over ~2.5s by a traveling edge
 * that reveals the next (stationary) image beneath it — a before/after
 * slider being dragged, never a slide-in. A small canvas strip rides the
 * edge with a particle effect that rotates per transition (embers / ice /
 * bubbles — see heroWipeEffects.ts).
 *
 * Web-only: native renders nothing (the plan defers mobile). Rather than a
 * .native.tsx/.web.tsx split (see GoogleSignInButton for that pattern), this
 * uses a runtime `Platform.OS` check like RecordScreen/AnalyzeScreen/
 * HeatChart do for their web-only bits — it keeps the wipe's DOM-refs-cast-
 * to-HTMLElement escape hatch (same trick HeatChart's wheel listener uses)
 * in one file, and makes the "web renders it, native doesn't" behavior
 * trivial to unit test by toggling Platform.OS.
 */
export default function HeroWipe() {
  if (Platform.OS !== "web") return null;
  return <HeroWipeWeb />;
}

function HeroWipeWeb() {
  const reducedMotion = usePrefersReducedMotion();
  // Reduced motion: no timers at all, just the first shuffled image, static.
  const schedule = useHeroWipeSchedule(HERO_IMAGES.length, undefined, reducedMotion);
  const topLayerRef = useRef<View>(null);
  const stripHostRef = useRef<View>(null);
  const [stripHeight, setStripHeight] = useState(0);

  const progressPct = schedule.progress * 100;

  // Drive clip-path on the top (current) image imperatively — react-native-web
  // Views are real DOM nodes on web, and clip-path has no RN StyleSheet type,
  // so we set it directly on the underlying element (the same ref-cast
  // pattern HeatChart uses for its web-only wheel listener). Clipping the
  // left `progress`% hides the wiped-away portion, revealing the stationary
  // next image underneath — never moves the image itself.
  useEffect(() => {
    if (reducedMotion) return;
    const node = topLayerRef.current as unknown as HTMLElement | null;
    if (!node || !node.style) return;
    node.style.clipPath = `inset(0 0 0 ${progressPct}%)`;
  }, [progressPct, reducedMotion]);

  // Mount the effect-strip canvas only while a wipe is actually in progress,
  // and tear it down the instant it ends — cheap by construction.
  useEffect(() => {
    if (reducedMotion || schedule.phase !== "wipe") return;
    if (typeof document === "undefined" || typeof document.createElement !== "function") {
      return;
    }
    const host = stripHostRef.current as unknown as HTMLElement | null;
    // In test environments (react-test-renderer) this ref is a test
    // instance, not a real DOM node — appendChild et al. don't exist there.
    // Guard every capability we're about to use so a test-only render never
    // throws, matching the same "absence must be safe" contract as the
    // effect runners themselves.
    if (
      !host ||
      typeof host.appendChild !== "function" ||
      typeof host.removeChild !== "function" ||
      typeof host.contains !== "function"
    ) {
      return;
    }

    let cleanupFns: (() => void) | undefined;
    try {
      const canvasEl = document.createElement("canvas");
      const height = Math.max(1, stripHeight || host.clientHeight || 1);
      canvasEl.width = EFFECT_STRIP_WIDTH;
      canvasEl.height = height;
      canvasEl.style.width = "100%";
      canvasEl.style.height = "100%";
      canvasEl.style.display = "block";
      while (host.firstChild) host.removeChild(host.firstChild); // clears any stale canvas — not untrusted content
      host.appendChild(canvasEl);

      const runner = HERO_WIPE_EFFECT_REGISTRY[schedule.effect];
      const cleanupEffect = runner({ canvas: canvasEl, width: EFFECT_STRIP_WIDTH, height });
      cleanupFns = () => {
        cleanupEffect();
        if (host.contains(canvasEl)) host.removeChild(canvasEl);
      };
    } catch {
      cleanupFns = undefined;
    }
    return cleanupFns;
  }, [schedule.phase, schedule.effect, reducedMotion, stripHeight]);

  function onLayout(e: LayoutChangeEvent) {
    setStripHeight(e.nativeEvent.layout.height);
  }

  if (reducedMotion) {
    return (
      <View
        testID="hero-wipe"
        style={styles.container}
        onLayout={onLayout}
        pointerEvents="none"
      >
        <Image
          source={HERO_IMAGES[schedule.currentIndex]}
          style={styles.image}
          resizeMode="cover"
        />
      </View>
    );
  }

  return (
    <View testID="hero-wipe" style={styles.container} onLayout={onLayout} pointerEvents="none">
      {/* Stationary base layer: the image being revealed. */}
      <Image
        source={HERO_IMAGES[schedule.nextIndex]}
        style={styles.image}
        resizeMode="cover"
      />
      {/* Top layer: the outgoing image, clipped from the left as the wipe
          travels — the edge moves, the pixels underneath never do. */}
      <View ref={topLayerRef} style={StyleSheet.absoluteFill}>
        <Image
          source={HERO_IMAGES[schedule.currentIndex]}
          style={styles.image}
          resizeMode="cover"
        />
      </View>
      {schedule.phase === "wipe" && (
        <View
          ref={stripHostRef}
          testID="hero-wipe-effect-strip"
          pointerEvents="none"
          style={[
            styles.effectStrip,
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            { left: `calc(${progressPct}% - ${EFFECT_STRIP_WIDTH / 2}px)` } as any,
          ]}
        />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    width: "100%",
    height: 200,
    borderRadius: 20,
    overflow: "hidden",
    marginBottom: 16,
    backgroundColor: "#111827",
  },
  image: {
    width: "100%",
    height: "100%",
  },
  effectStrip: {
    position: "absolute",
    top: 0,
    bottom: 0,
    width: EFFECT_STRIP_WIDTH,
  },
});
