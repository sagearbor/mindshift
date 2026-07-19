import React, { useEffect, useRef, useState } from "react";
import { Animated, StyleSheet, View, AccessibilityInfo } from "react-native";

interface PulseDotProps {
  /** Dot + halo color. Defaults to the house blue. */
  color?: string;
  /** Diameter of the solid dot in px (the halo scales from this). */
  size?: number;
  testID?: string;
  /** Force-disable motion (callers / tests). When false the static dot renders
   *  with no animation and no halo. */
  animate?: boolean;
}

// Deliberate restraint: a few gentle cycles, then still. Motion here is a
// pointer at real signal, not decoration, so it must settle rather than loop
// forever.
const PULSE_CYCLES = 3;

/**
 * A small attention marker: a solid dot with a soft radial halo that pulses a
 * few times and then settles to a static dot. Used SPARINGLY — only where a
 * genuine, data-backed delta wants the eye (e.g. a re-analysis that actually
 * changed a score). Respects the OS "reduce motion" setting (and an explicit
 * `animate={false}`) by rendering the static dot with no animation at all, so a
 * motion-sensitive user is never subjected to it.
 */
export default function PulseDot({
  color = "#4A90D9",
  size = 10,
  testID,
  animate = true,
}: PulseDotProps) {
  const scale = useRef(new Animated.Value(1)).current;
  const opacity = useRef(new Animated.Value(0.4)).current;
  // The halo only mounts once we've confirmed motion is allowed — so reduced-
  // motion (and tests) render a clean static dot with no animated node at all.
  const [haloOn, setHaloOn] = useState(false);

  useEffect(() => {
    if (!animate) return;
    let cancelled = false;
    let running: Animated.CompositeAnimation | null = null;
    const query =
      AccessibilityInfo?.isReduceMotionEnabled?.() ?? Promise.resolve(false);
    query
      .catch(() => false)
      .then((reduceMotion) => {
        if (cancelled || reduceMotion) return;
        setHaloOn(true);
        running = Animated.loop(
          Animated.sequence([
            Animated.parallel([
              Animated.timing(scale, {
                toValue: 2.4,
                duration: 700,
                useNativeDriver: false,
              }),
              Animated.timing(opacity, {
                toValue: 0,
                duration: 700,
                useNativeDriver: false,
              }),
            ]),
            // Reset instantly for the next cycle.
            Animated.parallel([
              Animated.timing(scale, {
                toValue: 1,
                duration: 0,
                useNativeDriver: false,
              }),
              Animated.timing(opacity, {
                toValue: 0.4,
                duration: 0,
                useNativeDriver: false,
              }),
            ]),
          ]),
          { iterations: PULSE_CYCLES },
        );
        running.start(() => {
          // Settle: halo fully faded, dot stays.
          if (!cancelled) opacity.setValue(0);
        });
      });
    return () => {
      cancelled = true;
      running?.stop();
    };
  }, [animate, scale, opacity]);

  const halo = size * 2.4;
  return (
    <View
      testID={testID}
      style={[styles.wrap, { width: size, height: size }]}
      pointerEvents="none"
    >
      {haloOn && (
        <Animated.View
          testID={testID ? `${testID}-halo` : undefined}
          style={[
            styles.halo,
            {
              width: halo,
              height: halo,
              borderRadius: halo / 2,
              backgroundColor: color,
              left: (size - halo) / 2,
              top: (size - halo) / 2,
              opacity,
              transform: [{ scale }],
            },
          ]}
        />
      )}
      <View
        style={[
          styles.dot,
          {
            width: size,
            height: size,
            borderRadius: size / 2,
            backgroundColor: color,
          },
        ]}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    alignItems: "center",
    justifyContent: "center",
  },
  halo: {
    position: "absolute",
  },
  dot: {},
});
