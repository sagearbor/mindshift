import React, { useCallback, useRef, useState } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  StyleSheet,
  Dimensions,
  type LayoutChangeEvent,
  type NativeSyntheticEvent,
  type NativeScrollEvent,
} from "react-native";

import {
  ONBOARDING_CARDS,
  clampCardIndex,
  nextCardIndex,
  prevCardIndex,
  isLastCard,
} from "./onboardingCards";

interface OnboardingScreenProps {
  /** Fired once — from either Skip (any card) or "Get started" (last card).
   *  The caller (App.tsx) persists the seen-state and returns to the app;
   *  this component never touches storage itself, keeping it a pure render +
   *  local-index component that's easy to test in isolation. */
  onFinish: () => void;
}

/**
 * First-launch onboarding walkthrough (Task P3-7): four swipeable,
 * skippable cards — Live Coach, Analyze a conversation, Your watch, Growth.
 * Shown once after sign-in (App.tsx gates that); re-runnable anytime from
 * Settings → "Show tutorial".
 *
 * Swiping is a real horizontal ScrollView with paging; the Back/Next/Skip
 * buttons are the primary, always-testable way to move (swipe gestures
 * aren't something a jest render test can simulate, and some users won't
 * discover swiping at all — the buttons are not a fallback, they're the
 * accessible path).
 */
export default function OnboardingScreen({ onFinish }: OnboardingScreenProps) {
  const [index, setIndex] = useState(0);
  const [width, setWidth] = useState(
    () => Dimensions.get("window").width || 320,
  );
  const scrollRef = useRef<ScrollView>(null);

  const goTo = useCallback(
    (next: number) => {
      const clamped = clampCardIndex(next, ONBOARDING_CARDS.length);
      setIndex(clamped);
      scrollRef.current?.scrollTo({ x: clamped * width, animated: true });
    },
    [width],
  );

  const handleLayout = useCallback((e: LayoutChangeEvent) => {
    const w = e.nativeEvent.layout.width;
    if (w > 0) setWidth(w);
  }, []);

  // Swiping past a page boundary updates `index` to match where the
  // ScrollView actually landed, so the dots/buttons stay in sync with a
  // manual swipe, not just button taps.
  const handleMomentumEnd = useCallback(
    (e: NativeSyntheticEvent<NativeScrollEvent>) => {
      const x = e.nativeEvent.contentOffset.x;
      const w = width || 1;
      setIndex(clampCardIndex(Math.round(x / w), ONBOARDING_CARDS.length));
    },
    [width],
  );

  const last = isLastCard(index, ONBOARDING_CARDS.length);

  return (
    <View style={styles.flex} testID="onboarding-screen">
      <View style={styles.topRow}>
        <View style={styles.dots} testID="onboarding-dots">
          {ONBOARDING_CARDS.map((card, i) => (
            <View
              key={card.id}
              testID={`onboarding-dot-${card.id}`}
              style={[styles.dot, i === index && styles.dotActive]}
            />
          ))}
        </View>
        <TouchableOpacity
          testID="onboarding-skip"
          accessibilityRole="button"
          style={styles.skipButton}
          onPress={onFinish}
        >
          <Text style={styles.skipText}>Skip</Text>
        </TouchableOpacity>
      </View>

      <ScrollView
        ref={scrollRef}
        testID="onboarding-scrollview"
        horizontal
        pagingEnabled
        showsHorizontalScrollIndicator={false}
        onLayout={handleLayout}
        onMomentumScrollEnd={handleMomentumEnd}
        style={styles.flex}
      >
        {ONBOARDING_CARDS.map((card) => (
          <View
            key={card.id}
            testID={`onboarding-card-${card.id}`}
            style={[styles.card, { width }]}
          >
            <Text style={styles.glyph} accessibilityElementsHidden>
              {card.glyph}
            </Text>
            <Text style={styles.title} testID={`onboarding-title-${card.id}`}>
              {card.title}
            </Text>
            <Text style={styles.body} testID={`onboarding-body-${card.id}`}>
              {card.body}
            </Text>
          </View>
        ))}
      </ScrollView>

      <View style={styles.bottomRow}>
        <TouchableOpacity
          testID="onboarding-back"
          accessibilityRole="button"
          style={[styles.navButton, index === 0 && styles.navButtonHidden]}
          onPress={() => goTo(prevCardIndex(index, ONBOARDING_CARDS.length))}
          disabled={index === 0}
        >
          <Text style={styles.navButtonText}>Back</Text>
        </TouchableOpacity>

        <TouchableOpacity
          testID={last ? "onboarding-get-started" : "onboarding-next"}
          accessibilityRole="button"
          style={[styles.navButton, styles.primaryButton]}
          onPress={() =>
            last ? onFinish() : goTo(nextCardIndex(index, ONBOARDING_CARDS.length))
          }
        >
          <Text style={styles.primaryButtonText}>
            {last ? "Get started" : "Next"}
          </Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  flex: {
    flex: 1,
  },
  topRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: 20,
    paddingTop: 16,
  },
  dots: {
    flexDirection: "row",
    gap: 6,
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: "#D1D5DB",
  },
  dotActive: {
    backgroundColor: "#4A90D9",
    width: 20,
  },
  skipButton: {
    minHeight: 44,
    minWidth: 44,
    justifyContent: "center",
    paddingHorizontal: 8,
  },
  skipText: {
    fontSize: 15,
    fontWeight: "600",
    color: "#6B7280",
  },
  card: {
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: 32,
    paddingVertical: 24,
  },
  glyph: {
    fontSize: 72,
    marginBottom: 24,
  },
  title: {
    fontSize: 24,
    fontWeight: "700",
    color: "#111827",
    textAlign: "center",
    marginBottom: 12,
  },
  body: {
    fontSize: 16,
    lineHeight: 23,
    color: "#4B5563",
    textAlign: "center",
  },
  bottomRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: 20,
    paddingBottom: 24,
    paddingTop: 8,
  },
  navButton: {
    minHeight: 48,
    minWidth: 88,
    justifyContent: "center",
    alignItems: "center",
    paddingHorizontal: 16,
  },
  navButtonHidden: {
    opacity: 0,
  },
  navButtonText: {
    fontSize: 16,
    fontWeight: "600",
    color: "#4A90D9",
  },
  primaryButton: {
    backgroundColor: "#4A90D9",
    borderRadius: 12,
    minWidth: 140,
  },
  primaryButtonText: {
    fontSize: 16,
    fontWeight: "700",
    color: "#FFFFFF",
  },
});
