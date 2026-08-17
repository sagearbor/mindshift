import React from "react";
import { View, Text, TouchableOpacity, ScrollView, StyleSheet } from "react-native";

import {
  PRIMARY_ELIGIBLE_DESTINATIONS,
  getDestination,
  type DestId,
} from "../nav/destinations";
import {
  useLayoutStore,
  TAB_SLOT_CAP,
  HOME_BOX_CAP,
} from "../store/layoutStore";
import { getIcon } from "../components/icons";

interface HomeDesignScreenProps {
  onBack: () => void;
}

/**
 * Settings → "Home screen design" (Task N5 of P3-10) — the owner's editor
 * for the two configurable surfaces N1's layoutStore persists: the bottom
 * bar (0–5 slots) and the home boxes (0–4). Every mutation goes straight
 * through layoutStore's validated setters (`setTabSlots`/`setHomeBoxes`) —
 * this screen never touches storage itself and never guesses at
 * dedup/eligibility/cap rules; the store already enforces all of that
 * (src/store/layoutStore.ts's `sanitizeSlots`), so an add/remove/reorder
 * here can never produce an invalid list. Because AppChrome's tab bar and
 * HomeScreen's box grid both subscribe to the same store, an edit here
 * reflects app-wide the instant the setter runs — no separate "apply" step.
 *
 * Reorder v1 is up/down buttons, not drag. The codebase's one existing
 * gesture-responder precedent (HeatChart's PanResponder) drives a single
 * continuous value drag inside one SVG, not cross-item list reordering with
 * drop-target detection — there's no drag-and-drop list primitive anywhere
 * in this repo to build on, so attempting real press-and-drag here would be
 * new, untested gesture-recognition code, not a low-risk reuse. Buttons are
 * the honest v1 per the plan; press-and-drag reordering is a ledgered
 * follow-up (P3-10 N5 follow-up: drag-to-reorder), not silently dropped.
 *
 * Reset to defaults is deliberately confirmation-free (owner directive,
 * P3-10 N5 scope): a mis-tap just means re-editing, never data loss beyond
 * this screen's own settings.
 */
export default function HomeDesignScreen({ onBack }: HomeDesignScreenProps) {
  const tabSlots = useLayoutStore((s) => s.tabSlots);
  const homeBoxes = useLayoutStore((s) => s.homeBoxes);
  const setTabSlots = useLayoutStore((s) => s.setTabSlots);
  const setHomeBoxes = useLayoutStore((s) => s.setHomeBoxes);
  const resetToDefaults = useLayoutStore((s) => s.resetToDefaults);

  return (
    <ScrollView
      style={styles.flex}
      contentContainerStyle={styles.content}
      testID="home-design-screen"
    >
      <TouchableOpacity
        testID="home-design-back"
        accessibilityRole="button"
        style={styles.backButton}
        onPress={onBack}
      >
        <Text style={styles.backText}>← Back</Text>
      </TouchableOpacity>

      <Text style={styles.heading} testID="home-design-heading">
        Home screen design
      </Text>
      <Text style={styles.intro}>
        Choose what shows up on your bottom bar and home screen, and put them
        in the order you want. Changes apply immediately, everywhere.
      </Text>

      <Text style={styles.sectionHeading}>Preview</Text>
      <View style={styles.preview} testID="home-design-preview">
        {tabSlots.length === 0 ? (
          <Text style={styles.previewEmpty} testID="home-design-preview-empty">
            No tabs — the bottom bar won’t show at all.
          </Text>
        ) : (
          tabSlots.map((id) => {
            const dest = getDestination(id);
            if (!dest) return null; // stale persisted id — drop it silently
            const Icon = getIcon(dest.iconId);
            return (
              <View
                key={id}
                style={styles.previewTab}
                testID={`home-design-preview-tab-${id}`}
              >
                <Icon size={20} />
                <Text style={styles.previewTabText} numberOfLines={1}>
                  {dest.title}
                </Text>
              </View>
            );
          })
        )}
      </View>

      <SlotSection
        prefix="tab"
        heading="Bottom bar"
        description="Up to 5 shortcuts, always visible at the bottom of the app."
        emptyHint="No tabs — the bottom bar is hidden entirely."
        ids={tabSlots}
        cap={TAB_SLOT_CAP}
        onChange={setTabSlots}
      />

      <SlotSection
        prefix="box"
        heading="Home boxes"
        description="Up to 4 icon shortcuts on your home screen."
        emptyHint="No boxes — home shows a hint to add some instead."
        ids={homeBoxes}
        cap={HOME_BOX_CAP}
        onChange={setHomeBoxes}
      />

      <TouchableOpacity
        testID="home-design-reset"
        accessibilityRole="button"
        style={styles.resetButton}
        onPress={resetToDefaults}
      >
        <Text style={styles.resetText}>Reset to defaults</Text>
      </TouchableOpacity>
    </ScrollView>
  );
}

/** Move the item at `index` one step earlier (dir -1) or later (dir +1).
 *  Out-of-bounds moves (top item up, bottom item down) are a no-op —
 *  callers don't need their own bounds check before wiring an up/down
 *  button's onPress straight to this. */
function moveItem<T>(list: readonly T[], index: number, dir: -1 | 1): T[] {
  const target = index + dir;
  if (target < 0 || target >= list.length) return [...list];
  const next = [...list];
  [next[index], next[target]] = [next[target], next[index]];
  return next;
}

/**
 * One editable slot list — the bottom bar or the home boxes — rendered
 * identically apart from copy/cap/testID prefix. Shared here rather than
 * duplicated because the add/remove/reorder rules (and their tests) are the
 * same shape for both; only which layoutStore setter `onChange` calls
 * differs, and that's the caller's concern, not this component's.
 */
function SlotSection({
  prefix,
  heading,
  description,
  emptyHint,
  ids,
  cap,
  onChange,
}: {
  prefix: "tab" | "box";
  heading: string;
  description: string;
  emptyHint: string;
  ids: readonly DestId[];
  cap: number;
  onChange: (next: DestId[]) => void;
}) {
  const idSet = new Set(ids);
  const remaining = PRIMARY_ELIGIBLE_DESTINATIONS.filter(
    (d) => !idSet.has(d.id),
  );
  const atCap = ids.length >= cap;

  return (
    <View style={styles.section} testID={`home-design-section-${prefix}`}>
      <Text style={styles.sectionHeading}>
        {heading} ({ids.length}/{cap})
      </Text>
      <Text style={styles.sectionDescription}>{description}</Text>

      {ids.length === 0 ? (
        <Text
          style={styles.emptyHint}
          testID={`home-design-${prefix}-empty`}
        >
          {emptyHint}
        </Text>
      ) : (
        ids.map((id, index) => {
          const dest = getDestination(id);
          if (!dest) return null; // stale persisted id — drop it silently
          const Icon = getIcon(dest.iconId);
          return (
            <View
              key={id}
              style={styles.slotRow}
              testID={`home-design-${prefix}-item-${id}`}
            >
              <Icon size={22} />
              <Text style={styles.slotTitle} numberOfLines={1}>
                {dest.title}
              </Text>
              <View style={styles.slotActions}>
                <TouchableOpacity
                  testID={`home-design-${prefix}-up-${id}`}
                  accessibilityRole="button"
                  accessibilityLabel={`Move ${dest.title} up`}
                  style={styles.slotButton}
                  onPress={() => onChange(moveItem(ids, index, -1))}
                >
                  <Text style={styles.slotButtonText}>↑</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  testID={`home-design-${prefix}-down-${id}`}
                  accessibilityRole="button"
                  accessibilityLabel={`Move ${dest.title} down`}
                  style={styles.slotButton}
                  onPress={() => onChange(moveItem(ids, index, 1))}
                >
                  <Text style={styles.slotButtonText}>↓</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  testID={`home-design-${prefix}-remove-${id}`}
                  accessibilityRole="button"
                  accessibilityLabel={`Remove ${dest.title}`}
                  style={styles.slotButton}
                  onPress={() =>
                    onChange(ids.filter((existing) => existing !== id))
                  }
                >
                  <Text style={styles.slotRemoveText}>✕</Text>
                </TouchableOpacity>
              </View>
            </View>
          );
        })
      )}

      {atCap || remaining.length === 0 ? (
        // Two distinct reasons land here — the cap is full, or (today, with
        // only 4 primary-eligible destinations registered against a 5-slot
        // tab cap) every eligible destination is already placed — but both
        // mean the same thing to the user: nothing left to add right now.
        <Text style={styles.capHint} testID={`home-design-${prefix}-cap-hint`}>
          {atCap
            ? "Full — remove one to add another."
            : "Nothing else to add yet — every shortcut is already placed."}
        </Text>
      ) : (
        <View testID={`home-design-${prefix}-add-list`}>
          <Text style={styles.addHeading}>Add</Text>
          {remaining.map((dest) => {
            const Icon = getIcon(dest.iconId);
            return (
              <TouchableOpacity
                key={dest.id}
                testID={`home-design-${prefix}-add-${dest.id}`}
                accessibilityRole="button"
                accessibilityLabel={`Add ${dest.title}`}
                style={styles.addRow}
                onPress={() => onChange([...ids, dest.id])}
              >
                <Icon size={20} />
                <Text style={styles.addRowText}>{dest.title}</Text>
              </TouchableOpacity>
            );
          })}
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  flex: {
    flex: 1,
  },
  content: {
    paddingTop: 24,
    paddingHorizontal: 20,
    paddingBottom: 40,
  },
  backButton: {
    alignSelf: "flex-start",
    minHeight: 44,
    justifyContent: "center",
    paddingRight: 12,
    marginBottom: 4,
  },
  backText: {
    fontSize: 16,
    fontWeight: "600",
    color: "#4A90D9",
  },
  heading: {
    fontSize: 24,
    fontWeight: "700",
    color: "#111827",
    marginBottom: 8,
  },
  intro: {
    fontSize: 14,
    lineHeight: 20,
    color: "#6B7280",
    marginBottom: 20,
  },
  preview: {
    flexDirection: "row",
    borderWidth: 1,
    borderColor: "#D1D5DB",
    borderRadius: 14,
    backgroundColor: "#FFFFFF",
    paddingVertical: 12,
    paddingHorizontal: 8,
    marginBottom: 20,
    gap: 4,
  },
  previewEmpty: {
    fontSize: 13.5,
    color: "#6B7280",
    fontStyle: "italic",
    paddingHorizontal: 8,
  },
  previewTab: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    gap: 2,
    paddingHorizontal: 2,
  },
  previewTabText: {
    fontSize: 10.5,
    fontWeight: "600",
    color: "#6B7280",
    textAlign: "center",
  },
  section: {
    marginBottom: 24,
  },
  sectionHeading: {
    fontSize: 13,
    fontWeight: "700",
    letterSpacing: 0.6,
    textTransform: "uppercase",
    color: "#9CA3AF",
    marginBottom: 4,
  },
  sectionDescription: {
    fontSize: 13,
    color: "#6B7280",
    marginBottom: 12,
  },
  emptyHint: {
    fontSize: 13.5,
    color: "#6B7280",
    fontStyle: "italic",
    marginBottom: 12,
  },
  slotRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    minHeight: 52,
    borderWidth: 1,
    borderColor: "#D1D5DB",
    borderRadius: 12,
    backgroundColor: "#FFFFFF",
    paddingHorizontal: 14,
    marginBottom: 8,
  },
  slotTitle: {
    flex: 1,
    fontSize: 15,
    fontWeight: "600",
    color: "#1F2937",
  },
  slotActions: {
    flexDirection: "row",
    gap: 4,
  },
  slotButton: {
    width: 36,
    height: 36,
    borderRadius: 10,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#F3F4F6",
  },
  slotButtonText: {
    fontSize: 16,
    fontWeight: "700",
    color: "#4A90D9",
  },
  slotRemoveText: {
    fontSize: 15,
    fontWeight: "700",
    color: "#DC2626",
  },
  capHint: {
    fontSize: 13,
    color: "#9CA3AF",
    fontStyle: "italic",
  },
  addHeading: {
    fontSize: 12.5,
    fontWeight: "700",
    color: "#9CA3AF",
    marginBottom: 6,
  },
  addRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
    minHeight: 44,
    borderRadius: 10,
    paddingHorizontal: 12,
    borderWidth: 1,
    borderColor: "#E5E7EB",
    backgroundColor: "#F9FAFB",
    marginBottom: 6,
  },
  addRowText: {
    fontSize: 14.5,
    fontWeight: "600",
    color: "#1F2937",
  },
  resetButton: {
    marginTop: 8,
    minHeight: 48,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#D1D5DB",
    alignItems: "center",
    justifyContent: "center",
  },
  resetText: {
    fontSize: 15,
    fontWeight: "700",
    color: "#6B7280",
  },
});
