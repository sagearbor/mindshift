import React, { useState } from "react";
import { View, Text, TouchableOpacity, StyleSheet } from "react-native";

import { getDestination, type DestScreen } from "../nav/destinations";
import { useLayoutStore } from "../store/layoutStore";
import { getIcon, CHROME_ICONS } from "./icons";
import Avatar, { type AvatarUser } from "./Avatar";
import DestinationCatalog from "./DestinationCatalog";
import AccountMenu from "./AccountMenu";

const Menu = CHROME_ICONS.menu;

interface AppChromeProps {
  /** The `name` of the Screen union member currently on screen — used only
   *  to highlight the matching tab, if any (a tab whose destination's
   *  `screen.name` doesn't match the current screen just renders inactive,
   *  never wrong). */
  screenName: string;
  /** Hand a destination's screen straight to App.tsx's setScreen — the
   *  hamburger catalog, the tab bar, and the account menu's Settings row all
   *  go through this one callback. */
  onNavigate: (screen: DestScreen) => void;
  /** Wordmark tap → Home. "Home" has no DestId in the registry (see
   *  nav/destinations.ts) — it's the app's root, not a destination you
   *  navigate TO — so it needs its own way back onto the primary surface,
   *  independent of the (registry-only) hamburger catalog and the
   *  (possibly zero-slot) tab bar. */
  onGoHome: () => void;
  onSignOut: () => void;
  user: AvatarUser | null;
  /** Task N6's slot: a captured selfie uri, once that task lands. Undefined
   *  today — Avatar falls back to the account's initial until then. */
  avatarUri?: string | null;
  children: React.ReactNode;
}

/**
 * The chrome around every PRIMARY screen (Task N3 of P3-10): a top bar
 * (hamburger → full destination catalog, wordmark → Home, avatar → account
 * menu) plus a bottom tab bar rendered from layoutStore.tabSlots (hidden
 * entirely at 0 slots). App.tsx decides which screens count as "primary"
 * (see its PRIMARY_SCREEN_NAMES comment) — this component just renders the
 * chrome and the two overlays it can open.
 */
export default function AppChrome({
  screenName,
  onNavigate,
  onGoHome,
  onSignOut,
  user,
  avatarUri,
  children,
}: AppChromeProps) {
  const [catalogOpen, setCatalogOpen] = useState(false);
  const [accountOpen, setAccountOpen] = useState(false);
  const tabSlots = useLayoutStore((s) => s.tabSlots);

  const navigate = (screen: DestScreen) => {
    setCatalogOpen(false);
    setAccountOpen(false);
    onNavigate(screen);
  };

  const openSettings = () => {
    const settings = getDestination("settings");
    if (settings) navigate(settings.screen);
  };

  return (
    <View style={styles.container} testID="app-chrome">
      <View style={styles.topBar}>
        <TouchableOpacity
          testID="chrome-hamburger-button"
          accessibilityRole="button"
          accessibilityLabel="Open menu"
          style={styles.iconButton}
          onPress={() => setCatalogOpen(true)}
          hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
        >
          <Menu size={22} />
        </TouchableOpacity>

        <TouchableOpacity
          testID="chrome-wordmark"
          accessibilityRole="button"
          accessibilityLabel="Home"
          onPress={onGoHome}
        >
          <Text style={styles.wordmark}>MindShift</Text>
        </TouchableOpacity>

        <TouchableOpacity
          testID="chrome-avatar-button"
          accessibilityRole="button"
          accessibilityLabel="Account"
          style={styles.iconButton}
          onPress={() => setAccountOpen(true)}
        >
          <Avatar user={user} photoUri={avatarUri} testID="chrome-avatar" />
        </TouchableOpacity>
      </View>

      <View style={styles.content}>{children}</View>

      {tabSlots.length > 0 ? (
        <View style={styles.tabBar} testID="chrome-tab-bar">
          {tabSlots.map((id) => {
            const dest = getDestination(id);
            if (!dest) return null; // stale persisted id — drop it silently
            const Icon = getIcon(dest.iconId);
            const active = dest.screen.name === screenName;
            return (
              <TouchableOpacity
                key={id}
                testID={`chrome-tab-${id}`}
                accessibilityRole="button"
                accessibilityLabel={dest.title}
                accessibilityState={{ selected: active }}
                style={styles.tab}
                onPress={() => navigate(dest.screen)}
              >
                <Icon size={22} color={active ? "#4A90D9" : "#6B7280"} />
                <Text
                  style={[styles.tabLabel, active && styles.tabLabelActive]}
                  numberOfLines={1}
                >
                  {dest.title}
                </Text>
              </TouchableOpacity>
            );
          })}
        </View>
      ) : null}

      {catalogOpen ? (
        <DestinationCatalog
          onSelect={navigate}
          onClose={() => setCatalogOpen(false)}
        />
      ) : null}

      {accountOpen ? (
        <AccountMenu
          user={user}
          onOpenSettings={openSettings}
          onSignOut={() => {
            setAccountOpen(false);
            onSignOut();
          }}
          onClose={() => setAccountOpen(false)}
        />
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  topBar: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: 20,
    paddingTop: 12,
    paddingBottom: 12,
  },
  iconButton: {
    width: 40,
    height: 40,
    alignItems: "center",
    justifyContent: "center",
  },
  wordmark: {
    fontSize: 18,
    fontWeight: "700",
    color: "#111827",
  },
  content: {
    flex: 1,
  },
  tabBar: {
    flexDirection: "row",
    borderTopWidth: 1,
    borderTopColor: "#E5E7EB",
    backgroundColor: "#FFFFFF",
    paddingTop: 8,
    paddingBottom: 12,
    paddingHorizontal: 8,
  },
  tab: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    gap: 2,
    paddingHorizontal: 4,
  },
  tabLabel: {
    fontSize: 11,
    fontWeight: "600",
    color: "#6B7280",
  },
  tabLabelActive: {
    color: "#4A90D9",
  },
});
