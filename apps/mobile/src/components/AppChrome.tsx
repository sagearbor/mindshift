import React, { forwardRef, useImperativeHandle, useState } from "react";
import { View, Text, TouchableOpacity, StyleSheet } from "react-native";

import type { Screen } from "../../App";
import { getDestination, type DestScreen } from "../nav/destinations";
import { CHROME_ICONS } from "./icons";
import ChromeTabBar from "./ChromeTabBar";
import Avatar, { type AvatarUser } from "./Avatar";
import DestinationCatalog from "./DestinationCatalog";
import AccountMenu from "./AccountMenu";

const Menu = CHROME_ICONS.menu;

interface AppChromeProps {
  /** The `name` of the Screen union member currently on screen — used only
   *  to highlight the matching tab, if any (a tab whose destination's
   *  `screen.name` doesn't match the current screen just renders inactive,
   *  never wrong). */
  screenName: Screen["name"];
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
  /** The captured selfie's uri (Task N6, avatarStore), or null/undefined —
   *  Avatar falls back to the account's initial when unset. */
  avatarUri?: string | null;
  children: React.ReactNode;
}

/** Imperative handle (Task N3 fix round 1, CRITICAL 1) so App.tsx's Android
 *  hardware-back wiring can close an open overlay BEFORE it falls through to
 *  screen navigation or double-back-to-exit — back used to act right
 *  underneath an open catalog/account menu, which could exit the app with a
 *  menu still open. Overlay open/closed state stays encapsulated inside this
 *  component (easier to test, no prop drilling from App.tsx) — this ref is
 *  the one deliberate escape hatch for the one caller that genuinely needs
 *  to reach in from outside: the back handler. */
export interface AppChromeHandle {
  /** Closes any open overlay and returns true if one was actually open;
   *  returns false (no-op) when there was nothing to close. */
  closeOverlays: () => boolean;
}

/**
 * The chrome around every PRIMARY screen (Task N3 of P3-10): a top bar
 * (hamburger → full destination catalog, wordmark → Home, avatar → account
 * menu) plus a bottom tab bar rendered from layoutStore.tabSlots (hidden
 * entirely at 0 slots). App.tsx decides which screens count as "primary"
 * (see its PRIMARY_SCREEN_NAMES/isPrimary comments) — this component just
 * renders the chrome and the two overlays it can open.
 */
const AppChrome = forwardRef<AppChromeHandle, AppChromeProps>(
  function AppChrome(
    {
      screenName,
      onNavigate,
      onGoHome,
      onSignOut,
      user,
      avatarUri,
      children,
    },
    ref,
  ) {
    const [catalogOpen, setCatalogOpen] = useState(false);
    const [accountOpen, setAccountOpen] = useState(false);
    const overlayOpen = catalogOpen || accountOpen;

    useImperativeHandle(
      ref,
      () => ({
        closeOverlays: () => {
          if (!catalogOpen && !accountOpen) return false;
          setCatalogOpen(false);
          setAccountOpen(false);
          return true;
        },
      }),
      [catalogOpen, accountOpen],
    );

    const navigate = (screen: DestScreen) => {
      setCatalogOpen(false);
      setAccountOpen(false);
      onNavigate(screen);
    };

    const openSettings = () => {
      const settings = getDestination("settings");
      if (settings) navigate(settings.screen);
    };

    // Fix round 1, MINOR: while an overlay is open, everything behind it is
    // either fully covered (the catalog) or unreachable-by-touch (the
    // account menu's full-screen backdrop already intercepts taps) — hide it
    // from assistive tech too, so a screen reader doesn't navigate into
    // background content it can't actually act on.
    const backgroundA11y = overlayOpen
      ? ("no-hide-descendants" as const)
      : ("auto" as const);

    return (
      <View style={styles.container} testID="app-chrome">
        <View style={styles.topBar} importantForAccessibility={backgroundA11y}>
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

        <View
          style={styles.content}
          testID="chrome-content"
          importantForAccessibility={backgroundA11y}
        >
          {children}
        </View>

        <ChromeTabBar
          screenName={screenName}
          onNavigate={navigate}
          importantForAccessibility={backgroundA11y}
        />

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
  },
);

export default AppChrome;

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
});
