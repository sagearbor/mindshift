import React, { useCallback, useState } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ActivityIndicator,
} from "react-native";

import { useOtaStatus, restartToApplyUpdate } from "../utils/otaUpdate";

/**
 * A subtle, honest "Update ready — restart to apply" bar.
 *
 * It appears ONLY once a newer OTA update has finished downloading and is
 * pending (`isUpdatePending`) — never mid-download, never on the embedded/first
 * launch. We deliberately do NOT hot-swap the JS under a running session; the
 * user chooses when to relaunch, so nothing changes out from under them.
 * Renders nothing (no layout footprint) in the common case where no update is
 * waiting — including on web, where OTA isn't in play.
 */
export default function UpdateBanner() {
  const { isUpdatePending } = useOtaStatus();
  const [restarting, setRestarting] = useState(false);

  const onRestart = useCallback(() => {
    setRestarting(true);
    // If the relaunch fails, drop back to the button so the user can retry
    // rather than being stuck on a spinner.
    restartToApplyUpdate().catch(() => setRestarting(false));
  }, []);

  if (!isUpdatePending) return null;

  return (
    <View style={styles.banner} testID="update-banner">
      <View style={styles.textWrap}>
        <Text style={styles.title}>Update ready</Text>
        <Text style={styles.sub}>Restart to apply the latest version.</Text>
      </View>
      <TouchableOpacity
        testID="update-banner-restart"
        accessibilityRole="button"
        accessibilityLabel="Restart to apply update"
        style={styles.button}
        onPress={onRestart}
        disabled={restarting}
      >
        {restarting ? (
          <ActivityIndicator size="small" color="#FFFFFF" />
        ) : (
          <Text style={styles.buttonText}>Restart</Text>
        )}
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  banner: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    backgroundColor: "#EEF2FF",
    borderBottomWidth: 1,
    borderBottomColor: "#C7D2FE",
    paddingHorizontal: 16,
    paddingVertical: 10,
  },
  textWrap: {
    flex: 1,
    paddingRight: 12,
  },
  title: {
    fontSize: 14,
    fontWeight: "700",
    color: "#3730A3",
  },
  sub: {
    marginTop: 1,
    fontSize: 12.5,
    color: "#4C51BF",
  },
  button: {
    minHeight: 36,
    minWidth: 76,
    paddingHorizontal: 14,
    borderRadius: 10,
    backgroundColor: "#4A90D9",
    alignItems: "center",
    justifyContent: "center",
  },
  buttonText: {
    color: "#FFFFFF",
    fontSize: 14,
    fontWeight: "600",
  },
});
