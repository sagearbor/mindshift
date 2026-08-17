import React, { useEffect, useState } from "react";
import { View, Text, Image, StyleSheet } from "react-native";

/** The slice of the signed-in user Avatar needs — matches authStore's
 *  AuthUser shape without importing it (keeps this component store-free). */
export interface AvatarUser {
  email: string | null;
  displayName: string | null;
}

interface AvatarProps {
  user: AvatarUser | null;
  /** A captured selfie's local uri (Task N6's slot). Undefined/null today —
   *  N6 hasn't landed yet — so Avatar always falls back to the initial
   *  below; once N6 ships, passing a uri here is the only change needed. */
  photoUri?: string | null;
  size?: number;
  testID?: string;
}

/** The account's initial — display name first, then email, uppercased.
 *  Never blank: an account with neither reads "?" rather than an empty
 *  circle. */
export function accountInitial(user: AvatarUser | null): string {
  const source = (user?.displayName || user?.email || "").trim();
  return source ? source.charAt(0).toUpperCase() : "?";
}

/** The top-bar avatar slot (Task N3): a photo once N6 lands, else a filled
 *  circle with the account's initial — the owner's directive that nothing in
 *  this nav reads as a bare/blank placeholder. */
export default function Avatar({
  user,
  photoUri,
  size = 36,
  testID = "avatar",
}: AvatarProps) {
  const dimensions = { width: size, height: size, borderRadius: size / 2 };

  // N7 fix round 1 (IMPORTANT 4): a persisted absolute file:// uri can go
  // stale (e.g. an iOS app-container path change across reinstalls/
  // updates), pointing at nothing. Branching purely on photoUri truthiness
  // rendered an <Image> that silently failed — a blank circle instead of
  // this component's otherwise-guaranteed honest fallback. Track load
  // failure locally, reset whenever photoUri itself changes so a fresh uri
  // (a new capture) always gets its own chance to load.
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    setFailed(false);
  }, [photoUri]);

  if (photoUri && !failed) {
    return (
      <Image
        testID={`${testID}-photo`}
        source={{ uri: photoUri }}
        style={[styles.circle, dimensions]}
        onError={() => setFailed(true)}
      />
    );
  }

  return (
    <View
      testID={`${testID}-initial`}
      style={[styles.circle, styles.initialCircle, dimensions]}
    >
      <Text style={[styles.initialText, { fontSize: size * 0.42 }]}>
        {accountInitial(user)}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  circle: {
    alignItems: "center",
    justifyContent: "center",
    overflow: "hidden",
  },
  initialCircle: {
    backgroundColor: "#4A90D9",
  },
  initialText: {
    color: "#FFFFFF",
    fontWeight: "700",
  },
});
