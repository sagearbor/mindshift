import React from "react";
import { View, Text, TouchableOpacity, StyleSheet, Platform } from "react-native";

/** Everything AvatarCaptureScreenNative's own filesystem write is behind —
 *  the same DI seam VoiceTrainingFlow.ts uses for its expo-file-system write
 *  (see that file's `saveWav`). Declared here (the shared/public contract)
 *  rather than in the native-only file so tests can pass it through this
 *  gate without reaching into native-only internals. Production code never
 *  supplies this — AvatarCaptureScreenNative falls back to its real
 *  filesystem implementation when `deps` is omitted. */
export interface AvatarCaptureDeps {
  /** Move the just-captured photo into permanent local storage (the app's
   *  document directory), overwriting any previously saved one — a retake
   *  always overwrites the same fixed filename rather than accumulating
   *  stale files. Returns the final persisted uri to save in avatarStore. */
  persistPhoto: (capturedUri: string, source?: PhotoSource) => Promise<string>;
}

/** Where the photo came from — decides move vs copy in persistPhoto: a
 *  camera capture is a temp file we MOVE; a photo chosen from the user's
 *  library is THEIRS and is COPIED, never taken out of their camera roll. */
export type PhotoSource = "camera" | "library";

export interface AvatarCaptureScreenProps {
  /** Cancel — back to wherever the avatar menu / Settings row launched from,
   *  without touching the stored photo. */
  onBack: () => void;
  /** The captured photo has been persisted and avatarStore updated — return
   *  to the launching screen (same shape as onBack, kept separate so a
   *  future caller can tell "saved" from "cancelled" apart if it ever needs
   *  to). */
  onSaved: () => void;
  /** Test-only DI seam (see AvatarCaptureDeps) — always omitted in
   *  production. Web ignores it; it's meaningless there. */
  deps?: AvatarCaptureDeps;
}

/**
 * Platform gate for the selfie-capture flow (Task N6 of P3-10).
 *
 * expo-camera's `CameraView` technically renders on web (getUserMedia), but
 * `takePictureAsync` there returns a base64 string instead of a file uri
 * ("file system URLs are not supported in the browser" — expo-camera's own
 * doc comment), and expo-file-system's modern File/Directory API — which
 * AvatarCaptureScreenNative uses to move the capture into permanent app
 * storage — is a set of empty stubs on web (ExpoFileSystem.web.d.ts has no
 * real `exists`/`create`/`moveSync`). Wiring either up would mean silently
 * degrading (or crashing) rather than the honest "isn't available here" this
 * app uses everywhere else a native capability doesn't reach the web build
 * (see RecordScreen.tsx's identical split, there for expo-media-library). A
 * `<input type="file">` fallback was considered and is a plausible follow-up,
 * but it isn't trivial given no image-picker/manipulator dependency exists
 * in this app yet — ledgered rather than half-built here.
 */
export default function AvatarCaptureScreen(props: AvatarCaptureScreenProps) {
  if (Platform.OS === "web") {
    return (
      <View style={styles.center} testID="avatar-capture-web-note">
        <Text style={styles.note}>
          Taking a profile photo isn't available in the web app yet — open
          MindShift on your phone to set one.
        </Text>
        <TouchableOpacity
          style={styles.backButton}
          onPress={props.onBack}
          testID="avatar-capture-web-back"
        >
          <Text style={styles.backText}>Back</Text>
        </TouchableOpacity>
      </View>
    );
  }
  // Lazy, native-only module resolution — mirrors RecordScreen.tsx exactly.
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const Native = require("./AvatarCaptureScreenNative").default;
  return <Native {...props} />;
}

const styles = StyleSheet.create({
  center: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    padding: 32,
    backgroundColor: "#F9FAFB",
  },
  note: {
    fontSize: 15,
    color: "#6B7280",
    textAlign: "center",
    marginBottom: 20,
  },
  backButton: {
    borderWidth: 1,
    borderColor: "#D1D5DB",
    borderRadius: 10,
    paddingVertical: 12,
    paddingHorizontal: 24,
  },
  backText: {
    color: "#1F2937",
    fontWeight: "600",
  },
});
