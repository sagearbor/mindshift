import React, { useCallback, useRef, useState } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Image,
} from "react-native";
import { CameraView, useCameraPermissions } from "expo-camera";
import { useAvatarStore } from "../store/avatarStore";
import type {
  AvatarCaptureDeps,
  AvatarCaptureScreenProps,
} from "./AvatarCaptureScreen";

// House colors — matches RecordScreenNative.tsx.
const PRIMARY = "#4A90D9";
const INK = "#1F2937";
const MUTED = "#6B7280";

/** The real filesystem write behind AvatarCaptureDeps.persistPhoto — moves
 *  the captured photo into the app's document directory under a fixed
 *  filename, so a retake always overwrites rather than accumulating stale
 *  files. Exercised for real on-device; tests supply `deps` instead of
 *  mocking expo-file-system's Directory/File API (see AvatarCaptureDeps's
 *  doc comment for why). */
async function defaultPersistPhoto(capturedUri: string): Promise<string> {
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const { Directory, File, Paths } = require("expo-file-system");
  const dir = new Directory(Paths.document, "avatar");
  if (!dir.exists) dir.create({ intermediates: true });
  const dest = new File(dir, "profile.jpg");
  if (dest.exists) dest.delete();
  new File(capturedUri).moveSync(dest);
  return dest.uri;
}

function defaultDeps(): AvatarCaptureDeps {
  return { persistPhoto: defaultPersistPhoto };
}

/**
 * Selfie capture: front camera -> preview -> Use/Retake (Task N6 of P3-10).
 * Honest by construction, same house style as RecordScreenNative.tsx: a
 * permission denial always shows an inline message with a grant retry
 * (never a black screen), and a save failure surfaces recovery instead of
 * silently dropping the shot.
 */
export default function AvatarCaptureScreen({
  onBack,
  onSaved,
  deps,
}: AvatarCaptureScreenProps) {
  const { persistPhoto } = deps ?? defaultDeps();
  const [camPerm, requestCamPerm] = useCameraPermissions();
  const cameraRef = useRef<CameraView>(null);

  const [captured, setCaptured] = useState<string | null>(null);
  const [capturing, setCapturing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleCapture = useCallback(async () => {
    if (capturing) return;
    setCapturing(true);
    setError(null);
    try {
      const photo = await cameraRef.current?.takePictureAsync({
        quality: 0.8,
      });
      if (photo?.uri) {
        setCaptured(photo.uri);
      } else {
        setError("Couldn’t take that photo. Please try again.");
      }
    } catch {
      setError("Couldn’t take that photo. Please try again.");
    } finally {
      setCapturing(false);
    }
  }, [capturing]);

  const handleRetake = useCallback(() => {
    setCaptured(null);
    setError(null);
  }, []);

  const handleUse = useCallback(async () => {
    if (!captured) return;
    setSaving(true);
    setError(null);
    try {
      const finalUri = await persistPhoto(captured);
      useAvatarStore.getState().setPhoto(finalUri);
      onSaved();
    } catch {
      setError("Couldn’t save your photo. Please try again.");
    } finally {
      setSaving(false);
    }
  }, [captured, persistPhoto, onSaved]);

  // --- Permission gate: honest, with a grant retry, and a path to Settings
  // once the OS says it won't ask again. Never a black screen.
  if (!camPerm?.granted) {
    const canAskAgain = camPerm?.canAskAgain !== false;
    return (
      <View style={styles.flex}>
        <Header onBack={onBack} />
        <View style={styles.centered} testID="avatar-permission-gate">
          <Text style={styles.noteTitle}>Camera access needed</Text>
          <Text style={styles.noteText}>
            {canAskAgain
              ? "MindShift needs your camera to take a profile photo."
              : "Camera access was denied. Enable it for MindShift in your device Settings, then come back here to take a profile photo."}
          </Text>
          {canAskAgain ? (
            <TouchableOpacity
              testID="avatar-grant-camera"
              style={styles.primaryButton}
              onPress={() => void requestCamPerm()}
            >
              <Text style={styles.primaryButtonText}>Grant access</Text>
            </TouchableOpacity>
          ) : null}
          <TouchableOpacity
            testID="avatar-capture-back"
            style={styles.secondaryButton}
            onPress={onBack}
          >
            <Text style={styles.secondaryButtonText}>Back</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  // --- Preview: the just-captured shot, with Use / Retake.
  if (captured) {
    return (
      <View style={styles.flex}>
        <Header onBack={onBack} title="Use this photo?" />
        <View style={styles.previewWrap}>
          <Image
            testID="avatar-preview-image"
            source={{ uri: captured }}
            style={styles.previewImage}
          />
        </View>
        {error ? (
          <Text style={styles.errorText} testID="avatar-capture-error">
            {error}
          </Text>
        ) : null}
        <View style={styles.previewControls}>
          <TouchableOpacity
            testID="avatar-retake-button"
            style={styles.secondaryButton}
            onPress={handleRetake}
            disabled={saving}
          >
            <Text style={styles.secondaryButtonText}>Retake</Text>
          </TouchableOpacity>
          <TouchableOpacity
            testID="avatar-use-button"
            style={styles.primaryButton}
            onPress={() => void handleUse()}
            disabled={saving}
          >
            <Text style={styles.primaryButtonText}>
              {saving ? "Saving…" : "Use photo"}
            </Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  // --- Live front-camera preview + shutter.
  return (
    <View style={styles.flex}>
      <Header onBack={onBack} />
      <CameraView
        ref={cameraRef}
        style={styles.camera}
        facing="front"
        testID="avatar-camera-view"
      >
        <View style={styles.overlay} pointerEvents="box-none">
          {error ? (
            <Text style={styles.errorTextOverlay} testID="avatar-capture-error">
              {error}
            </Text>
          ) : null}
          <View style={styles.controls} pointerEvents="box-none">
            <TouchableOpacity
              testID="avatar-shutter-button"
              style={styles.shutterButton}
              onPress={() => void handleCapture()}
              disabled={capturing}
            >
              <View style={styles.shutterIcon} />
            </TouchableOpacity>
          </View>
        </View>
      </CameraView>
    </View>
  );
}

function Header({
  onBack,
  title = "Profile photo",
}: {
  onBack: () => void;
  title?: string;
}) {
  return (
    <View style={styles.header}>
      <TouchableOpacity testID="avatar-back" onPress={onBack}>
        <Text style={styles.backText}>‹ Back</Text>
      </TouchableOpacity>
      <Text style={styles.headerTitle}>{title}</Text>
      <View style={styles.headerSpacer} />
    </View>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1, backgroundColor: "#000000" },
  header: {
    flexDirection: "row",
    alignItems: "center",
    paddingTop: 56,
    paddingBottom: 12,
    paddingHorizontal: 16,
    backgroundColor: "#FFFFFF",
    borderBottomWidth: 1,
    borderBottomColor: "#E5E7EB",
  },
  backText: { fontSize: 16, color: PRIMARY, fontWeight: "600", width: 64 },
  headerTitle: {
    flex: 1,
    textAlign: "center",
    fontSize: 17,
    fontWeight: "700",
    color: INK,
  },
  headerSpacer: { width: 64 },
  centered: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    padding: 24,
    backgroundColor: "#F9FAFB",
  },
  noteTitle: { fontSize: 18, fontWeight: "700", color: INK, marginBottom: 10 },
  noteText: {
    fontSize: 14,
    lineHeight: 20,
    color: MUTED,
    textAlign: "center",
    marginBottom: 18,
    maxWidth: 340,
  },
  primaryButton: {
    backgroundColor: PRIMARY,
    paddingVertical: 12,
    paddingHorizontal: 28,
    borderRadius: 10,
    marginBottom: 10,
    alignItems: "center",
  },
  primaryButtonText: { color: "#FFFFFF", fontSize: 15, fontWeight: "600" },
  secondaryButton: {
    paddingVertical: 12,
    paddingHorizontal: 28,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: "#D1D5DB",
    alignItems: "center",
  },
  secondaryButtonText: { color: MUTED, fontSize: 15, fontWeight: "600" },
  camera: { flex: 1 },
  overlay: {
    flex: 1,
    justifyContent: "flex-end",
    paddingVertical: 24,
    paddingHorizontal: 16,
  },
  controls: { alignItems: "center" },
  shutterButton: {
    width: 76,
    height: 76,
    borderRadius: 38,
    borderWidth: 4,
    borderColor: "#FFFFFF",
    alignItems: "center",
    justifyContent: "center",
  },
  shutterIcon: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: "#FFFFFF",
  },
  errorTextOverlay: {
    alignSelf: "center",
    color: "#FCA5A5",
    fontSize: 13,
    fontWeight: "600",
    textAlign: "center",
    marginBottom: 12,
  },
  errorText: {
    color: "#DC2626",
    fontSize: 13,
    fontWeight: "600",
    textAlign: "center",
    marginTop: 12,
  },
  previewWrap: { flex: 1, backgroundColor: "#000000" },
  previewImage: { flex: 1, resizeMode: "cover" },
  previewControls: {
    flexDirection: "row",
    gap: 12,
    padding: 16,
    backgroundColor: "#FFFFFF",
  },
});
