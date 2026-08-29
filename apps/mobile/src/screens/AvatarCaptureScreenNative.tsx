import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Image,
  BackHandler,
} from "react-native";
import { CameraView, useCameraPermissions } from "expo-camera";
import { useAvatarStore } from "../store/avatarStore";
import PhotoLibraryPicker from "../components/PhotoLibraryPicker";
import type {
  AvatarCaptureDeps,
  AvatarCaptureScreenProps,
  PhotoSource,
} from "./AvatarCaptureScreen";

// House colors — matches RecordScreenNative.tsx.
const PRIMARY = "#4A90D9";
const INK = "#1F2937";
const MUTED = "#6B7280";

/** Monotonic per-process counter backing defaultPersistPhoto's cache-busting
 *  suffix (see that function's doc comment) — guarantees two sequential
 *  captures never produce the same returned uri even if they land in the
 *  same Date.now() millisecond. */
let persistSeq = 0;

/** The real filesystem write behind AvatarCaptureDeps.persistPhoto — moves
 *  the captured photo into the app's document directory under a fixed
 *  filename, so a retake always overwrites the SAME on-disk file rather
 *  than accumulating stale files. Exercised for real on-device; tests
 *  supply `deps` instead of mocking expo-file-system's Directory/File API
 *  (see AvatarCaptureDeps's doc comment for why).
 *
 *  N7 fix round 1 (CRITICAL 1): the RETURNED uri must still change on every
 *  call even though the underlying path never does — the old code returned
 *  `dest.uri` verbatim, so a retake called setPhoto() with a string
 *  identical to the current one. zustand's set({ uri }) with an unchanged
 *  value never notifies subscribers, and even if it did, <Image
 *  source={{uri}}/> is cached by exact uri string on both Fresco (Android)
 *  and iOS — so the same path would keep serving the stale bitmap either
 *  way. Appending a monotonic cache-busting query suffix fixes both at
 *  once, for a one-line diff, without touching the on-disk overwrite
 *  behavior at all. */
export async function defaultPersistPhoto(
  capturedUri: string,
  source: PhotoSource = "camera",
): Promise<string> {
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const { Directory, File, Paths } = require("expo-file-system");
  const dir = new Directory(Paths.document, "avatar");
  if (!dir.exists) dir.create({ intermediates: true });
  const dest = new File(dir, "profile.jpg");
  if (dest.exists) dest.delete();
  // A camera capture is our own temp file — move it. A library photo belongs
  // to the user's camera roll — COPY it; moving would delete their original.
  if (source === "library") new File(capturedUri).copy(dest);
  else new File(capturedUri).moveSync(dest);
  persistSeq += 1;
  return `${dest.uri}?v=${Date.now()}-${persistSeq}`;
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
  const [capturedFrom, setCapturedFrom] = useState<PhotoSource>("camera");
  // "Choose from your photos" — an in-app roll grid (see PhotoLibraryPicker),
  // reachable from BOTH the live camera and the camera-permission gate: the
  // owner asked for it from a dark room where the camera was useless.
  const [browsing, setBrowsing] = useState(false);
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
        setCapturedFrom("camera");
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
    // A library pick goes back to the grid; a camera shot back to the camera.
    setBrowsing(capturedFrom === "library");
    setCaptured(null);
    setError(null);
  }, [capturedFrom]);

  // N7 fix round 1 (IMPORTANT 5): the branch's back contract is overlay ->
  // pop -> double-back-exit, but the "Use this photo? / Retake" preview is
  // an in-screen sub-state that chain doesn't know about — without this,
  // backTarget sees {name:"avatar-capture"} and pops straight to returnTo,
  // silently dropping the just-captured photo. Only subscribe while a photo
  // is actually captured, so this consumes the press (and resets to the
  // live camera, same as tapping Retake) exactly when there's an in-progress
  // capture to protect; with no capture, no listener is registered here at
  // all, so the outer chain's own back handling (App.tsx's
  // useAndroidBackHandler) applies unchanged. RN's BackHandler calls the
  // most-recently-registered listener first, so this local subscription —
  // mounted after the app-level one — naturally gets first refusal.
  useEffect(() => {
    if (captured === null) return;
    const subscription = BackHandler.addEventListener(
      "hardwareBackPress",
      () => {
        handleRetake();
        return true;
      },
    );
    return () => subscription.remove();
  }, [captured, handleRetake]);

  const handleUse = useCallback(async () => {
    if (!captured) return;
    setSaving(true);
    setError(null);
    try {
      const finalUri = await persistPhoto(captured, capturedFrom);
      useAvatarStore.getState().setPhoto(finalUri);
      onSaved();
    } catch {
      setError("Couldn’t save your photo. Please try again.");
    } finally {
      setSaving(false);
    }
  }, [captured, capturedFrom, persistPhoto, onSaved]);

  const handlePicked = useCallback((uri: string) => {
    setCapturedFrom("library");
    setCaptured(uri);
    setBrowsing(false);
    setError(null);
  }, []);

  // --- Library: pick an existing photo (no camera needed).
  if (browsing && !captured) {
    return (
      <View style={styles.flex}>
        <Header onBack={() => setBrowsing(false)} title="Choose a photo" />
        <PhotoLibraryPicker
          onPick={handlePicked}
          onCancel={() => setBrowsing(false)}
        />
      </View>
    );
  }

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
            testID="avatar-choose-library"
            style={styles.secondaryButton}
            onPress={() => setBrowsing(true)}
          >
            <Text style={styles.secondaryButtonText}>Choose from your photos</Text>
          </TouchableOpacity>
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
            <Text style={styles.secondaryButtonText}>
              {capturedFrom === "library" ? "Choose another" : "Retake"}
            </Text>
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
            <TouchableOpacity
              testID="avatar-choose-library"
              style={styles.libraryButton}
              onPress={() => setBrowsing(true)}
            >
              <Text style={styles.libraryButtonText}>Choose from your photos</Text>
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
  controls: { alignItems: "center", gap: 14 },
  libraryButton: {
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 999,
    backgroundColor: "rgba(0,0,0,0.45)",
  },
  libraryButtonText: { color: "#FFFFFF", fontSize: 14, fontWeight: "600" },
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
