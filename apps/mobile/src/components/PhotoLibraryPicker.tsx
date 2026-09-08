import React, { useCallback, useEffect, useState } from "react";
import {
  View,
  Text,
  Image,
  TouchableOpacity,
  StyleSheet,
  FlatList,
  useWindowDimensions,
} from "react-native";
import * as MediaLibrary from "expo-media-library";
import {
  getAssetsAsync,
  getAssetInfoAsync,
  MediaType,
  SortBy,
} from "expo-media-library/legacy";

/**
 * A minimal in-app "choose from your photos" grid.
 *
 * Why not a system picker: no image-picker native module is in the installed
 * binary, and adding one can't ship as an OTA (runtimeVersion policy is
 * appVersion — a native module means a new build). expo-media-library IS in
 * the binary (RecordScreen saves videos with it), and its read side gives us
 * the camera roll's recent photos directly, so this is deliverable today.
 *
 * Honest by construction, same house style as the capture screens: a
 * permission denial shows an inline message with a grant retry (and a path
 * to Settings once the OS won't ask again), an empty roll says so, and a
 * failed read surfaces recovery rather than an empty grid.
 */

const PAGE = 60;
const COLUMNS = 3;
const GAP = 3;

export interface PhotoLibraryPickerProps {
  /** The user tapped a photo — `uri` is a file uri the caller can copy
   *  (`localUri` when the library gives one, else the asset uri). */
  onPick: (uri: string) => void;
  /** Leave the picker without choosing. */
  onCancel: () => void;
}

type Thumb = { id: string; uri: string };

export default function PhotoLibraryPicker({
  onPick,
  onCancel,
}: PhotoLibraryPickerProps) {
  const [perm, requestPerm] = MediaLibrary.usePermissions();
  const [thumbs, setThumbs] = useState<Thumb[]>([]);
  const [cursor, setCursor] = useState<string | undefined>(undefined);
  const [hasMore, setHasMore] = useState(false);
  const [loading, setLoading] = useState(false);
  const [resolving, setResolving] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { width } = useWindowDimensions();
  const cell = Math.floor((width - GAP * (COLUMNS - 1)) / COLUMNS);

  const loadPage = useCallback(
    async (after?: string) => {
      setLoading(true);
      setError(null);
      try {
        const page = await getAssetsAsync({
          first: PAGE,
          after,
          mediaType: [MediaType.photo],
          sortBy: [[SortBy.creationTime, false]],
        });
        const next = page.assets.map((a) => ({ id: a.id, uri: a.uri }));
        setThumbs((prev) => (after ? [...prev, ...next] : next));
        setCursor(page.endCursor);
        setHasMore(page.hasNextPage);
      } catch {
        setError("Couldn’t read your photos. Please try again.");
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  const granted = perm?.granted === true;
  useEffect(() => {
    if (granted) void loadPage();
  }, [granted, loadPage]);

  const handlePick = useCallback(
    async (t: Thumb) => {
      if (resolving) return;
      setResolving(t.id);
      setError(null);
      try {
        // iOS assets are `ph://` — only `localUri` is a copyable file. Android
        // usually hands back a file uri already; fall back to it either way.
        const info = await getAssetInfoAsync(t.id);
        onPick(info.localUri || t.uri);
      } catch {
        setError("Couldn’t open that photo. Please try another one.");
      } finally {
        setResolving(null);
      }
    },
    [onPick, resolving],
  );

  if (!granted) {
    const canAskAgain = perm?.canAskAgain !== false;
    return (
      <View style={styles.centered} testID="photo-library-permission-gate">
        <Text style={styles.noteTitle}>Photo access needed</Text>
        <Text style={styles.noteText}>
          {canAskAgain
            ? "MindShift needs access to your photos to pick a profile picture."
            : "Photo access was denied. Enable it for MindShift in your device Settings, then come back here to pick a profile picture."}
        </Text>
        {canAskAgain ? (
          <TouchableOpacity
            testID="photo-library-grant"
            style={styles.primaryButton}
            onPress={() => void requestPerm()}
          >
            <Text style={styles.primaryButtonText}>Grant access</Text>
          </TouchableOpacity>
        ) : null}
        <TouchableOpacity
          testID="photo-library-cancel"
          style={styles.secondaryButton}
          onPress={onCancel}
        >
          <Text style={styles.secondaryButtonText}>Back</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.flex} testID="photo-library-picker">
      {error ? (
        <Text style={styles.errorText} testID="photo-library-error">
          {error}
        </Text>
      ) : null}
      {!loading && !error && thumbs.length === 0 ? (
        <View style={styles.centered} testID="photo-library-empty">
          <Text style={styles.noteText}>No photos on this device yet.</Text>
        </View>
      ) : null}
      <FlatList
        data={thumbs}
        keyExtractor={(t) => t.id}
        numColumns={COLUMNS}
        testID="photo-library-grid"
        columnWrapperStyle={{ gap: GAP }}
        contentContainerStyle={{ gap: GAP }}
        onEndReachedThreshold={0.5}
        onEndReached={() => {
          if (hasMore && !loading) void loadPage(cursor);
        }}
        renderItem={({ item }) => (
          <TouchableOpacity
            testID={`photo-library-thumb-${item.id}`}
            onPress={() => void handlePick(item)}
            disabled={resolving !== null}
            style={{ width: cell, height: cell }}
          >
            <Image
              source={{ uri: item.uri }}
              style={[styles.thumb, resolving === item.id && styles.thumbBusy]}
            />
          </TouchableOpacity>
        )}
        ListFooterComponent={
          loading ? (
            <Text style={styles.footerText} testID="photo-library-loading">
              Loading…
            </Text>
          ) : null
        }
      />
    </View>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1, backgroundColor: "#000000" },
  centered: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: 24,
    gap: 12,
  },
  noteTitle: { color: "#FFFFFF", fontSize: 18, fontWeight: "700" },
  noteText: { color: "#D1D5DB", fontSize: 15, textAlign: "center", lineHeight: 21 },
  errorText: {
    color: "#FCA5A5",
    fontSize: 14,
    textAlign: "center",
    paddingVertical: 8,
    paddingHorizontal: 16,
  },
  footerText: { color: "#9CA3AF", textAlign: "center", paddingVertical: 12 },
  thumb: { width: "100%", height: "100%", backgroundColor: "#1F2937" },
  thumbBusy: { opacity: 0.5 },
  primaryButton: {
    backgroundColor: "#4A90D9",
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 10,
  },
  primaryButtonText: { color: "#FFFFFF", fontWeight: "700", fontSize: 16 },
  secondaryButton: {
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: "#6B7280",
  },
  secondaryButtonText: { color: "#E5E7EB", fontWeight: "600", fontSize: 16 },
});
