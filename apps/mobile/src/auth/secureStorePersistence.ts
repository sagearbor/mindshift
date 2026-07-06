import * as SecureStore from "expo-secure-store";

/**
 * An AsyncStorage-shaped adapter (getItem/setItem/removeItem) backed by
 * expo-secure-store, for Firebase Auth's `getReactNativePersistence`. Firebase
 * persists the signed-in user (uid, tokens, refresh token) here so a session
 * survives app restarts — on the OS keystore/keychain rather than plaintext
 * AsyncStorage, which matters for a therapy-adjacent app.
 *
 * SecureStore keys may only contain [A-Za-z0-9._-]; Firebase's keys contain
 * ':' and '[DEFAULT]'. `encodeKey` escapes every other character (and the
 * escape char itself) to a `_uXXXX` form, so the mapping is total and
 * collision-free (deterministic, no two distinct keys collide).
 */
function encodeKey(key: string): string {
  return key.replace(
    /[^A-Za-z0-9.\-]/g,
    (c) => "_u" + c.charCodeAt(0).toString(16).padStart(4, "0"),
  );
}

export const secureStorePersistence = {
  async getItem(key: string): Promise<string | null> {
    return SecureStore.getItemAsync(encodeKey(key));
  },
  async setItem(key: string, value: string): Promise<void> {
    await SecureStore.setItemAsync(encodeKey(key), value);
  },
  async removeItem(key: string): Promise<void> {
    await SecureStore.deleteItemAsync(encodeKey(key));
  },
};
