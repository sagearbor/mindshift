import { useAuthStore } from "../store/authStore";

/**
 * Who owns recordings written to this device's document directory right now.
 *
 * The recorder's files outlive a Firebase sign-out, so ownership is what keeps
 * one account's audio from being offered to the next one on a shared phone.
 * Keyed on UID, never email: a "Continue as guest" session is a real Firebase
 * (anonymous) account with a real uid and `email === null`, and two different
 * guests are two different uids.
 *
 * Read as a thunk (never captured once) so a store built before a sign-out
 * sees the account that is signed in at scan time, not the one that was.
 */
export function currentRecordingOwnerUid(): string | null {
  try {
    return useAuthStore.getState().user?.uid ?? null;
  } catch {
    // Auth not initialized (or torn down): unknown owner, which claims
    // nothing — the fail-closed side.
    return null;
  }
}
