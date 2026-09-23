import { create } from "zustand";

/**
 * A one-shot request to open the guest banner's "Create account" form from
 * somewhere else in the app.
 *
 * Why a store for one boolean: the form's open/closed state lives inside
 * GuestBanner, and the place that most needs to open it — the confirm shown
 * when a guest is about to log out — is a different subtree entirely. That
 * confirm is the single best moment to offer an account: it is the only screen
 * where the user has been told, in those words, that they are about to put
 * their recordings out of reach. Telling them to go and find a button
 * elsewhere, having just cancelled a dialog, is the version of this that
 * nobody follows.
 *
 * One-shot on purpose: GuestBanner consumes the request and clears it, so a
 * later re-render cannot re-open a form the user has since closed.
 */
interface GuestUpgradeState {
  requested: boolean;
  /** Ask the guest banner to expand its account-creation form. */
  request: () => void;
  /** Called by the banner once it has acted on the request. */
  consume: () => void;
}

export const useGuestUpgradeStore = create<GuestUpgradeState>((set) => ({
  requested: false,
  request: () => set({ requested: true }),
  consume: () => set({ requested: false }),
}));
