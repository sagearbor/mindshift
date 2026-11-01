/**
 * In-app purchases via RevenueCat (react-native-purchases).
 *
 * Honest by construction: with no platform SDK key in the build env
 * (EXPO_PUBLIC_REVENUECAT_IOS_KEY / EXPO_PUBLIC_REVENUECAT_ANDROID_KEY) or on
 * web, purchases are DISABLED and the UI keeps saying "Coming soon". Nothing
 * here ever fakes a purchase. RevenueCat's app user id is the same Firebase uid
 * the server keys plans on (server/revenuecat.py reads it from the webhook);
 * guests (anonymous uids) never log in, so they cannot buy.
 *
 * Offering `default` carries one package per paid level whose identifier is the
 * level itself: basic | plus | pro | max (products mindshift_<level>_monthly).
 */
import { Platform } from "react-native";

export const OFFERING_ID = "default";

type Sdk = typeof import("react-native-purchases").default;

let sdk: Sdk | null = null;
let configuredKey: string | null = null;
let loggedInAs: string | null = null;

export function sdkKey(): string | null {
  const key =
    Platform.OS === "ios"
      ? process.env.EXPO_PUBLIC_REVENUECAT_IOS_KEY
      : Platform.OS === "android"
        ? process.env.EXPO_PUBLIC_REVENUECAT_ANDROID_KEY
        : undefined;
  return key && key.trim() ? key.trim() : null;
}

/** True only when this build has a RevenueCat key for this platform. */
export function purchasesEnabled(): boolean {
  return sdkKey() !== null;
}

function load(): Sdk {
  if (!sdk) {
    // Lazy so web, tests and key-less builds never touch the native module.
    sdk = (require("react-native-purchases") as { default: Sdk }).default;
  }
  return sdk;
}

export type PurchaseErrorKind = "disabled" | "guest" | "unavailable" | "cancelled" | "failed";

export class PurchaseError extends Error {
  kind: PurchaseErrorKind;
  constructor(kind: PurchaseErrorKind, message: string) {
    super(message);
    this.kind = kind;
  }
}

/** Configure once, then log in as `uid`. Throws PurchaseError when not allowed. */
export async function ensureReady(user: { uid: string; isAnonymous: boolean } | null): Promise<Sdk> {
  const key = sdkKey();
  if (!key) throw new PurchaseError("disabled", "Paid plans are coming soon.");
  if (!user || user.isAnonymous) {
    throw new PurchaseError("guest", "Create an account (or sign in) to subscribe.");
  }
  const sdkRef = load();
  if (configuredKey !== key) {
    sdkRef.configure({ apiKey: key });
    configuredKey = key;
    loggedInAs = null;
  }
  if (loggedInAs !== user.uid) {
    await sdkRef.logIn(user.uid);
    loggedInAs = user.uid;
  }
  return sdkRef;
}

/** Sign RevenueCat out (call on account sign-out). No-op if never logged in. */
export async function resetPurchases(): Promise<void> {
  if (!sdk || !loggedInAs) return;
  loggedInAs = null;
  try {
    await sdk.logOut();
  } catch {
    // already anonymous
  }
}

export function friendlyError(e: unknown): PurchaseError {
  if (e instanceof PurchaseError) return e;
  const err = e as { userCancelled?: boolean; code?: string | number; message?: string };
  if (err?.userCancelled) return new PurchaseError("cancelled", "Purchase cancelled.");
  const code = String(err?.code ?? "");
  if (code === "2" || /NETWORK/i.test(code)) {
    return new PurchaseError("failed", "Couldn't reach the store. Check your connection and try again.");
  }
  if (code === "3" || /NOT_ALLOWED|PURCHASE_NOT_ALLOWED/i.test(code)) {
    return new PurchaseError("failed", "Purchases are turned off on this device.");
  }
  if (code === "5" || /UNAVAILABLE|PRODUCT_NOT_AVAILABLE/i.test(code)) {
    return new PurchaseError("unavailable", "That plan isn't available in the store right now. Please try again later.");
  }
  if (code === "7" || /ALREADY_PURCHASED/i.test(code)) {
    return new PurchaseError("failed", "You already own this plan. Try Restore purchases.");
  }
  return new PurchaseError("failed", "Something went wrong with the purchase. You were not charged. Please try again.");
}

/** Buy the `default` offering's package whose identifier is `level`. */
export async function purchaseLevel(
  level: string,
  user: { uid: string; isAnonymous: boolean } | null,
): Promise<"purchased"> {
  try {
    const rc = await ensureReady(user);
    const offerings = await rc.getOfferings();
    const offering = offerings.all?.[OFFERING_ID] ?? offerings.current;
    const pkg = offering?.availablePackages?.find((p) => p.identifier === level);
    if (!pkg) throw new PurchaseError("unavailable", "That plan isn't available in the store right now. Please try again later.");
    await rc.purchasePackage(pkg);
    return "purchased";
  } catch (e) {
    throw friendlyError(e);
  }
}

export async function restore(user: { uid: string; isAnonymous: boolean } | null): Promise<void> {
  try {
    const rc = await ensureReady(user);
    await rc.restorePurchases();
  } catch (e) {
    throw friendlyError(e);
  }
}

/** Test hook: forget module state. */
export function _resetForTests(): void {
  sdk = null;
  configuredKey = null;
  loggedInAs = null;
}
