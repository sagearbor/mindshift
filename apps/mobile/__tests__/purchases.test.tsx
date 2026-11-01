import React from "react";
import renderer, { act } from "react-test-renderer";

const mockSdk = {
  configure: jest.fn(),
  logIn: jest.fn().mockResolvedValue({}),
  logOut: jest.fn().mockResolvedValue({}),
  getOfferings: jest.fn(),
  purchasePackage: jest.fn(),
  restorePurchases: jest.fn(),
};
jest.mock("react-native-purchases", () => ({ __esModule: true, default: mockSdk }));
jest.mock("../src/api/plan", () => ({ getPlan: jest.fn(), redeemCode: jest.fn() }));
jest.mock("react-native", () => {
  const rn = jest.requireActual("react-native");
  rn.Platform.OS = "ios";
  return rn;
});

import { getPlan } from "../src/api/plan";
import * as P from "../src/purchases/purchases";
import { startUpgrade, startRestore } from "../src/purchases/upgradeFlow";
import { useAuthStore } from "../src/store/authStore";
import PlanCard from "../src/components/PlanCard";

const USER = { uid: "uid-real-1", email: "a@b.c", displayName: null, isAnonymous: false };
const OFFERINGS = {
  all: { default: { availablePackages: ["basic", "plus", "pro", "max"].map((identifier) => ({ identifier })) } },
};
const plan = (level: string, extra = {}) => ({
  plan: level, label: level[0].toUpperCase() + level.slice(1), expires_at: null, source: "revenuecat",
  is_guest: false, limits: {}, daily: {}, library: { items: 0, chars: 0 }, flags: {},
  upgrade_plan: null, redeem_available: true, windows: {},
  tiers: [
    { level: "basic", label: "Basic", price_usd: 5, hours_per_month: 4, library_items: 10, popular: false, business: false },
    { level: "pro", label: "Pro", price_usd: 25, hours_per_month: 20, library_items: 200, popular: false, business: false },
  ],
  purchase: { available: true }, ...extra,
});

beforeEach(() => {
  jest.clearAllMocks();
  P._resetForTests();
  process.env.EXPO_PUBLIC_REVENUECAT_IOS_KEY = "appl_test";
  delete process.env.EXPO_PUBLIC_REVENUECAT_ANDROID_KEY;
  useAuthStore.setState({ user: USER } as never);
  mockSdk.getOfferings.mockResolvedValue(OFFERINGS);
  mockSdk.purchasePackage.mockResolvedValue({});
});

test("no key => disabled, never touches the SDK", async () => {
  delete process.env.EXPO_PUBLIC_REVENUECAT_IOS_KEY;
  expect(P.purchasesEnabled()).toBe(false);
  const r = await startUpgrade("pro", 0);
  expect(r.ok).toBe(false);
  expect(r.message).toMatch(/coming soon/i);
  expect(mockSdk.configure).not.toHaveBeenCalled();
  expect(mockSdk.purchasePackage).not.toHaveBeenCalled();
});

test("guests never log in or buy", async () => {
  useAuthStore.setState({ user: { ...USER, isAnonymous: true } } as never);
  const r = await startUpgrade("pro", 0);
  expect(r.ok).toBe(false);
  expect(mockSdk.logIn).not.toHaveBeenCalled();
  expect(mockSdk.purchasePackage).not.toHaveBeenCalled();
});

test("purchase configures with the key, logs in as the uid, buys the matching package, refreshes /plan", async () => {
  (getPlan as jest.Mock).mockResolvedValueOnce(plan("free")).mockResolvedValueOnce(plan("pro"));
  const r = await startUpgrade("pro", 0);
  expect(mockSdk.configure).toHaveBeenCalledWith({ apiKey: "appl_test" });
  expect(mockSdk.logIn).toHaveBeenCalledWith("uid-real-1");
  expect(mockSdk.purchasePackage).toHaveBeenCalledWith({ identifier: "pro" });
  expect(getPlan).toHaveBeenCalledTimes(2); // waited for the webhook
  expect(r.ok).toBe(true);
  expect(r.message).toContain("Pro");
});

test("user cancel is quiet, not an error", async () => {
  mockSdk.purchasePackage.mockRejectedValue({ userCancelled: true });
  const r = await startUpgrade("plus", 0);
  expect(r).toMatchObject({ ok: false, cancelled: true });
  expect(getPlan).not.toHaveBeenCalled();
});

test("store failures get friendly text", async () => {
  mockSdk.purchasePackage.mockRejectedValue({ code: "2", message: "raw sdk text" });
  const r = await startUpgrade("plus", 0);
  expect(r.ok).toBe(false);
  expect(r.message).toMatch(/connection/);
  expect(r.message).not.toContain("raw sdk text");
});

test("missing package in the offering is reported, not faked", async () => {
  mockSdk.getOfferings.mockResolvedValue({ all: { default: { availablePackages: [] } } });
  const r = await startUpgrade("plus", 0);
  expect(r.ok).toBe(false);
  expect(r.message).toMatch(/isn't available/);
});

test("restore calls the SDK and refreshes /plan", async () => {
  (getPlan as jest.Mock).mockResolvedValue(plan("basic"));
  mockSdk.restorePurchases.mockResolvedValue({});
  const r = await startRestore(0);
  expect(mockSdk.restorePurchases).toHaveBeenCalled();
  expect(r.ok).toBe(true);
  expect(r.message).toContain("Basic");
});

async function render() {
  let tree!: renderer.ReactTestRenderer;
  await act(async () => {
    tree = renderer.create(<PlanCard />);
  });
  return tree;
}

test("Plan card: Subscribe buys; Restore button present; 'Coming soon' without a key", async () => {
  (getPlan as jest.Mock).mockResolvedValue(plan("free"));
  let t = await render();
  expect(t.root.findByProps({ testID: "tier-basic-buy" }).props.disabled).toBe(false);
  expect(JSON.stringify(t.toJSON())).toContain("Subscribe");
  expect(t.root.findByProps({ testID: "plan-restore" })).toBeTruthy();
  await act(async () => {
    t.root.findByProps({ testID: "tier-basic-buy" }).props.onPress();
  });
  expect(mockSdk.purchasePackage).toHaveBeenCalledWith({ identifier: "basic" });

  delete process.env.EXPO_PUBLIC_REVENUECAT_IOS_KEY;
  t = await render();
  expect(t.root.findByProps({ testID: "tier-basic-buy" }).props.disabled).toBe(true);
  expect(JSON.stringify(t.toJSON())).toContain("Coming soon");
  expect(t.root.findAllByProps({ testID: "plan-restore" })).toHaveLength(0);
});
