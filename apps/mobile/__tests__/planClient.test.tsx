import React from "react";
import renderer, { act } from "react-test-renderer";
import { getPlan, parsePlanLimit, redeemCode } from "../src/api/plan";
import { reportPlanLimit, usePlanLimitStore } from "../src/store/planLimitStore";
import PlanLimitSheet from "../src/components/PlanLimitSheet";
import { setCachedToken, setTokenProvider } from "../src/auth/authToken";

const mockFetch = global.fetch as jest.Mock;

beforeEach(() => {
  mockFetch.mockReset();
  setCachedToken(null);
  setTokenProvider(async () => "tok");
  usePlanLimitStore.setState({ notice: null });
});
afterAll(() => setTokenProvider(null));

const PLAN = { plan: "free", label: "Free", redeem_available: true };

test("getPlan returns the server summary", async () => {
  mockFetch.mockResolvedValue({ ok: true, status: 200, json: async () => PLAN });
  expect((await getPlan()).label).toBe("Free");
  expect(mockFetch.mock.calls[0][0]).toMatch(/\/plan$/);
});

test("redeemCode posts the code and returns the new plan", async () => {
  mockFetch.mockResolvedValue({
    ok: true,
    status: 200,
    json: async () => ({ status: "granted", plan: { ...PLAN, plan: "pro", label: "Pro" } }),
  });
  const r = await redeemCode("ABCD-2345-EFGH");
  expect(r.status).toBe("granted");
  expect(JSON.parse(mockFetch.mock.calls[0][1].body)).toEqual({ code: "ABCD-2345-EFGH" });
});

test("redeemCode surfaces the server's friendly message and status", async () => {
  mockFetch.mockResolvedValue({
    ok: false,
    status: 400,
    json: async () => ({ detail: { reason: "invalid_code", message: "That code isn't valid." } }),
  });
  await expect(redeemCode("x")).rejects.toMatchObject({
    message: "That code isn't valid.",
    status: 400,
    reason: "invalid_code",
  });
});

test("parsePlanLimit only accepts typed limit bodies", () => {
  expect(parsePlanLimit("nope")).toBeNull();
  expect(parsePlanLimit({ type: "other", message: "x" })).toBeNull();
  expect(
    parsePlanLimit({ type: "plan_limit", message: "Full", upgrade_plan: "pro", limit: "library_items" }),
  ).toMatchObject({ message: "Full", upgradePlan: "pro", limit: "library_items" });
  expect(parsePlanLimit({ type: "quota_notice", message: "Daily", plan: "free" })?.plan).toBe("free");
});

test("a plan-limit report opens the upgrade sheet with the server's sentence", () => {
  expect(reportPlanLimit({ type: "plan_limit", message: "Your Free plan holds 5 items.", upgrade_plan: "pro" })).toBe(
    "Your Free plan holds 5 items.",
  );
  let tree!: renderer.ReactTestRenderer;
  act(() => {
    tree = renderer.create(<PlanLimitSheet />);
  });
  expect(tree.root.findByProps({ testID: "plan-limit-message" }).props.children).toBe("Your Free plan holds 5 items.");
  act(() => {
    tree.root.findByProps({ testID: "plan-limit-dismiss" }).props.onPress();
  });
  expect(usePlanLimitStore.getState().notice).toBeNull();
});
