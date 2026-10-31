import React from "react";
import renderer, { act } from "react-test-renderer";
import PlanCard from "../src/components/PlanCard";
import { getPlan, redeemCode } from "../src/api/plan";

jest.mock("../src/api/plan", () => ({
  getPlan: jest.fn(),
  redeemCode: jest.fn(),
}));

const BASE = {
  plan: "free",
  label: "Free",
  expires_at: null,
  source: null,
  is_guest: false,
  limits: { library_items: 5 },
  daily: {},
  windows: { stt_lifetime: { cap: 7200, used: 3600 } },
  library: { items: 0, chars: 0 },
  flags: {},
  upgrade_plan: "pro",
  redeem_available: true,
  referral: { code: "ABCD-2345", rewards_earned: 1, bonus_hours_each: 2, bonus_hours: 2, max_bonus_hours: 10, already_referred: false },
  purchase: { available: false },
};

async function render() {
  let tree!: renderer.ReactTestRenderer;
  await act(async () => {
    tree = renderer.create(<PlanCard />);
  });
  return tree;
}

const text = (t: renderer.ReactTestRenderer) => JSON.stringify(t.toJSON());

test("shows plan, free hours left, share code and a disabled upgrade", async () => {
  (getPlan as jest.Mock).mockResolvedValue(BASE);
  const t = await render();
  expect(text(t)).toContain("Free");
  expect(text(t)).toContain("1.0 of 2 free transcription hours left");
  expect(t.root.findByProps({ testID: "plan-referral-code" }).props.children).toBe("ABCD-2345");
  expect(text(t)).toContain("Invite a friend: you both get");
  expect(text(t)).toContain("more free hours");
  expect(t.root.findByProps({ testID: "plan-upgrade" }).props.disabled).toBe(true);
});

test("redeeming a friend's code updates the card", async () => {
  (getPlan as jest.Mock).mockResolvedValue(BASE);
  (redeemCode as jest.Mock).mockResolvedValue({
    status: "granted",
    plan: { ...BASE, plan: "pro", label: "Pro", source: "promo", expires_at: "2030-01-01T00:00:00Z", windows: {} },
  });
  const t = await render();
  await act(async () => {
    t.root.findByProps({ testID: "plan-code-input" }).props.onChangeText("WXYZ-9876");
  });
  await act(async () => {
    await t.root.findByProps({ testID: "plan-redeem-button" }).props.onPress();
  });
  expect(redeemCode).toHaveBeenCalledWith("WXYZ-9876");
  expect(t.root.findByProps({ testID: "plan-name" }).props.children.join("")).toContain("Pro");
});

test("guests are told to create an account", async () => {
  (getPlan as jest.Mock).mockResolvedValue({ ...BASE, is_guest: true, redeem_available: false, referral: null });
  const t = await render();
  expect(text(t)).toContain("Create an account to redeem a code.");
});
