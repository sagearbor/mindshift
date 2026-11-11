import React from "react";
import renderer, { act } from "react-test-renderer";

jest.mock("../src/emotion/devPrefs", () => ({
  loadEmotionDevSettings: jest.fn(),
  saveEmotionDevSettings: jest.fn().mockResolvedValue(undefined),
}));
jest.mock("../src/components/EmotionReview", () => ({ EmotionReview: () => null }));
jest.mock("../src/components/ScriptedEmotionTest", () => ({ ScriptedEmotionTest: () => null }));

import { EmotionDevPanel } from "../src/components/EmotionDevPanel";
import { DEFAULT_USER_SETTINGS } from "../src/emotion/config";
import { emotionDevHub } from "../src/emotion/hub";
import { loadEmotionDevSettings } from "../src/emotion/devPrefs";

const find = (c: renderer.ReactTestRenderer, id: string) => c.root.findAll((n) => n.props?.testID === id);

async function mount(enabled: boolean, onStartLiveCoach?: () => void) {
  (loadEmotionDevSettings as jest.Mock).mockResolvedValue({ ...DEFAULT_USER_SETTINGS, enabled });
  let c!: renderer.ReactTestRenderer;
  await act(async () => {
    c = renderer.create(<EmotionDevPanel userId="u1" onStartLiveCoach={onStartLiveCoach} />);
  });
  await act(async () => {});
  return c;
}

afterEach(async () => {
  await act(async () => {
    await emotionDevHub.configure(DEFAULT_USER_SETTINGS);
  });
});

describe("EmotionDevPanel — Start Live Coach shortcut", () => {
  it("is offered when the tester is on and the coach is not running, and opens Live Coach", async () => {
    const onStart = jest.fn();
    const c = await mount(true, onStart);
    const btn = find(c, "emotion-dev-start-live-coach")[0];
    expect(btn).toBeTruthy();
    act(() => btn.props.onPress());
    expect(onStart).toHaveBeenCalledTimes(1);
    act(() => c.unmount());
  });

  it("is absent when the tester is off or no handler is wired", async () => {
    const off = await mount(false, jest.fn());
    expect(find(off, "emotion-dev-start-live-coach")).toHaveLength(0);
    act(() => off.unmount());
    const unwired = await mount(true, undefined);
    expect(find(unwired, "emotion-dev-start-live-coach")).toHaveLength(0);
    act(() => unwired.unmount());
  });
});
