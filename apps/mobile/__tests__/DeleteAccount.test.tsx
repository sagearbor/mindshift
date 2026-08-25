/**
 * The "Delete my account" flow in Settings (AdvancedScreen) and the API call
 * behind it (src/api/account.ts).
 *
 * Its own file rather than more cases in AdvancedScreen.test.tsx: this is the
 * one irreversible action in the app, and the gating (typed confirmation),
 * the sign-out on success and the error handling deserve to be findable
 * together. The screen is rendered for real; only the network module and the
 * token provider are stubbed.
 */
import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import AdvancedScreen from "../src/screens/AdvancedScreen";
import { deleteAccount, DELETE_CONFIRMATION } from "../src/api/account";
import { getMe } from "../src/api/me";
import { getVoiceProfile, listRecordings } from "../src/api/client";
import { useAuthStore } from "../src/store/authStore";
import { useAvatarStore } from "../src/store/avatarStore";
import type { OtaStatus } from "../src/utils/otaUpdate";

let mockOta: OtaStatus;
jest.mock("../src/utils/otaUpdate", () => ({
  __esModule: true,
  useOtaStatus: () => mockOta,
  restartToApplyUpdate: jest.fn(),
}));

jest.mock("../src/api/client", () => ({
  getVoiceProfile: jest.fn(),
  listRecordings: jest.fn(),
  deleteVoiceSample: jest.fn(),
  forgetVoice: jest.fn(),
}));
jest.mock("../src/api/me", () => ({ getMe: jest.fn() }));
jest.mock("../src/api/therapist", () => ({
  getTherapistLink: jest.fn(() => Promise.resolve({ linked: false })),
  setTherapistLink: jest.fn(),
  setAutoShare: jest.fn(),
  unlinkTherapist: jest.fn(),
}));

// The screen imports DELETE_CONFIRMATION from the same module it calls, so the
// mock must keep the real constant — the whole point is that the string the UI
// asks for and the string the server demands cannot drift.
jest.mock("../src/api/account", () => ({
  __esModule: true,
  DELETE_CONFIRMATION: "DELETE",
  deleteAccount: jest.fn(),
}));

const mockDeleteAccount = deleteAccount as jest.Mock;

function queryId(
  comp: renderer.ReactTestRenderer,
  id: string,
): ReactTestInstance | null {
  const found = comp.root.findAll((n) => n.props?.testID === id);
  return found.length > 0 ? found[0] : null;
}

function textOf(node: ReactTestInstance): string {
  return node
    .findAll((n) => typeof n.type === "string")
    .flatMap((n) => n.children)
    .filter((c): c is string => typeof c === "string")
    .join("");
}

function makeHandlers() {
  return {
    onBack: jest.fn(),
    onOpenDashboard: jest.fn(),
    onSignOut: jest.fn(),
    onOpenReplay: jest.fn(),
    onOpenWatchSetup: jest.fn(),
    onOpenTutorial: jest.fn(),
    onOpenHomeDesign: jest.fn(),
    onSetProfilePhoto: jest.fn(),
  };
}

async function render(handlers = makeHandlers()) {
  let comp!: renderer.ReactTestRenderer;
  await act(async () => {
    comp = renderer.create(<AdvancedScreen {...handlers} />);
  });
  return comp;
}

/** Open the flow and type `text` into the confirmation field. */
async function open(comp: renderer.ReactTestRenderer, text?: string) {
  await act(async () => queryId(comp, "delete-account-open")!.props.onPress());
  if (text !== undefined) {
    await act(async () =>
      queryId(comp, "delete-account-confirm-input")!.props.onChangeText(text),
    );
  }
}

beforeEach(() => {
  mockOta = {
    supported: false,
    isEmbeddedLaunch: true,
    channel: null,
    createdAt: null,
    runtimeVersion: "1.18.0",
    updateId: null,
    isUpdatePending: false,
    errored: false,
  };
  (getVoiceProfile as jest.Mock).mockReset().mockRejectedValue(new Error("503"));
  (listRecordings as jest.Mock).mockReset().mockRejectedValue(new Error("503"));
  (getMe as jest.Mock).mockReset().mockRejectedValue(new Error("401"));
  mockDeleteAccount.mockReset();
  act(() => {
    useAuthStore.setState({ user: null });
    useAvatarStore.setState({ uri: null, hydrated: false });
  });
});

describe("Settings → Delete my account", () => {
  it("is collapsed by default and calls nothing until it is opened", async () => {
    const comp = await render();
    expect(queryId(comp, "delete-account-open")).not.toBeNull();
    expect(queryId(comp, "delete-account-card")).toBeNull();
    expect(queryId(comp, "delete-account-confirm-input")).toBeNull();
    expect(mockDeleteAccount).not.toHaveBeenCalled();
    act(() => comp.unmount());
  });

  it("lists what will be deleted and what cannot be reached", async () => {
    const comp = await render();
    await open(comp);

    const scope = textOf(queryId(comp, "delete-account-scope")!);
    expect(scope).toContain("recording");
    expect(scope).toContain("voiceprint");
    expect(scope).toContain("therapist");
    expect(scope).toContain("sign-in account");

    const limits = textOf(queryId(comp, "delete-account-limits")!);
    expect(limits).toContain("photo library");
    expect(limits).toContain("Server logs");

    act(() => comp.unmount());
  });

  it("keeps the button disabled until the exact word is typed", async () => {
    const comp = await render();
    await open(comp);

    const isDisabled = () => queryId(comp, "delete-account-submit")!.props.disabled;
    expect(isDisabled()).toBe(true);

    for (const wrong of ["", "delete", "DELET", "DELETE ME", "Delete"]) {
      await act(async () =>
        queryId(comp, "delete-account-confirm-input")!.props.onChangeText(wrong),
      );
      expect(isDisabled()).toBe(true);
    }

    await act(async () =>
      queryId(comp, "delete-account-confirm-input")!.props.onChangeText(
        DELETE_CONFIRMATION,
      ),
    );
    expect(isDisabled()).toBe(false);
    expect(mockDeleteAccount).not.toHaveBeenCalled();
    act(() => comp.unmount());
  });

  it("does not delete when the disabled button is pressed anyway", async () => {
    const comp = await render();
    await open(comp, "nope");
    await act(async () => queryId(comp, "delete-account-submit")!.props.onPress());
    expect(mockDeleteAccount).not.toHaveBeenCalled();
    act(() => comp.unmount());
  });

  it("deletes and signs out on success", async () => {
    const handlers = makeHandlers();
    mockDeleteAccount.mockResolvedValue({
      deleted: true,
      firebase_user_deleted: true,
      counts: { recordings: 3, voiceprints: 1 },
    });

    const comp = await render(handlers);
    await open(comp, DELETE_CONFIRMATION);
    await act(async () => queryId(comp, "delete-account-submit")!.props.onPress());

    expect(mockDeleteAccount).toHaveBeenCalledTimes(1);
    expect(handlers.onSignOut).toHaveBeenCalledTimes(1);
    // Back to the collapsed row — the account is gone, so is the flow.
    expect(queryId(comp, "delete-account-card")).toBeNull();
    act(() => comp.unmount());
  });

  it("keeps the account and shows the server's reason when deletion fails", async () => {
    const handlers = makeHandlers();
    const err = new Error(
      "Some of your data could not be deleted, so your account was left in place.",
    ) as Error & { status?: number };
    err.status = 500;
    mockDeleteAccount.mockRejectedValue(err);

    const comp = await render(handlers);
    await open(comp, DELETE_CONFIRMATION);
    await act(async () => queryId(comp, "delete-account-submit")!.props.onPress());

    expect(handlers.onSignOut).not.toHaveBeenCalled();
    expect(textOf(queryId(comp, "delete-account-error")!)).toContain(
      "left in place",
    );
    // Still open and still armed, so the user can just press again.
    expect(queryId(comp, "delete-account-submit")!.props.disabled).toBe(false);
    act(() => comp.unmount());
  });

  it("explains a 401 as an expired sign-in rather than a generic failure", async () => {
    const err = new Error("invalid or expired token") as Error & {
      status?: number;
    };
    err.status = 401;
    mockDeleteAccount.mockRejectedValue(err);

    const comp = await render();
    await open(comp, DELETE_CONFIRMATION);
    await act(async () => queryId(comp, "delete-account-submit")!.props.onPress());

    expect(textOf(queryId(comp, "delete-account-error")!)).toContain(
      "sign in again",
    );
    act(() => comp.unmount());
  });

  it("cancelling closes the flow and forgets what was typed", async () => {
    const handlers = makeHandlers();
    const comp = await render(handlers);
    await open(comp, DELETE_CONFIRMATION);

    await act(async () => queryId(comp, "delete-account-cancel")!.props.onPress());
    expect(queryId(comp, "delete-account-card")).toBeNull();
    expect(mockDeleteAccount).not.toHaveBeenCalled();
    expect(handlers.onSignOut).not.toHaveBeenCalled();

    // Re-opening starts disarmed, never pre-filled from last time.
    await open(comp);
    expect(queryId(comp, "delete-account-confirm-input")!.props.value).toBe("");
    expect(queryId(comp, "delete-account-submit")!.props.disabled).toBe(true);
    act(() => comp.unmount());
  });
});
