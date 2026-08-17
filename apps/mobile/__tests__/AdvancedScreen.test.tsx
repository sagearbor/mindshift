import React from "react";
import { Alert } from "react-native";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import AdvancedScreen from "../src/screens/AdvancedScreen";
import {
  deleteVoiceSample,
  forgetVoice,
  getVoiceProfile,
  listRecordings,
} from "../src/api/client";
import type { VoiceProfile } from "../src/api/client";
import { useAuthStore } from "../src/store/authStore";
import type { OtaStatus } from "../src/utils/otaUpdate";

// The About section reads OTA status through this hook; drive it per-test. The
// `mock`-prefixed name is the only ref jest.mock's hoisted factory may close over.
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

// The guided training flow is its own tested component (VoiceTrainingFlow
// .test.tsx); here it is a stub that records the props AdvancedScreen wires
// so the toggle + completion refresh can be driven without a recorder.
let mockVtProps: {
  onDone: (count: number) => void;
  onCancel: () => void;
} | null = null;
jest.mock("../src/components/VoiceTrainingFlow", () => {
  const React = require("react");
  const { View } = require("react-native");
  return {
    __esModule: true,
    default: (props: { onDone: (count: number) => void; onCancel: () => void }) => {
      mockVtProps = props;
      return React.createElement(View, { testID: "voice-training-flow" });
    },
  };
});
const mockProfile = getVoiceProfile as jest.Mock;
const mockList = listRecordings as jest.Mock;
const mockDeleteSample = deleteVoiceSample as jest.Mock;
const mockForget = forgetVoice as jest.Mock;

function baseOta(overrides: Partial<OtaStatus> = {}): OtaStatus {
  return {
    supported: false,
    isEmbeddedLaunch: true,
    channel: null,
    createdAt: null,
    runtimeVersion: "1.14.0",
    updateId: null,
    isUpdatePending: false,
    errored: false,
    ...overrides,
  };
}

function enrolledProfile(overrides: Partial<VoiceProfile> = {}): VoiceProfile {
  return {
    available: true,
    storage_enabled: true,
    enrolled: true,
    enroll_count: 2,
    updated_at: "2026-07-12T10:00:00+00:00",
    model: "m@rev",
    dim: 192,
    samples: [
      {
        id: "s1",
        recording_id: "rec-1",
        speaker: "Speaker B",
        at: "2026-07-12T10:00:00+00:00",
      },
      {
        id: "s2",
        recording_id: "rec-gone",
        speaker: "Speaker A",
        at: "2026-07-01T10:00:00+00:00",
      },
    ],
    ...overrides,
  };
}

function queryId(
  comp: renderer.ReactTestRenderer,
  id: string,
): ReactTestInstance | null {
  const found = comp.root.findAll((n) => n.props?.testID === id);
  return found.length > 0 ? found[0] : null;
}

/** Concatenated string content rendered under a node (its Text leaves). */
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
  };
}

async function render(handlers = makeHandlers()) {
  let comp!: renderer.ReactTestRenderer;
  await act(async () => {
    comp = renderer.create(<AdvancedScreen {...handlers} />);
  });
  return comp;
}

beforeEach(() => {
  mockOta = baseOta();
  mockVtProps = null;
  mockProfile.mockReset();
  mockList.mockReset();
  mockDeleteSample.mockReset();
  mockForget.mockReset();
  // Default: no voice feature — the card stays hidden and the classic
  // Advanced tests below run against the unchanged surface.
  mockProfile.mockRejectedValue(new Error("API error: 503"));
  mockList.mockRejectedValue(new Error("API error: 503"));
  act(() => {
    useAuthStore.setState({ user: null });
  });
});

describe("AdvancedScreen", () => {
  it("is titled Settings, not Advanced", async () => {
    const comp = await render();
    expect(textOf(queryId(comp, "settings-heading")!)).toBe("Settings");
    act(() => comp.unmount());
  });

  it("groups its rows under labeled sections: Your tools, About, Account", async () => {
    // No voice feature in this default mock state, so no Voice section —
    // an empty section header would be worse than none.
    const comp = await render();
    expect(textOf(queryId(comp, "section-your-tools")!)).toBe("Your tools");
    expect(textOf(queryId(comp, "section-about")!)).toBe("About");
    expect(textOf(queryId(comp, "section-account")!)).toBe("Account");
    expect(queryId(comp, "section-voice")).toBeNull();
    act(() => comp.unmount());
  });

  it("shows the Voice section header only when the voice card renders", async () => {
    mockProfile.mockResolvedValue(enrolledProfile());
    mockList.mockResolvedValue([]);
    const comp = await render();
    expect(textOf(queryId(comp, "section-voice")!)).toBe("Voice");
    act(() => comp.unmount());
  });

  it("renders the dashboard entry, log out, and back — and wires each press", async () => {
    const handlers = makeHandlers();
    const comp = await render(handlers);

    act(() => queryId(comp, "advanced-dashboard")!.props.onPress());
    expect(handlers.onOpenDashboard).toHaveBeenCalledTimes(1);

    const logOut = queryId(comp, "advanced-sign-out")!;
    expect(textOf(logOut)).toBe("Log out");
    act(() => logOut.props.onPress());
    expect(handlers.onSignOut).toHaveBeenCalledTimes(1);

    act(() => queryId(comp, "advanced-back")!.props.onPress());
    expect(handlers.onBack).toHaveBeenCalledTimes(1);

    act(() => comp.unmount());
  });

  it("renders the watch-setup entry and wires its press", async () => {
    const handlers = makeHandlers();
    const comp = await render(handlers);

    act(() => queryId(comp, "advanced-watch-setup")!.props.onPress());
    expect(handlers.onOpenWatchSetup).toHaveBeenCalledTimes(1);

    act(() => comp.unmount());
  });

  it("About shows version, build, account email, backend host, and the honest store-build OTA line", async () => {
    process.env.EXPO_PUBLIC_API_URL = "https://mindshift-api.example.run.app";
    act(() => {
      useAuthStore.setState({
        user: { uid: "u1", email: "tester@example.com", displayName: "T" },
      });
    });
    mockOta = baseOta({ supported: false });

    const comp = await render();

    expect(queryId(comp, "about-section")).toBeTruthy();
    // Mocked expo-application / expo-constants from jest-setup.
    expect(textOf(queryId(comp, "about-version")!)).toContain("1.14.0");
    expect(textOf(queryId(comp, "about-build")!)).toContain("29");
    expect(textOf(queryId(comp, "about-account")!)).toContain(
      "tester@example.com",
    );
    expect(textOf(queryId(comp, "about-backend")!)).toContain(
      "mindshift-api.example.run.app",
    );
    // No OTA module in this build → an honest store-build line, not a fake channel.
    expect(textOf(queryId(comp, "about-update")!)).toContain(
      "Store build (no OTA yet)",
    );

    act(() => comp.unmount());
  });

  it("About reports a downloaded OTA update with its publish time and channel", async () => {
    mockOta = baseOta({
      supported: true,
      isEmbeddedLaunch: false,
      channel: "production",
      createdAt: new Date("2026-07-19T15:30:00Z"),
      updateId: "abc",
    });

    const comp = await render();

    const updateText = textOf(queryId(comp, "about-update")!);
    expect(updateText).toContain("Updated");
    expect(updateText).toContain("production channel");

    act(() => comp.unmount());
  });

  it("About always shows a MindShift line", async () => {
    const comp = await render();
    expect(textOf(queryId(comp, "about-app-name")!)).toBe("MindShift");
    act(() => comp.unmount());
  });

  it("About shows the running update's id only when one is known", async () => {
    mockOta = baseOta({
      supported: true,
      isEmbeddedLaunch: false,
      channel: "production",
      createdAt: new Date("2026-07-19T15:30:00Z"),
      updateId: "abc-123",
    });
    const comp = await render();
    expect(textOf(queryId(comp, "about-update-id")!)).toContain("abc-123");
    act(() => comp.unmount());
  });

  it("About hides the update-id row on a store build with no OTA applied", async () => {
    // baseOta() defaults updateId to null — the honest default.
    const comp = await render();
    expect(queryId(comp, "about-update-id")).toBeNull();
    act(() => comp.unmount());
  });

  it("About falls back honestly when signed in without an email on file", async () => {
    act(() => {
      useAuthStore.setState({
        user: { uid: "u2", email: null, displayName: null },
      });
    });

    const comp = await render();

    expect(textOf(queryId(comp, "about-account")!)).toContain(
      "No email on this account",
    );

    act(() => comp.unmount());
  });
});

describe("AdvancedScreen — voice profile card", () => {
  it("stays hidden when the server can't do voice ID", async () => {
    mockProfile.mockResolvedValue(
      enrolledProfile({ available: false, storage_enabled: false }),
    );
    mockList.mockResolvedValue([]);
    const comp = await render();
    expect(queryId(comp, "voice-profile-card")).toBeNull();
    act(() => comp.unmount());
  });

  it("explains the This-is-me flow when not enrolled", async () => {
    mockProfile.mockResolvedValue(
      enrolledProfile({ enrolled: false, enroll_count: 0, samples: [] }),
    );
    mockList.mockResolvedValue([]);
    const comp = await render();
    const status = queryId(comp, "voice-profile-status")!;
    expect(textOf(status)).toContain("Not enrolled");
    expect(textOf(status)).toContain("This is me");
    expect(queryId(comp, "advanced-forget-voice")).toBeNull();
    act(() => comp.unmount());
  });

  it("shows status, dated samples with provenance, and honest deleted-source rows", async () => {
    mockProfile.mockResolvedValue(enrolledProfile());
    mockList.mockResolvedValue([
      { id: "rec-1", title: "photos_share.mp4", filename: "photos_share.mp4" },
    ]);
    const handlers = makeHandlers();
    const comp = await render(handlers);

    expect(textOf(queryId(comp, "voice-profile-status")!)).toContain(
      "Enrolled · 2 samples",
    );

    // Sample from a live recording: title + speaker + play.
    const s1 = queryId(comp, "voice-sample-s1")!;
    expect(textOf(s1)).toContain("photos_share.mp4");
    expect(textOf(s1)).toContain("Speaker B");
    act(() => queryId(comp, "voice-sample-play-s1")!.props.onPress());
    expect(handlers.onOpenReplay).toHaveBeenCalledWith("rec-1");

    // Sample whose source recording is gone: said plainly, and not playable.
    const s2 = queryId(comp, "voice-sample-s2")!;
    expect(textOf(s2)).toContain("source recording deleted");
    expect(queryId(comp, "voice-sample-play-s2")).toBeNull();

    // The add-another explainer is always present when enrolled.
    expect(textOf(queryId(comp, "voice-add-sample-hint")!)).toContain(
      "This is me",
    );
    act(() => comp.unmount());
  });

  it("labels the migrated pre-v2 blend with its note and no play", async () => {
    mockProfile.mockResolvedValue(
      enrolledProfile({
        enroll_count: 1,
        samples: [
          {
            id: "legacy-blend",
            recording_id: null,
            speaker: null,
            at: "2026-06-01T10:00:00+00:00",
            note: "pre-v2 blend of 3 enrollments",
          },
        ],
      }),
    );
    mockList.mockResolvedValue([]);
    const comp = await render();
    const row = queryId(comp, "voice-sample-legacy-blend")!;
    expect(textOf(row)).toContain("pre-v2 blend of 3 enrollments");
    expect(queryId(comp, "voice-sample-play-legacy-blend")).toBeNull();
    act(() => comp.unmount());
  });

  it("deletes a sample optimistically and trusts the server's remaining count", async () => {
    mockProfile.mockResolvedValue(enrolledProfile());
    mockList.mockResolvedValue([]);
    let resolveDelete!: (v: unknown) => void;
    mockDeleteSample.mockReturnValue(
      new Promise((resolve) => {
        resolveDelete = resolve;
      }),
    );
    const comp = await render();

    act(() => queryId(comp, "voice-sample-delete-s2")!.props.onPress());
    // Optimistic: the row is gone before the server answers.
    expect(queryId(comp, "voice-sample-s2")).toBeNull();
    expect(textOf(queryId(comp, "voice-profile-status")!)).toContain(
      "1 sample",
    );

    await act(async () => {
      resolveDelete({ deleted: true, enrolled: true, enroll_count: 1 });
    });
    expect(mockDeleteSample).toHaveBeenCalledWith("s2");
    expect(queryId(comp, "voice-sample-s2")).toBeNull();
    expect(queryId(comp, "voice-sample-s1")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("rolls the removal back with an honest message when the delete fails", async () => {
    mockProfile.mockResolvedValue(enrolledProfile());
    mockList.mockResolvedValue([]);
    mockDeleteSample.mockRejectedValue(
      Object.assign(new Error("API error: 503"), { status: 503 }),
    );
    const comp = await render();

    await act(async () =>
      queryId(comp, "voice-sample-delete-s1")!.props.onPress(),
    );
    // Rolled back: the sample is still there, and the failure is stated.
    expect(queryId(comp, "voice-sample-s1")).toBeTruthy();
    expect(textOf(queryId(comp, "voice-sample-error")!)).toContain(
      "Couldn’t delete",
    );
    act(() => comp.unmount());
  });

  it("deleting the last sample leaves the not-enrolled state", async () => {
    mockProfile.mockResolvedValue(
      enrolledProfile({
        enroll_count: 1,
        samples: [
          {
            id: "s1",
            recording_id: "rec-1",
            speaker: "Speaker A",
            at: "2026-07-12T10:00:00+00:00",
          },
        ],
      }),
    );
    mockList.mockResolvedValue([]);
    mockDeleteSample.mockResolvedValue({
      deleted: true,
      enrolled: false,
      enroll_count: 0,
    });
    const comp = await render();

    await act(async () =>
      queryId(comp, "voice-sample-delete-s1")!.props.onPress(),
    );
    expect(textOf(queryId(comp, "voice-profile-status")!)).toContain(
      "Not enrolled",
    );
    expect(queryId(comp, "advanced-forget-voice")).toBeNull();
    act(() => comp.unmount());
  });

  it("offers Train my voice in both the enrolled and not-enrolled card", async () => {
    mockProfile.mockResolvedValue(enrolledProfile());
    mockList.mockResolvedValue([]);
    let comp = await render();
    expect(queryId(comp, "voice-train-button")).toBeTruthy();
    act(() => comp.unmount());

    mockProfile.mockResolvedValue(
      enrolledProfile({ enrolled: false, enroll_count: 0, samples: [] }),
    );
    comp = await render();
    expect(queryId(comp, "voice-train-button")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("opens the guided flow in place and closes it on cancel", async () => {
    mockProfile.mockResolvedValue(enrolledProfile());
    mockList.mockResolvedValue([]);
    const comp = await render();

    expect(queryId(comp, "voice-training-flow")).toBeNull();
    act(() => queryId(comp, "voice-train-button")!.props.onPress());
    expect(queryId(comp, "voice-training-flow")).toBeTruthy();
    // The button hides while the flow is open (one affordance at a time).
    expect(queryId(comp, "voice-train-button")).toBeNull();

    act(() => mockVtProps!.onCancel());
    expect(queryId(comp, "voice-training-flow")).toBeNull();
    expect(queryId(comp, "voice-train-button")).toBeTruthy();

    act(() => comp.unmount());
  });

  it("refreshes the profile from the server when guided training completes", async () => {
    mockProfile.mockResolvedValue(
      enrolledProfile({ enrolled: false, enroll_count: 0, samples: [] }),
    );
    mockList.mockResolvedValue([]);
    const comp = await render();
    expect(textOf(queryId(comp, "voice-profile-status")!)).toContain(
      "Not enrolled",
    );

    act(() => queryId(comp, "voice-train-button")!.props.onPress());
    // The server is the source of truth after enrollment: onDone refetches.
    mockProfile.mockResolvedValue(
      enrolledProfile({
        enroll_count: 1,
        samples: [
          {
            id: "g1",
            recording_id: null,
            speaker: null,
            at: "2026-08-15T10:00:00+00:00",
            note: "guided enrollment",
          },
        ],
      }),
    );
    await act(async () => mockVtProps!.onDone(1));

    expect(mockProfile).toHaveBeenCalledTimes(2);
    expect(queryId(comp, "voice-training-flow")).toBeNull();
    expect(textOf(queryId(comp, "voice-profile-status")!)).toContain(
      "Enrolled · 1 sample",
    );
    // The guided sample's provenance note is shown.
    expect(textOf(queryId(comp, "voice-sample-g1")!)).toContain(
      "guided enrollment",
    );

    act(() => comp.unmount());
  });

  it("forget-my-voice confirms, calls the API, and empties the card", async () => {
    mockProfile.mockResolvedValue(enrolledProfile());
    mockList.mockResolvedValue([]);
    mockForget.mockResolvedValue({ deleted: true });
    const alertSpy = jest.spyOn(Alert, "alert").mockImplementation(() => {});
    const comp = await render();

    act(() => queryId(comp, "advanced-forget-voice")!.props.onPress());
    expect(alertSpy).toHaveBeenCalled();
    // Fire the destructive confirm button from the Alert's config.
    const buttons = alertSpy.mock.calls[0][2] as Array<{
      style?: string;
      onPress?: () => void;
    }>;
    const destructive = buttons.find((b) => b.style === "destructive")!;
    await act(async () => destructive.onPress!());

    expect(mockForget).toHaveBeenCalledTimes(1);
    expect(textOf(queryId(comp, "voice-profile-status")!)).toContain(
      "Not enrolled",
    );
    alertSpy.mockRestore();
    act(() => comp.unmount());
  });
});
