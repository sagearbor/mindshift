/**
 * Pins the behaviors added by the 2026-09-23 UX walkthrough (web build walked
 * as a first-time guest; source review alongside). Each `it` names the
 * problem it prevents coming back.
 */
import React from "react";
import { Platform } from "react-native";
import renderer, { act, type ReactTestInstance } from "react-test-renderer";

import AccountMenu from "../src/components/AccountMenu";
import LiveTranscript from "../src/components/LiveTranscript";
import JournalPanel from "../src/components/JournalPanel";
import RecoveryHomeCard from "../src/recorder/RecoveryHomeCard";
import { IDLE_JOURNAL_STATE } from "../src/live/journalRecorder";
import { DESTINATIONS } from "../src/nav/destinations";
import { showAlert } from "../src/utils/showAlert";
import { PRIVACY_POLICY_URL } from "../src/utils/legalLinks";

function byId(c: renderer.ReactTestRenderer, id: string): ReactTestInstance | null {
  const found = c.root.findAll((n) => n.props?.testID === id);
  return found[0] ?? null;
}
function textOf(node: ReactTestInstance | null): string {
  if (!node) return "";
  // Host Text nodes only — a composite <Text> and the host node it renders
  // both carry the same string, which would double every word.
  return node
    .findAll((n) => String(n.type) === "Text" && typeof n.props?.children === "string")
    .map((n) => n.props.children as string)
    .join("");
}

describe("account menu — a guest is not 'signed in as this account'", () => {
  it("names the guest session for an anonymous user", () => {
    let c!: renderer.ReactTestRenderer;
    act(() => {
      c = renderer.create(
        <AccountMenu
          user={{ email: null, displayName: null, isAnonymous: true }}
          onOpenSettings={() => {}}
          onSignOut={() => {}}
          onClose={() => {}}
        />,
      );
    });
    expect(textOf(byId(c, "chrome-account-email"))).toBe("Guest session — no account yet");
  });
  it("still says who a real account is signed in as", () => {
    let c!: renderer.ReactTestRenderer;
    act(() => {
      c = renderer.create(
        <AccountMenu
          user={{ email: "a@b.c", displayName: null }}
          onOpenSettings={() => {}}
          onSignOut={() => {}}
          onClose={() => {}}
        />,
      );
    });
    expect(textOf(byId(c, "chrome-account-email"))).toBe("Signed in as a@b.c");
  });
});

describe("live transcript — before Start it must not claim to be listening", () => {
  it("tells an idle user what to do instead of 'Waiting for conversation'", () => {
    let c!: renderer.ReactTestRenderer;
    act(() => {
      c = renderer.create(<LiveTranscript entries={[]} idle />);
    });
    const text = textOf(byId(c, "live-transcript-empty"));
    expect(text).toMatch(/Tap Start Listening/);
    expect(text).not.toMatch(/Waiting/);
  });
  it("keeps the waiting copy once a session is running", () => {
    let c!: renderer.ReactTestRenderer;
    act(() => {
      c = renderer.create(<LiveTranscript entries={[]} />);
    });
    expect(textOf(byId(c, "live-transcript-empty"))).toMatch(/Waiting for conversation/);
  });
});

describe("journal gate — 'enroll your voice first' now has somewhere to go", () => {
  it("renders a Train my voice button that opens enrollment", () => {
    const onEnroll = jest.fn();
    let c!: renderer.ReactTestRenderer;
    act(() => {
      c = renderer.create(
        <JournalPanel state={IDLE_JOURNAL_STATE} sessionActive={false} gate="missing" onEnroll={onEnroll} />,
      );
    });
    const btn = byId(c, "journal-enroll-button");
    expect(btn).not.toBeNull();
    act(() => btn!.props.onPress());
    expect(onEnroll).toHaveBeenCalledTimes(1);
  });
  it("stays text-only when no enrollment path is wired", () => {
    let c!: renderer.ReactTestRenderer;
    act(() => {
      c = renderer.create(<JournalPanel state={IDLE_JOURNAL_STATE} sessionActive={false} gate="missing" />);
    });
    expect(byId(c, "journal-gate")).not.toBeNull();
    expect(byId(c, "journal-enroll-button")).toBeNull();
  });
});

describe("home recovery card — a crash lands on Home, so Home must mention the saved audio", () => {
  const fakeStore = (recoverable: number, orphans: number) =>
    ({
      listRecoverable: () => Array.from({ length: recoverable }, (_, i) => ({ id: `s${i}` })),
      listOrphanStitched: () => Array.from({ length: orphans }, (_, i) => ({ uri: `o${i}` })),
    }) as never;

  it("renders nothing when there is nothing to recover", () => {
    let c!: renderer.ReactTestRenderer;
    act(() => {
      c = renderer.create(<RecoveryHomeCard store={fakeStore(0, 0)} onOpen={() => {}} />);
    });
    expect(byId(c, "recovery-home-card")).toBeNull();
  });
  it("points at Analyze when an unfinished recording exists", () => {
    const onOpen = jest.fn();
    let c!: renderer.ReactTestRenderer;
    act(() => {
      c = renderer.create(<RecoveryHomeCard store={fakeStore(1, 0)} onOpen={onOpen} />);
    });
    const card = byId(c, "recovery-home-card");
    expect(textOf(card)).toMatch(/An unfinished recording was saved/);
    act(() => card!.props.onPress());
    expect(onOpen).toHaveBeenCalledTimes(1);
  });
  it("counts orphaned stitched files too", () => {
    let c!: renderer.ReactTestRenderer;
    act(() => {
      c = renderer.create(<RecoveryHomeCard store={fakeStore(1, 1)} onOpen={() => {}} />);
    });
    expect(textOf(byId(c, "recovery-home-card"))).toMatch(/2 unfinished recordings/);
  });
});

describe("destinations — 'Voice profile' lands on the Voice section, not the top of Settings", () => {
  it("carries the section anchor", () => {
    const voice = DESTINATIONS.find((d) => d.id === "voiceProfile")!;
    expect(voice.screen).toEqual({ name: "advanced", section: "voice" });
    const settings = DESTINATIONS.find((d) => d.id === "settings")!;
    expect(settings.screen).toEqual({ name: "advanced" });
  });
  it("uses sentence case and says whose dashboard it is", () => {
    const titles = Object.fromEntries(DESTINATIONS.map((d) => [d.id, d.title]));
    expect(titles.analyze).toBe("Analyze a conversation");
    expect(titles.growth).toBe("Your growth");
    expect(titles.therapistDashboard).toBe("Therapist dashboard");
  });
});

describe("showAlert — react-native-web's Alert is a no-op, so web gets a real dialog", () => {
  const original = Platform.OS;
  afterEach(() => {
    Object.defineProperty(Platform, "OS", { value: original, configurable: true });
    delete (globalThis as { alert?: unknown }).alert;
    delete (globalThis as { confirm?: unknown }).confirm;
  });

  it("uses window.alert for a one-button message and still runs its handler", () => {
    Object.defineProperty(Platform, "OS", { value: "web", configurable: true });
    const alertSpy = jest.fn();
    (globalThis as { alert?: unknown }).alert = alertSpy;
    const onPress = jest.fn();
    showAlert("Voice forgotten", "Your voice signature was deleted.", [{ text: "OK", onPress }]);
    expect(alertSpy).toHaveBeenCalledWith("Voice forgotten\n\nYour voice signature was deleted.");
    expect(onPress).toHaveBeenCalledTimes(1);
  });

  it("uses window.confirm for a two-button choice: OK runs the action, Cancel runs cancel", () => {
    Object.defineProperty(Platform, "OS", { value: "web", configurable: true });
    const confirmSpy = jest.fn().mockReturnValueOnce(true).mockReturnValueOnce(false);
    (globalThis as { confirm?: unknown }).confirm = confirmSpy;
    const cancel = jest.fn();
    const forget = jest.fn();
    const buttons = [
      { text: "Cancel", style: "cancel" as const, onPress: cancel },
      { text: "Forget", style: "destructive" as const, onPress: forget },
    ];
    showAlert("Forget this person?", undefined, buttons);
    showAlert("Forget this person?", undefined, buttons);
    expect(forget).toHaveBeenCalledTimes(1);
    expect(cancel).toHaveBeenCalledTimes(1);
  });
});

describe("legal links", () => {
  it("is the same privacy URL the store listings declare", () => {
    expect(PRIVACY_POLICY_URL).toBe("https://arborfam-hub.web.app/privacy");
  });
});
