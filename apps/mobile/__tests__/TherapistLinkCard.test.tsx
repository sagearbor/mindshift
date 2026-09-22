import React from "react";
import renderer, { act } from "react-test-renderer";
import TherapistLinkCard from "../src/components/TherapistLinkCard";
import {
  getTherapistLink,
  setTherapistLink,
  setAutoShare,
  setTherapistConsent,
  unlinkTherapist,
} from "../src/api/therapist";

/**
 * Unmount every tree this file creates.
 *
 * react-test-renderer keeps a mounted tree scheduled, and a state update that
 * lands AFTER Jest tears the environment down throws "You are trying to
 * `import` a file after the Jest environment has been torn down" — which is
 * not a test failure but a WORKER CRASH, taking every unrelated suite sharing
 * that worker with it. That is why running the whole suite at once used to
 * fail a handful of random files that each passed on their own.
 */
const __trees: renderer.ReactTestRenderer[] = [];
function track<T extends renderer.ReactTestRenderer>(t: T): T {
  __trees.push(t);
  return t;
}
afterEach(async () => {
  await act(async () => {});
  act(() => {
    for (const t of __trees.splice(0)) {
      try {
        t.unmount();
      } catch {
        // A tree a test already unmounted, or one whose teardown throws, must
        // not fail the test that otherwise passed.
      }
    }
  });
});


// The network functions are mocked; `disclosureFor` / `consentGranted` are
// pure readers of the server's reply and stay real, so these tests exercise
// the same parsing the app does.
jest.mock("../src/api/therapist", () => ({
  ...jest.requireActual("../src/api/therapist"),
  getTherapistLink: jest.fn(),
  setTherapistLink: jest.fn(),
  setAutoShare: jest.fn(),
  setTherapistConsent: jest.fn(),
  unlinkTherapist: jest.fn(),
}));
const mockGet = getTherapistLink as jest.Mock;
const mockSet = setTherapistLink as jest.Mock;
const mockAuto = setAutoShare as jest.Mock;
const mockConsent = setTherapistConsent as jest.Mock;
const mockUnlink = unlinkTherapist as jest.Mock;

/** The exact wording server/consent.py ships today. The card must render
 *  THIS — the server's sentence — never a copy of its own. */
const EPISODES_TEXT =
  "Your therapist will see this session's transcript, tone and suggestions — " +
  "including what you could have said. You can turn sharing off, or un-share " +
  "any single session, at any time.";
const LIVE_TEXT =
  "Your therapist can join your calls and watch the transcript, tone and " +
  "coaching as it happens. Everyone on the call is asked before she can listen.";

/** `GET /therapist/link`'s consent block. */
function consentBlock(granted: { episodes?: boolean; live?: boolean } = {}) {
  return {
    text_version: "2026-08-25",
    scopes: {
      episodes: {
        granted: granted.episodes === true,
        at: granted.episodes ? "2026-08-25T10:00:00+00:00" : null,
        granted_text_version: granted.episodes ? "2026-08-25" : null,
        disclosure: EPISODES_TEXT,
      },
      live: {
        granted: granted.live === true,
        at: granted.live ? "2026-08-26T10:00:00+00:00" : null,
        granted_text_version: granted.live ? "2026-08-25" : null,
        disclosure: LIVE_TEXT,
      },
    },
  };
}
const UNLINKED = { linked: false, consent: consentBlock() };
const LINKED = {
  linked: true,
  therapist_email: "mom@example.com",
  status: "pending",
  auto_share: true,
  consent: consentBlock({ episodes: true }),
};

const flush = () => act(async () => { await Promise.resolve(); });
/** All rendered text, joined — RN splits interpolated strings into fragments. */
const text = (root: renderer.ReactTestRenderer) =>
  root.root
    .findAll((n) => typeof n.type === "string")
    .flatMap((n) => n.children)
    .filter((c): c is string => typeof c === "string")
    .join("");

beforeEach(() => {
  mockGet.mockReset();
  mockSet.mockReset();
  mockAuto.mockReset();
  mockConsent.mockReset();
  mockUnlink.mockReset();
});

describe("TherapistLinkCard", () => {
  it("unlinked: explains, links by email (pending, auto-share on), then offers the switch + unlink", async () => {
    mockGet.mockResolvedValue(UNLINKED);
    mockSet.mockResolvedValue(LINKED);
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = track(renderer.create(<TherapistLinkCard />));
    });
    await flush();
    expect(text(root!)).toContain("Enter your therapist’s MindShift account email");
    const submit = root!.root.findByProps({ testID: "therapist-link-submit" });
    expect(submit.props.disabled).toBe(true); // nothing typed yet
    act(() => {
      root!.root.findByProps({ testID: "therapist-email-input" }).props.onChangeText(" mom@example.com ");
    });
    await act(async () => {
      await root!.root.findByProps({ testID: "therapist-link-submit" }).props.onPress();
    });
    expect(mockSet).toHaveBeenCalledWith("mom@example.com");
    const t = text(root!);
    expect(t).toContain("Linked to mom@example.com");
    expect(t).toContain("waiting for them to accept");
    expect(root!.root.findByProps({ testID: "therapist-auto-share" }).props.value).toBe(true);
    expect(root!.root.findByProps({ testID: "therapist-unlink" })).toBeTruthy();
  });

  it("surfaces the server's detail when the email has no account", async () => {
    mockGet.mockResolvedValue(UNLINKED);
    mockSet.mockRejectedValue(Object.assign(new Error("x"), { status: 404, detail: "no MindShift account with that email" }));
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = track(renderer.create(<TherapistLinkCard />));
    });
    await flush();
    act(() => {
      root!.root.findByProps({ testID: "therapist-email-input" }).props.onChangeText("nobody@example.com");
    });
    await act(async () => {
      await root!.root.findByProps({ testID: "therapist-link-submit" }).props.onPress();
    });
    expect(text(root!)).toContain("no MindShift account with that email");
    expect(root!.root.findByProps({ testID: "therapist-email-input" })).toBeTruthy(); // still unlinked
  });

  it("linked + accepted: toggles auto-share optimistically (rolled back on failure) and unlinks", async () => {
    mockGet.mockResolvedValue({ ...LINKED, status: "accepted" });
    mockAuto.mockResolvedValueOnce({ ...LINKED, status: "accepted", auto_share: false });
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = track(renderer.create(<TherapistLinkCard />));
    });
    await flush();
    expect(text(root!)).toContain("· accepted");
    await act(async () => {
      await root!.root.findByProps({ testID: "therapist-auto-share" }).props.onValueChange(false);
    });
    expect(mockAuto).toHaveBeenCalledWith(false);
    expect(root!.root.findByProps({ testID: "therapist-auto-share" }).props.value).toBe(false);

    mockAuto.mockRejectedValueOnce(Object.assign(new Error("x"), { status: 503 }));
    await act(async () => {
      await root!.root.findByProps({ testID: "therapist-auto-share" }).props.onValueChange(true);
    });
    expect(root!.root.findByProps({ testID: "therapist-auto-share" }).props.value).toBe(false); // rolled back
    expect(text(root!)).toContain("Linking isn’t available right now.");

    mockUnlink.mockResolvedValue(undefined);
    await act(async () => {
      await root!.root.findByProps({ testID: "therapist-unlink" }).props.onPress();
    });
    expect(mockUnlink).toHaveBeenCalled();
    expect(root!.root.findByProps({ testID: "therapist-email-input" })).toBeTruthy();
  });

  it("says when the link couldn't be loaded instead of guessing", async () => {
    mockGet.mockRejectedValue(Object.assign(new Error("x"), { status: 401 }));
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = track(renderer.create(<TherapistLinkCard />));
    });
    await flush();
    expect(text(root!)).toContain("Couldn’t load your therapist link (Please sign in again.)");
  });
});

/**
 * Consent integrity (server/consent.py). PUT /therapist/link records an
 * `episodes` consent stamped with a `text_version`. These tests are the
 * proof that the patient was actually shown the wording that record cites
 * — and that the card refuses to take the consent when it wasn't.
 */
describe("TherapistLinkCard — the disclosure", () => {
  const render = async () => {
    let root!: renderer.ReactTestRenderer;
    act(() => {
      root = track(renderer.create(<TherapistLinkCard />));
    });
    await flush();
    return root;
  };

  it("shows the server's episodes wording BEFORE the patient submits", async () => {
    mockGet.mockResolvedValue(UNLINKED);
    mockSet.mockResolvedValue(LINKED);
    const root = await render();
    // On screen, verbatim, above the field — not after the tap.
    expect(
      root.root.findByProps({ testID: "therapist-disclosure-episodes" }).props.children,
    ).toBe(EPISODES_TEXT);
    expect(text(root)).toContain("By linking them you agree:");
    expect(mockSet).not.toHaveBeenCalled();
  });

  it("renders the SERVER's sentence, not a copy of its own", async () => {
    // Reword the disclosure server-side: the card must follow it exactly.
    const reworded = "Reworded 2027: your therapist reads everything you say here.";
    mockGet.mockResolvedValue({
      linked: false,
      consent: {
        text_version: "2027-01-01",
        scopes: {
          episodes: { granted: false, disclosure: reworded },
          live: { granted: false, disclosure: LIVE_TEXT },
        },
      },
    });
    const root = await render();
    expect(
      root.root.findByProps({ testID: "therapist-disclosure-episodes" }).props.children,
    ).toBe(reworded);
    expect(text(root)).not.toContain(EPISODES_TEXT);
  });

  it("refuses to link at all when the server sent no disclosure", async () => {
    // An older server: no consent block. Linking would write a record
    // citing wording nobody saw, so the button stays dead and says why.
    mockGet.mockResolvedValue({ linked: false });
    const root = await render();
    expect(root.root.findAllByProps({ testID: "therapist-disclosure" })).toHaveLength(0);
    expect(text(root)).toContain("Couldn’t load what you’d be agreeing to");
    act(() => {
      root.root.findByProps({ testID: "therapist-email-input" }).props.onChangeText("mom@example.com");
    });
    const submit = root.root.findByProps({ testID: "therapist-link-submit" });
    expect(submit.props.disabled).toBe(true);
    // Even forced (a keyboard "go"), the handler refuses.
    await act(async () => {
      await submit.props.onPress();
    });
    expect(mockSet).not.toHaveBeenCalled();
  });

  it("keeps the wording readable after linking, with what was agreed", async () => {
    mockGet.mockResolvedValue(LINKED);
    const root = await render();
    expect(text(root)).toContain("You agreed:");
    expect(
      root.root.findByProps({ testID: "therapist-disclosure-episodes" }).props.children,
    ).toBe(EPISODES_TEXT);
  });

  it("`live` is a separate, explicit, default-off agreement with its own text", async () => {
    mockGet.mockResolvedValue(LINKED);
    mockConsent.mockResolvedValue({ ...LINKED, consent: consentBlock({ episodes: true, live: true }) });
    const root = await render();
    const toggle = root.root.findByProps({ testID: "therapist-live-consent" });
    // Naming a therapist did NOT hand over live listening.
    expect(toggle.props.value).toBe(false);
    expect(
      root.root.findByProps({ testID: "therapist-disclosure-live" }).props.children,
    ).toBe(LIVE_TEXT);
    await act(async () => {
      await toggle.props.onValueChange(true);
    });
    expect(mockConsent).toHaveBeenCalledWith("live", true);
    expect(root.root.findByProps({ testID: "therapist-live-consent" }).props.value).toBe(true);

    // And it can be taken back.
    mockConsent.mockResolvedValueOnce({ ...LINKED, consent: consentBlock({ episodes: true }) });
    await act(async () => {
      await root.root.findByProps({ testID: "therapist-live-consent" }).props.onValueChange(false);
    });
    expect(mockConsent).toHaveBeenLastCalledWith("live", false);
    expect(root.root.findByProps({ testID: "therapist-live-consent" }).props.value).toBe(false);
  });

  it("a failed live grant is reported, and the switch does not lie", async () => {
    mockGet.mockResolvedValue(LINKED);
    mockConsent.mockRejectedValue(Object.assign(new Error("x"), { status: 503 }));
    const root = await render();
    await act(async () => {
      await root.root.findByProps({ testID: "therapist-live-consent" }).props.onValueChange(true);
    });
    // Never optimistic: consent is only ever what the server stored.
    expect(root.root.findByProps({ testID: "therapist-live-consent" }).props.value).toBe(false);
    expect(text(root)).toContain("Linking isn’t available right now.");
  });
});
