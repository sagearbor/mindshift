import React from "react";
import renderer, { act } from "react-test-renderer";
import TherapistLinkCard from "../src/components/TherapistLinkCard";
import {
  getTherapistLink,
  setTherapistLink,
  setAutoShare,
  unlinkTherapist,
} from "../src/api/therapist";

jest.mock("../src/api/therapist", () => ({
  getTherapistLink: jest.fn(),
  setTherapistLink: jest.fn(),
  setAutoShare: jest.fn(),
  unlinkTherapist: jest.fn(),
}));
const mockGet = getTherapistLink as jest.Mock;
const mockSet = setTherapistLink as jest.Mock;
const mockAuto = setAutoShare as jest.Mock;
const mockUnlink = unlinkTherapist as jest.Mock;

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
  mockUnlink.mockReset();
});

describe("TherapistLinkCard", () => {
  it("unlinked: explains, links by email (pending, auto-share on), then offers the switch + unlink", async () => {
    mockGet.mockResolvedValue({ linked: false });
    mockSet.mockResolvedValue({ linked: true, therapist_email: "mom@example.com", status: "pending", auto_share: true });
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(<TherapistLinkCard />);
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
    mockGet.mockResolvedValue({ linked: false });
    mockSet.mockRejectedValue(Object.assign(new Error("x"), { status: 404, detail: "no MindShift account with that email" }));
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(<TherapistLinkCard />);
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
    mockGet.mockResolvedValue({ linked: true, therapist_email: "mom@example.com", status: "accepted", auto_share: true });
    mockAuto.mockResolvedValueOnce({ linked: true, therapist_email: "mom@example.com", status: "accepted", auto_share: false });
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(<TherapistLinkCard />);
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
      root = renderer.create(<TherapistLinkCard />);
    });
    await flush();
    expect(text(root!)).toContain("Couldn’t load your therapist link (Please sign in again.)");
  });
});
