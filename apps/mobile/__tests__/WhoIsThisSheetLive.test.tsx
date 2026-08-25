/**
 * WhoIsThisSheet in LIVE (mid-call) mode: `onLiveLabel` replaces the
 * stored-recording PATCH — picking a person or typing a new name hands the
 * choice to the running session and shows its outcome; no "Remember this
 * voice" stage (the live flow decides that itself) and no "clear" row.
 */
import React from "react";
import renderer, { act, type ReactTestInstance } from "react-test-renderer";
import WhoIsThisSheet from "../src/components/WhoIsThisSheet";
import { patchSpeakerLabels } from "../src/api/client";
import type { VoicePerson } from "../src/api/client";

jest.mock("../src/api/client", () => ({
  patchSpeakerLabels: jest.fn(),
  enrollPersonFromRecording: jest.fn(),
}));
const mockPatch = patchSpeakerLabels as jest.Mock;

function queryId(comp: renderer.ReactTestRenderer, id: string): ReactTestInstance | null {
  const found = comp.root.findAll((n) => n.props?.testID === id);
  return found.length > 0 ? found[0] : null;
}
function textOf(node: ReactTestInstance | null): string {
  if (!node) return "";
  return node
    .findAll((n) => typeof n.type === "string")
    .flatMap((n) => n.children)
    .filter((c): c is string => typeof c === "string")
    .join("");
}
function person(over: Partial<VoicePerson>): VoicePerson {
  return {
    available: true, storage_enabled: true, enrolled: true, enroll_count: 2,
    person_id: "mom", display_name: "Mom", is_self: false, samples: [], ...over,
  };
}
const PEOPLE = [person({ person_id: "mom", display_name: "Mom" }), person({ person_id: "self", display_name: null, is_self: true })];
const flush = () => act(async () => {
  await Promise.resolve();
  await Promise.resolve();
});

async function mount(onLiveLabel: jest.Mock, over: Partial<React.ComponentProps<typeof WhoIsThisSheet>> = {}) {
  let comp!: renderer.ReactTestRenderer;
  await act(async () => {
    comp = renderer.create(
      <WhoIsThisSheet
        visible
        speaker="Speaker B"
        currentLabel="Mom"
        people={PEOPLE}
        hasAudio={false}
        onClose={jest.fn()}
        onLiveLabel={onLiveLabel}
        {...over}
      />,
    );
  });
  return comp;
}

beforeEach(() => {
  mockPatch.mockReset();
});

describe("WhoIsThisSheet — live mode", () => {
  it("picking an enrolled person hands the choice to the session and shows its outcome", async () => {
    const onLiveLabel = jest.fn().mockResolvedValue({ text: "Mom is labeled for the rest of this call." });
    const comp = await mount(onLiveLabel);
    // No clear row in live mode even though the label differs from the raw id.
    expect(queryId(comp, "who-clear")).toBeNull();
    await act(async () => {
      queryId(comp, "who-person-self")!.props.onPress();
    });
    await flush();
    expect(onLiveLabel).toHaveBeenCalledWith({ personId: "self", displayName: "You", isSelf: true, isNew: false });
    expect(mockPatch).not.toHaveBeenCalled();
    expect(queryId(comp, "who-remember-stage")).toBeNull();
    expect(textOf(queryId(comp, "who-done-text"))).toBe("Mom is labeled for the rest of this call.");
  });

  it("a new name becomes a slugged new person", async () => {
    const onLiveLabel = jest.fn().mockResolvedValue({ text: "ok" });
    const comp = await mount(onLiveLabel);
    await act(async () => {
      queryId(comp, "who-new-person")!.props.onPress();
    });
    await act(async () => {
      queryId(comp, "who-name-input")!.props.onChangeText("Mom");
    });
    await act(async () => {
      queryId(comp, "who-save-name")!.props.onPress();
    });
    await flush();
    // "mom" is taken by the enrolled person → the slug gets a suffix.
    expect(onLiveLabel).toHaveBeenCalledWith({ personId: "mom-2", displayName: "Mom", isSelf: false, isNew: true });
    expect(textOf(queryId(comp, "who-done-text"))).toBe("ok");
  });

  it("a failing live label shows the error inline and keeps the sheet open", async () => {
    const onLiveLabel = jest.fn().mockRejectedValue(new Error("boom"));
    const comp = await mount(onLiveLabel);
    await act(async () => {
      queryId(comp, "who-person-mom")!.props.onPress();
    });
    await flush();
    expect(queryId(comp, "who-error")).not.toBeNull();
    expect(queryId(comp, "who-done-stage")).toBeNull();
  });
});
