import React from "react";
import renderer, { act } from "react-test-renderer";
import ConversationContextPanel from "../src/components/ConversationContextPanel";
import {
  clampSessionContext,
  createFsSessionContextStore,
  onDeviceSessionContext,
  ON_DEVICE_CONTEXT_MAX_CHARS,
  SESSION_CONTEXT_MAX_CHARS,
} from "../src/live/sessionContext";
import { buildPrompt } from "../src/live/localLlm";
import { MemoryFs } from "../src/recorder/memoryFs";

describe("sessionContext helpers", () => {
  it("caps at 4,000 characters", () => {
    expect(SESSION_CONTEXT_MAX_CHARS).toBe(4000);
    expect(clampSessionContext("a".repeat(4001))).toHaveLength(4000);
    expect(clampSessionContext("short")).toBe("short");
  });

  it("trims for the on-device prompt at a word boundary, within its budget", () => {
    expect(onDeviceSessionContext("")).toBeNull();
    expect(onDeviceSessionContext("  \n ")).toBeNull();
    expect(onDeviceSessionContext("Raise talk\nwith my boss")).toBe("Raise talk with my boss");
    const long = Array.from({ length: 400 }, (_, i) => `word${i}`).join(" ");
    const t = onDeviceSessionContext(long)!;
    expect(t.length).toBeLessThanOrEqual(ON_DEVICE_CONTEXT_MAX_CHARS);
    expect(t.endsWith("…")).toBe(true);
    // Cut after a whole word, and a prefix of the original.
    expect(t.slice(0, -1)).toMatch(/word\d+$/);
    expect(long.startsWith(t.slice(0, -1))).toBe(true);
  });

  it("the on-device prompt carries it only when present", () => {
    const base = { text: "no", speaker: "Speaker B", isSelf: false, empathy: 50, context: [], mode: "earpiece" as const };
    expect(buildPrompt(base).user).not.toMatch(/Background/);
    expect(buildPrompt({ ...base, sessionContext: "raise talk" }).user).toMatch(/^Background from the coached person: "raise talk"/);
  });

  it("is remembered on the device and cleared in one go", () => {
    const fs = new MemoryFs();
    const uri = `${fs.documentDirUri()}/live-session-context.txt`;
    const store = createFsSessionContextStore(fs, uri);
    expect(store.load()).toBe("");
    store.save("Raise talk");
    expect(createFsSessionContextStore(fs, uri).load()).toBe("Raise talk");
    store.save("y".repeat(5000));
    expect(store.load()).toHaveLength(4000);
    store.clear();
    expect(store.load()).toBe("");
    expect(fs.exists(uri)).toBe(false);
  });
});

describe("ConversationContextPanel", () => {
  const render = (props: Partial<React.ComponentProps<typeof ConversationContextPanel>> = {}) => {
    const onSessionContextChange = jest.fn();
    const onClearSessionContext = jest.fn();
    let root!: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(
        <ConversationContextPanel
          relationship={null}
          onRelationshipChange={jest.fn()}
          sessionContext=""
          onSessionContextChange={onSessionContextChange}
          onClearSessionContext={onClearSessionContext}
          {...props}
        />,
      );
    });
    return { root, onSessionContextChange, onClearSessionContext };
  };
  const counterText = (root: renderer.ReactTestRenderer) =>
    [root.root.findAllByProps({ testID: "session-context-counter" })[0].props.children].flat().join("");

  it("shows a counter, enforces the cap while typing, and offers Clear only when there is text", () => {
    const empty = render();
    expect(counterText(empty.root)).toBe("0 / 4000");
    expect(empty.root.root.findAllByProps({ testID: "session-context-clear" })).toHaveLength(0);
    const input = empty.root.root.findByProps({ testID: "session-context-input" });
    expect(input.props.maxLength).toBe(4000);
    act(() => input.props.onChangeText("z".repeat(4500)));
    expect(empty.onSessionContextChange.mock.calls[0][0]).toHaveLength(4000);

    const filled = render({ sessionContext: "Raise talk" });
    expect(counterText(filled.root)).toBe("10 / 4000");
    act(() => filled.root.root.findByProps({ testID: "session-context-clear" }).props.onPress());
    expect(filled.onClearSessionContext).toHaveBeenCalled();
  });
});
