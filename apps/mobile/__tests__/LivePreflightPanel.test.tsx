import React from "react";
import renderer, { act } from "react-test-renderer";
import LivePreflightPanel, { describeLlm } from "../src/components/LivePreflightPanel";

/** All rendered text, joined — RN splits interpolated strings into fragments. */
function textOf(root: renderer.ReactTestRenderer): string {
  return root.root
    .findAll((n) => typeof n.type === "string")
    .flatMap((n) => n.children)
    .filter((c): c is string => typeof c === "string")
    .join("");
}

describe("LivePreflightPanel", () => {
  it("describeLlm: first local provider or 'cloud'", () => {
    expect(describeLlm(undefined)).toBe("cloud");
    expect(describeLlm(["cloud"])).toBe("cloud");
    expect(describeLlm(["os", "cloud"])).toBe("os");
    expect(describeLlm(["os", "bundled", "cloud"])).toBe("os → bundled");
  });

  it("not capable: says why, and that the server labels voices", () => {
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(
        <LivePreflightPanel
          liveCapable={false}
          liveCapabilityReason="on-device speech recognition isn't available here"
          liveMode={false}
          preflight={null}
          people={null}
          peopleError={null}
        />,
      );
    });
    const t = textOf(root!);
    expect(t).toContain("on-device speech recognition isn't available here");
    expect(t).toContain("server labels voices by speaking order");
    expect(t).toContain("Loading enrolled people");
  });

  it("probing then ready: reflects the actual capabilities and the reason speaker-ID is off", () => {
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(
        <LivePreflightPanel
          liveCapable
          liveCapabilityReason="ok"
          liveMode
          preflight={{ status: "probing" }}
          people={[]}
          peopleError={null}
        />,
      );
    });
    expect(textOf(root!)).toContain("Loading models");
    expect(root!.root.findByProps({ testID: "whos-here-empty" })).toBeTruthy();
    act(() => {
      root!.update(
        <LivePreflightPanel
          liveCapable
          liveCapabilityReason="ok"
          liveMode
          preflight={{
            status: "ready",
            capabilities: {
              vad: "energy",
              speakerId: { active: true, reason: "model cached", enrolled: 2, model: "cached", droppedForModel: 0 },
              llm: ["cloud"],
            },
          }}
          people={[
            { personId: "self", displayName: "You", isSelf: true, enrollCount: 3 },
            { personId: "mom", displayName: "Mom", isSelf: false, enrollCount: 1 },
          ]}
          peopleError={null}
        />,
      );
    });
    const t = textOf(root!);
    expect(t).toContain("2 enrolled · model cached");
    expect(t).toContain("energy VAD (Silero unavailable)");
    expect(root!.root.findByProps({ testID: "whos-here-mom" })).toBeTruthy();
    // LLM row reads "cloud" when no local provider exists.
    const llm = root!.root.findAll(
      (n) => typeof n.type === "string" && n.props?.testID === "preflight-llm",
    )[0];
    const llmText = llm
      .findAll((n) => typeof n.type === "string")
      .flatMap((n) => n.children)
      .filter((c): c is string => typeof c === "string")
      .join("");
    expect(llmText).toContain("cloud");
  });

  it("a failed probe and a failed people fetch both show their reasons", () => {
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(
        <LivePreflightPanel
          liveCapable
          liveCapabilityReason="ok"
          liveMode
          preflight={{ status: "failed", reason: "ONNX session failed" }}
          people={[]}
          peopleError="not signed in (401)"
        />,
      );
    });
    const t = textOf(root!);
    expect(t).toContain("ONNX session failed");
    expect(t).toContain("not signed in (401)");
  });
});
