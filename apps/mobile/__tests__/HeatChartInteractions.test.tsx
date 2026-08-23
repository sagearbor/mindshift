import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import HeatChart from "../src/components/HeatChart";
import type { AnalyzePerTurn } from "../src/api/client";

/**
 * Interaction tests for the heat chart's TIME-AXIS ZOOM RESET (bug #10: "once
 * zoomed there's no way back") and the LEGEND SPEAKER ISOLATION (the audition /
 * voice-library seam). Gestures are driven through the PanResponder's public
 * responder props with fabricated touch events — the same seam the RN gesture
 * system uses — so the double-tap and pinch paths are exercised for real.
 */

const perTurn: AnalyzePerTurn[] = [
  { index: 0, speaker: "Alice", heat: 20, markers: [], is_spike: false, trigger_phrase: null },
  { index: 1, speaker: "Bob", heat: 30, markers: [], is_spike: false, trigger_phrase: null },
  { index: 2, speaker: "Alice", heat: 45, markers: [], is_spike: false, trigger_phrase: null },
  { index: 3, speaker: "Bob", heat: 90, markers: [], is_spike: true, trigger_phrase: null },
];

const timing = [
  { start_time: 0, end_time: 2 },
  { start_time: 2, end_time: 3 },
  { start_time: 3, end_time: 9 },
  { start_time: 9, end_time: 10 },
];

const DURATION = 10;

function render(extraProps: Record<string, unknown> = {}) {
  let comp!: renderer.ReactTestRenderer;
  act(() => {
    comp = renderer.create(
      <HeatChart
        perTurn={perTurn}
        turns={perTurn.map((t) => ({ speaker: t.speaker, text: `t${t.index}` }))}
        turnsTiming={timing}
        durationSeconds={DURATION}
        {...extraProps}
      />,
    );
  });
  return comp;
}

function queryId(
  comp: renderer.ReactTestRenderer,
  id: string,
): ReactTestInstance | null {
  const found = comp.root.findAll((n) => n.props?.testID === id);
  return found.length > 0 ? found[0] : null;
}

/** First node with this testID that carries a callable onPress. */
function pressable(comp: renderer.ReactTestRenderer, id: string): ReactTestInstance {
  return comp.root.findAll(
    (n) => n.props?.testID === id && typeof n.props.onPress === "function",
  )[0];
}

/** Fire the measured layout so the SVG geometry (and zoom) activates. */
function layout(comp: renderer.ReactTestRenderer, width = 300) {
  const node = comp.root.findAll((n) => typeof n.props?.onLayout === "function")[0];
  act(() => {
    node.props.onLayout({ nativeEvent: { layout: { width, height: 180 } } });
  });
}

/** The gesture surface that carries the PanResponder handlers. */
function surfaceOf(comp: renderer.ReactTestRenderer): ReactTestInstance {
  return comp.root.findAll(
    (n) =>
      n.props?.testID === "heat-chart-surface" &&
      typeof n.props.onStartShouldSetResponderCapture === "function",
  )[0];
}

// Fabricated responder events. PanResponder's wrappers read `touchHistory`
// (centroid math over the touchBank) and bail on repeated timestamps, so each
// event carries a populated touchBank and a fresh mostRecentTimeStamp.
let ts = 0;
function touchEvt(touches: { locationX: number; locationY: number }[]) {
  ts += 1;
  return {
    nativeEvent: { touches },
    touchHistory: {
      touchBank: touches.map((touch) => ({
        touchActive: true,
        startPageX: touch.locationX,
        startPageY: touch.locationY,
        startTimeStamp: ts,
        currentPageX: touch.locationX,
        currentPageY: touch.locationY,
        currentTimeStamp: ts,
        previousPageX: touch.locationX,
        previousPageY: touch.locationY,
        previousTimeStamp: ts,
      })),
      numberActiveTouches: touches.length,
      indexOfSingleActiveTouch: 0,
      mostRecentTimeStamp: ts,
    },
  };
}

const t = (x: number) => ({ locationX: x, locationY: 60 });

/** Two-finger pinch-out about the chart middle: spread 100px → 180px, i.e. a
 *  1.8× zoom-in of the time window. */
function pinchZoomIn(comp: renderer.ReactTestRenderer) {
  const surface = surfaceOf(comp);
  act(() => {
    const start = touchEvt([t(100), t(200)]);
    if (surface.props.onStartShouldSetResponderCapture(start)) {
      surface.props.onResponderGrant(start);
    }
    surface.props.onResponderMove(touchEvt([t(60), t(240)]));
    surface.props.onResponderRelease(touchEvt([]));
  });
}

describe("HeatChart zoom reset (bug #10)", () => {
  it("shows no reset affordance in the full (unzoomed) view", () => {
    const comp = render();
    layout(comp);
    expect(queryId(comp, "chart-zoom-reset")).toBeNull();
    expect(queryId(comp, "heat-scrubber")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("pinch-zooming reveals the reset affordance; pressing it restores the full view", () => {
    const comp = render();
    layout(comp);

    pinchZoomIn(comp);
    expect(queryId(comp, "chart-zoom-reset")).toBeTruthy();
    expect(queryId(comp, "heat-zoom-hint")).toBeTruthy();
    expect(queryId(comp, "heat-scrubber")).toBeNull(); // hidden while zoomed

    act(() => pressable(comp, "chart-zoom-reset").props.onPress());
    expect(queryId(comp, "chart-zoom-reset")).toBeNull();
    expect(queryId(comp, "heat-zoom-hint")).toBeNull();
    expect(queryId(comp, "heat-scrubber")).toBeTruthy(); // full view is back
    act(() => comp.unmount());
  });

  it("the reset control lives OUTSIDE the pan-gesture surface, so its tap can't be stolen by the responder", () => {
    const comp = render();
    layout(comp);
    pinchZoomIn(comp);

    const surface = surfaceOf(comp);
    const chip = pressable(comp, "chart-zoom-reset");
    // The chip must not be a descendant of the surface that owns the
    // PanResponder — a finger wobble >6px during the chip press would otherwise
    // convert the tap into a pan and cancel the press (the device-reported
    // "can't get back" trap).
    let node: ReactTestInstance | null = chip;
    let insideSurface = false;
    while (node) {
      if (node === surface) insideSurface = true;
      node = node.parent as ReactTestInstance | null;
    }
    expect(insideSurface).toBe(false);
    act(() => comp.unmount());
  });

  it("double-tap on the chart surface resets the zoom (capture path)", () => {
    const comp = render();
    layout(comp);
    pinchZoomIn(comp);
    expect(queryId(comp, "chart-zoom-reset")).toBeTruthy();

    const surface = surfaceOf(comp);
    act(() => {
      // First tap: recorded, not claimed (falls through to dash presses).
      expect(surface.props.onStartShouldSetResponderCapture(touchEvt([t(150)]))).toBe(false);
      // Second tap within the double-tap window: claimed → reset on grant.
      const second = touchEvt([t(150)]);
      expect(surface.props.onStartShouldSetResponderCapture(second)).toBe(true);
      surface.props.onResponderGrant(second);
      surface.props.onResponderRelease(touchEvt([]));
    });

    expect(queryId(comp, "chart-zoom-reset")).toBeNull();
    expect(queryId(comp, "heat-scrubber")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("double-tap on a DASH also resets while zoomed (belt-and-suspenders for SVG touch handling)", () => {
    const comp = render();
    layout(comp);
    pinchZoomIn(comp);

    // On some devices the SVG's own touch handling can swallow the surface's
    // capture negotiation — so a rapid second tap landing on a dash must reset
    // through the dash-press path too.
    const dash = pressable(comp, "heat-dash-0");
    act(() => {
      dash.props.onPress();
      dash.props.onPress();
    });
    expect(queryId(comp, "chart-zoom-reset")).toBeNull();
    expect(queryId(comp, "heat-scrubber")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("a SINGLE tap on a dash while zoomed still selects the turn (no reset)", () => {
    const comp = render();
    layout(comp);
    pinchZoomIn(comp);

    act(() => pressable(comp, "heat-dash-0").props.onPress());
    expect(queryId(comp, "turn-inspector")).toBeTruthy();
    expect(queryId(comp, "chart-zoom-reset")).toBeTruthy(); // still zoomed
    act(() => comp.unmount());
  });

  it("rapid taps on dashes while UNZOOMED never trigger a reset-select regression (both select)", () => {
    const comp = render();
    layout(comp);

    const dash0 = pressable(comp, "heat-dash-0");
    const dash1 = pressable(comp, "heat-dash-1");
    act(() => {
      dash0.props.onPress();
      dash1.props.onPress();
    });
    // Second tap selected turn 1 (Bob, "t1") — nothing swallowed it.
    const inspector = queryId(comp, "turn-inspector");
    expect(inspector).toBeTruthy();
    expect(JSON.stringify(comp.toJSON())).toContain("t1");
    act(() => comp.unmount());
  });
});

describe("HeatChart legend speaker isolation", () => {
  it("tapping a legend speaker isolates it: others dim, an All chip appears", () => {
    const comp = render();
    layout(comp);

    act(() => pressable(comp, "legend-item-Alice").props.onPress());

    // Alice's dashes stay at full strength; Bob's dim way down.
    const aliceDash = queryId(comp, "heat-dash-0")!;
    const bobDash = queryId(comp, "heat-dash-1")!;
    expect(aliceDash.props.strokeOpacity).toBe(1);
    expect(bobDash.props.strokeOpacity).toBeLessThan(0.3);
    // Bob's spike marker dims with him.
    const bobSpike = queryId(comp, "heat-spike-3")!;
    expect(bobSpike.props.fillOpacity).toBeLessThan(0.3);
    // The way back is explicit.
    expect(queryId(comp, "legend-all")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("tapping the isolated speaker again returns to All (nothing dimmed)", () => {
    const comp = render();
    layout(comp);

    act(() => pressable(comp, "legend-item-Alice").props.onPress());
    act(() => pressable(comp, "legend-item-Alice").props.onPress());

    expect(queryId(comp, "legend-all")).toBeNull();
    expect(queryId(comp, "heat-dash-1")!.props.strokeOpacity).toBe(1);
    act(() => comp.unmount());
  });

  it("tapping another speaker switches the isolation directly", () => {
    const comp = render();
    layout(comp);

    act(() => pressable(comp, "legend-item-Alice").props.onPress());
    act(() => pressable(comp, "legend-item-Bob").props.onPress());

    expect(queryId(comp, "heat-dash-1")!.props.strokeOpacity).toBe(1); // Bob full
    expect(queryId(comp, "heat-dash-0")!.props.strokeOpacity).toBeLessThan(0.3); // Alice dim
    act(() => comp.unmount());
  });

  it("the All chip clears the isolation", () => {
    const comp = render();
    layout(comp);

    act(() => pressable(comp, "legend-item-Bob").props.onPress());
    act(() => pressable(comp, "legend-all").props.onPress());

    expect(queryId(comp, "legend-all")).toBeNull();
    expect(queryId(comp, "heat-dash-0")!.props.strokeOpacity).toBe(1);
    expect(queryId(comp, "heat-dash-1")!.props.strokeOpacity).toBe(1);
    act(() => comp.unmount());
  });

  it("reports isolation changes through onIsolateSpeaker (the naming-flow seam)", () => {
    const onIsolate = jest.fn();
    const comp = render({ onIsolateSpeaker: onIsolate });
    layout(comp);

    act(() => pressable(comp, "legend-item-Alice").props.onPress());
    expect(onIsolate).toHaveBeenLastCalledWith("Alice");
    act(() => pressable(comp, "legend-item-Bob").props.onPress());
    expect(onIsolate).toHaveBeenLastCalledWith("Bob");
    act(() => pressable(comp, "legend-item-Bob").props.onPress());
    expect(onIsolate).toHaveBeenLastCalledWith(null);
    act(() => comp.unmount());
  });

  it("supports CONTROLLED isolation via the isolatedSpeaker prop (parent owns the state)", () => {
    const onIsolate = jest.fn();
    const comp = render({ isolatedSpeaker: "Bob", onIsolateSpeaker: onIsolate });
    layout(comp);

    // Renders the prop's isolation…
    expect(queryId(comp, "heat-dash-0")!.props.strokeOpacity).toBeLessThan(0.3);
    expect(queryId(comp, "heat-dash-1")!.props.strokeOpacity).toBe(1);

    // …and taps only REQUEST a change; the value doesn't move until the parent
    // re-renders with a new prop.
    act(() => pressable(comp, "legend-item-Alice").props.onPress());
    expect(onIsolate).toHaveBeenLastCalledWith("Alice");
    expect(queryId(comp, "heat-dash-0")!.props.strokeOpacity).toBeLessThan(0.3);
    act(() => comp.unmount());
  });

  it("isolation also dims the legacy (no-timing) polylines and points", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <HeatChart
          perTurn={perTurn}
          turns={perTurn.map((p) => ({ speaker: p.speaker, text: "" }))}
        />,
      );
    });
    layout(comp);

    act(() => pressable(comp, "legend-item-Alice").props.onPress());
    const bobLine = queryId(comp, "heat-line-Bob")!;
    expect(bobLine.props.strokeOpacity).toBeLessThan(0.3);
    const bobPoint = queryId(comp, "heat-point-1")!;
    expect(bobPoint.props.fillOpacity).toBeLessThan(0.3);
    act(() => comp.unmount());
  });
});

describe("HeatChart PanResponder — stuck-gesture investigation (owner report: unresponsive until app restart)", () => {
  // Traced every exit path of the gesture state (gestureRef.mode/startDist/
  // focusSec, lastTapRef) across Grant, Move, Release, and Terminate — all
  // reset correctly on the happy path. The one gap: beginPinch() mutates
  // gestureRef.mode as soon as this surface merely EXPRESSES interest in
  // becoming the responder (returning true from a "should set" callback),
  // before a transfer is confirmed. If another view already holds the
  // responder and refuses to yield, the negotiation is REJECTED rather than
  // granted, and neither Release nor Terminate ever fires for that attempt.
  // onPanResponderReject is the system's notification of exactly that outcome
  // — this test proves it's wired and the surface stays fully usable
  // afterward (a rejected claim must never leave the chart inert).
  it("stays fully usable after a REJECTED responder claim — no swallowed input", () => {
    const comp = render();
    layout(comp);
    const surface = surfaceOf(comp);

    act(() => {
      // A two-finger touch begins a pinch bid (mutates internal gesture mode)
      // …
      const start = touchEvt([t(100), t(200)]);
      expect(surface.props.onStartShouldSetResponderCapture(start)).toBe(true);
      // … but instead of being granted, it's rejected (another responder won
      // the negotiation). onResponderReject must exist and not throw.
      expect(typeof surface.props.onResponderReject).toBe("function");
      expect(() => surface.props.onResponderReject(start)).not.toThrow();
    });

    // A plain single tap on a dash still selects the turn — the chart isn't
    // stuck swallowing input after the lost negotiation.
    act(() => pressable(comp, "heat-dash-0").props.onPress());
    expect(queryId(comp, "turn-inspector")).toBeTruthy();

    // A fresh pinch still works too — the surface can still legitimately win
    // a NEW negotiation afterward.
    pinchZoomIn(comp);
    expect(queryId(comp, "chart-zoom-reset")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("never blocks a sibling's termination request (explicit onPanResponderTerminationRequest)", () => {
    const comp = render();
    layout(comp);
    const surface = surfaceOf(comp);
    expect(typeof surface.props.onResponderTerminationRequest).toBe("function");
    expect(surface.props.onResponderTerminationRequest()).toBe(true);
    act(() => comp.unmount());
  });
});
