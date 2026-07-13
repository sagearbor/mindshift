import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import HomeScreen from "../src/screens/HomeScreen";

function queryId(
  comp: renderer.ReactTestRenderer,
  id: string,
): ReactTestInstance | null {
  const found = comp.root.findAll((n) => n.props?.testID === id);
  return found.length > 0 ? found[0] : null;
}

function makeHandlers() {
  return {
    onLiveCoach: jest.fn(),
    onAnalyze: jest.fn(),
    onOpenRecordings: jest.fn(),
    onOpenAdvanced: jest.fn(),
  };
}

describe("HomeScreen", () => {
  it("renders the two primary modes, the history entry, and the advanced corner", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<HomeScreen {...makeHandlers()} />);
    });
    expect(queryId(comp, "home-live-coach")).toBeTruthy();
    expect(queryId(comp, "home-analyze")).toBeTruthy();
    expect(queryId(comp, "home-recordings-link")).toBeTruthy();
    expect(queryId(comp, "home-advanced-button")).toBeTruthy();
    expect(comp.toJSON()).toMatchSnapshot();
    act(() => comp.unmount());
  });

  it("each tap target calls exactly its own handler", () => {
    const handlers = makeHandlers();
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<HomeScreen {...handlers} />);
    });

    act(() => queryId(comp, "home-live-coach")!.props.onPress());
    expect(handlers.onLiveCoach).toHaveBeenCalledTimes(1);

    act(() => queryId(comp, "home-analyze")!.props.onPress());
    expect(handlers.onAnalyze).toHaveBeenCalledTimes(1);

    act(() => queryId(comp, "home-recordings-link")!.props.onPress());
    expect(handlers.onOpenRecordings).toHaveBeenCalledTimes(1);

    act(() => queryId(comp, "home-advanced-button")!.props.onPress());
    expect(handlers.onOpenAdvanced).toHaveBeenCalledTimes(1);

    // No cross-talk.
    expect(handlers.onLiveCoach).toHaveBeenCalledTimes(1);
    expect(handlers.onAnalyze).toHaveBeenCalledTimes(1);
    expect(handlers.onOpenRecordings).toHaveBeenCalledTimes(1);
    act(() => comp.unmount());
  });
});
