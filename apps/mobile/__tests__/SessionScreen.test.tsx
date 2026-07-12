import React from "react";
import renderer, { act } from "react-test-renderer";
import SessionScreen from "../src/screens/SessionScreen";
import { useSessionStore } from "../src/store/sessionStore";

// Reset store between tests
beforeEach(() => {
  act(() => {
    useSessionStore.setState({
      role: "Husband / Wife",
      empathyLevel: 50,
      turns: [],
      suggestions: [],
      loading: false,
    });
  });
});

describe("SessionScreen", () => {
  it("renders the initial screen", () => {
    let component: renderer.ReactTestRenderer;
    act(() => {
      component = renderer.create(<SessionScreen />);
    });
    expect(component!.toJSON()).toMatchSnapshot();
  });

  it("renders with turns in the transcript", () => {
    act(() => {
      useSessionStore.setState({
        turns: [
          { speaker: "Alice", text: "I feel like you never listen to me." },
          { speaker: "Bob", text: "That's not fair, I always try." },
        ],
      });
    });

    let component: renderer.ReactTestRenderer;
    act(() => {
      component = renderer.create(<SessionScreen />);
    });
    expect(component!.toJSON()).toMatchSnapshot();
  });

  it("renders with suggestions", () => {
    act(() => {
      useSessionStore.setState({
        turns: [
          { speaker: "Alice", text: "You never help around the house." },
        ],
        suggestions: [
          {
            text: "I hear that you're feeling overwhelmed with housework.",
            tone: "empathetic",
          },
          {
            text: "Let's talk about how we can split things more evenly.",
            tone: "balanced",
          },
          {
            text: "I understand. What would help most right now?",
            tone: "validating",
          },
        ],
      });
    });

    let component: renderer.ReactTestRenderer;
    act(() => {
      component = renderer.create(<SessionScreen />);
    });
    expect(component!.toJSON()).toMatchSnapshot();
  });

  it("shows the analyze-dynamics button only at >= 4 turns", () => {
    const mkTurns = (n: number) =>
      Array.from({ length: n }, (_, i) => ({
        speaker: i % 2 === 0 ? "Alice" : "Bob",
        text: `turn ${i}`,
      }));
    const hasButton = (comp: renderer.ReactTestRenderer) =>
      comp.root.findAll((x) => x.props?.testID === "analyze-dynamics-button")
        .length > 0;

    // 3 turns: below threshold, hidden even with a handler.
    act(() => {
      useSessionStore.setState({ turns: mkTurns(3) });
    });
    let three!: renderer.ReactTestRenderer;
    act(() => {
      three = renderer.create(<SessionScreen onAnalyzeDynamics={() => {}} />);
    });
    expect(hasButton(three)).toBe(false);

    // 4 turns: shown, and pressing it invokes the handler.
    act(() => {
      useSessionStore.setState({ turns: mkTurns(4) });
    });
    const onAnalyze = jest.fn();
    let four!: renderer.ReactTestRenderer;
    act(() => {
      four = renderer.create(<SessionScreen onAnalyzeDynamics={onAnalyze} />);
    });
    expect(hasButton(four)).toBe(true);
    act(() => {
      four.root
        .find((x) => x.props?.testID === "analyze-dynamics-button")
        .props.onPress();
    });
    expect(onAnalyze).toHaveBeenCalledTimes(1);
  });

  it("renders loading state", () => {
    act(() => {
      useSessionStore.setState({
        turns: [{ speaker: "Alice", text: "We need to talk." }],
        loading: true,
      });
    });

    let component: renderer.ReactTestRenderer;
    act(() => {
      component = renderer.create(<SessionScreen />);
    });
    expect(component!.toJSON()).toMatchSnapshot();
  });
});
