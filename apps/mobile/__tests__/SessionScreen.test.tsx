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
    let tree: renderer.ReactTestRendererJSON | null = null;
    act(() => {
      tree = renderer.create(<SessionScreen />).toJSON();
    });
    expect(tree).toMatchSnapshot();
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

    let tree: renderer.ReactTestRendererJSON | null = null;
    act(() => {
      tree = renderer.create(<SessionScreen />).toJSON();
    });
    expect(tree).toMatchSnapshot();
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

    let tree: renderer.ReactTestRendererJSON | null = null;
    act(() => {
      tree = renderer.create(<SessionScreen />).toJSON();
    });
    expect(tree).toMatchSnapshot();
  });

  it("renders loading state", () => {
    act(() => {
      useSessionStore.setState({
        turns: [{ speaker: "Alice", text: "We need to talk." }],
        loading: true,
      });
    });

    let tree: renderer.ReactTestRendererJSON | null = null;
    act(() => {
      tree = renderer.create(<SessionScreen />).toJSON();
    });
    expect(tree).toMatchSnapshot();
  });
});
