import React from "react";
import renderer, { act } from "react-test-renderer";
import SessionDetail from "../src/screens/SessionDetail";
import { useDashboardStore } from "../src/store/dashboardStore";

const mockSession = {
  id: "s1",
  date: "2026-02-20T10:00:00Z",
  role: "Husband / Wife",
  avgPleasantness: 58,
  turns: [
    {
      speaker: "Alice",
      text: "I feel like you never listen.",
      empathyLevel: 50,
      toneScores: {
        warmth: 30,
        constructiveness: 40,
        calmness: 35,
        respect: 45,
        engagement: 50,
        pleasantness: 38,
      },
    },
    {
      speaker: "Bob",
      text: "I hear you, and I want to do better.",
      empathyLevel: 75,
      toneScores: {
        warmth: 80,
        constructiveness: 75,
        calmness: 85,
        respect: 70,
        engagement: 65,
        pleasantness: 78,
      },
    },
    {
      speaker: "Alice",
      text: "That means a lot to me.",
      empathyLevel: 60,
      toneScores: {
        warmth: 75,
        constructiveness: 60,
        calmness: 70,
        respect: 65,
        engagement: 55,
        pleasantness: 67,
      },
    },
  ],
};

beforeEach(() => {
  act(() => {
    useDashboardStore.setState({
      sessions: [mockSession],
      selectedSessionId: null,
      roleFilter: null,
      loading: false,
    });
  });
});

describe("SessionDetail", () => {
  it("renders session detail with transcript", () => {
    let component: renderer.ReactTestRenderer;
    act(() => {
      component = renderer
        .create(<SessionDetail sessionId="s1" onBack={jest.fn()} />)
        ;
    });
    expect(component!.toJSON()).toMatchSnapshot();
  });

  it("renders empty state for missing session", () => {
    let component: renderer.ReactTestRenderer;
    act(() => {
      component = renderer
        .create(<SessionDetail sessionId="nonexistent" onBack={jest.fn()} />)
        ;
    });
    expect(component!.toJSON()).toMatchSnapshot();
  });
});
