import React from "react";
import renderer, { act } from "react-test-renderer";
import TherapistDashboard from "../src/screens/TherapistDashboard";
import { useDashboardStore } from "../src/store/dashboardStore";

const mockSessions = [
  {
    id: "s1",
    date: "2026-02-20T10:00:00Z",
    role: "Husband / Wife",
    avgPleasantness: 62,
    turns: [
      {
        speaker: "Alice",
        text: "I feel unheard.",
        empathyLevel: 50,
        toneScores: {
          warmth: 40,
          constructiveness: 50,
          calmness: 60,
          respect: 70,
          engagement: 55,
          pleasantness: 55,
        },
      },
      {
        speaker: "Bob",
        text: "I'm trying my best.",
        empathyLevel: 50,
        toneScores: {
          warmth: 65,
          constructiveness: 70,
          calmness: 75,
          respect: 60,
          engagement: 60,
          pleasantness: 68,
        },
      },
    ],
  },
  {
    id: "s2",
    date: "2026-02-21T14:00:00Z",
    role: "Parent / Child",
    avgPleasantness: 45,
    turns: [
      {
        speaker: "Parent",
        text: "You need to focus on school.",
        empathyLevel: 30,
        toneScores: {
          warmth: 30,
          constructiveness: 55,
          calmness: 40,
          respect: 50,
          engagement: 45,
          pleasantness: 45,
        },
      },
    ],
  },
];

beforeEach(() => {
  act(() => {
    useDashboardStore.setState({
      sessions: [],
      selectedSessionId: null,
      roleFilter: null,
      loading: false,
    });
  });
});

describe("TherapistDashboard", () => {
  it("renders empty state", () => {
    let tree: renderer.ReactTestRendererJSON | null = null;
    act(() => {
      tree = renderer
        .create(<TherapistDashboard onSelectSession={jest.fn()} />)
        .toJSON();
    });
    expect(tree).toMatchSnapshot();
  });

  it("renders with sessions", () => {
    act(() => {
      useDashboardStore.setState({ sessions: mockSessions });
    });

    let tree: renderer.ReactTestRendererJSON | null = null;
    act(() => {
      tree = renderer
        .create(<TherapistDashboard onSelectSession={jest.fn()} />)
        .toJSON();
    });
    expect(tree).toMatchSnapshot();
  });

  it("renders loading state", () => {
    act(() => {
      useDashboardStore.setState({ loading: true });
    });

    let tree: renderer.ReactTestRendererJSON | null = null;
    act(() => {
      tree = renderer
        .create(<TherapistDashboard onSelectSession={jest.fn()} />)
        .toJSON();
    });
    expect(tree).toMatchSnapshot();
  });

  it("renders filtered by role", () => {
    act(() => {
      useDashboardStore.setState({
        sessions: mockSessions,
        roleFilter: "Husband / Wife",
      });
    });

    let tree: renderer.ReactTestRendererJSON | null = null;
    act(() => {
      tree = renderer
        .create(<TherapistDashboard onSelectSession={jest.fn()} />)
        .toJSON();
    });
    expect(tree).toMatchSnapshot();
  });
});
