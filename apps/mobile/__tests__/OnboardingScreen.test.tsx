import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import OnboardingScreen from "../src/screens/OnboardingScreen";
import { ONBOARDING_CARDS } from "../src/screens/onboardingCards";

function render(onFinish: () => void) {
  let root!: renderer.ReactTestRenderer;
  act(() => {
    root = renderer.create(<OnboardingScreen onFinish={onFinish} />);
  });
  return root;
}

function find(root: renderer.ReactTestRenderer, testID: string): ReactTestInstance {
  return root.root.findByProps({ testID });
}

function tryFind(
  root: renderer.ReactTestRenderer,
  testID: string,
): ReactTestInstance | null {
  const matches = root.root.findAllByProps({ testID });
  return matches.length > 0 ? matches[0] : null;
}

describe("OnboardingScreen", () => {
  it("renders the first card on mount", () => {
    const root = render(jest.fn());
    expect(
      find(root, `onboarding-title-${ONBOARDING_CARDS[0].id}`).props.children,
    ).toBe(ONBOARDING_CARDS[0].title);
    // The active dot is the first one.
    expect(
      find(root, `onboarding-dot-${ONBOARDING_CARDS[0].id}`).props.style,
    ).toEqual(
      expect.arrayContaining([expect.objectContaining({ width: 20 })]),
    );
  });

  it("shows Next (not Get started) on every card but the last", () => {
    const root = render(jest.fn());
    expect(tryFind(root, "onboarding-next")).not.toBeNull();
    expect(tryFind(root, "onboarding-get-started")).toBeNull();
  });

  it("Skip calls onFinish immediately from the first card", () => {
    const onFinish = jest.fn();
    const root = render(onFinish);
    act(() => {
      find(root, "onboarding-skip").props.onPress();
    });
    expect(onFinish).toHaveBeenCalledTimes(1);
  });

  it("Next advances through all four cards to Get started, which finishes", () => {
    const onFinish = jest.fn();
    const root = render(onFinish);

    // Card 1 -> 2 -> 3 -> 4 via Next.
    for (let i = 0; i < ONBOARDING_CARDS.length - 1; i++) {
      act(() => {
        find(root, "onboarding-next").props.onPress();
      });
    }

    // Now on the last card: title matches, and the button is "Get started".
    const lastCard = ONBOARDING_CARDS[ONBOARDING_CARDS.length - 1];
    expect(find(root, `onboarding-title-${lastCard.id}`).props.children).toBe(
      lastCard.title,
    );
    expect(tryFind(root, "onboarding-next")).toBeNull();
    expect(onFinish).not.toHaveBeenCalled();

    act(() => {
      find(root, "onboarding-get-started").props.onPress();
    });
    expect(onFinish).toHaveBeenCalledTimes(1);
  });

  it("Back returns to the previous card and is inert on the first card", () => {
    const root = render(jest.fn());

    act(() => {
      find(root, "onboarding-next").props.onPress();
    });
    expect(
      find(root, `onboarding-title-${ONBOARDING_CARDS[1].id}`).props.children,
    ).toBe(ONBOARDING_CARDS[1].title);

    act(() => {
      find(root, "onboarding-back").props.onPress();
    });
    expect(
      find(root, `onboarding-title-${ONBOARDING_CARDS[0].id}`).props.children,
    ).toBe(ONBOARDING_CARDS[0].title);

    // Tapping Back again on the first card is a no-op (still card 0).
    act(() => {
      find(root, "onboarding-back").props.onPress();
    });
    expect(
      find(root, `onboarding-title-${ONBOARDING_CARDS[0].id}`).props.children,
    ).toBe(ONBOARDING_CARDS[0].title);
  });

  it("Skip is present on every card, including the last", () => {
    const onFinish = jest.fn();
    const root = render(onFinish);
    for (let i = 0; i < ONBOARDING_CARDS.length - 1; i++) {
      act(() => {
        find(root, "onboarding-next").props.onPress();
      });
    }
    act(() => {
      find(root, "onboarding-skip").props.onPress();
    });
    expect(onFinish).toHaveBeenCalledTimes(1);
  });

  it("renders all four card bodies somewhere in the scroll content", () => {
    const root = render(jest.fn());
    for (const card of ONBOARDING_CARDS) {
      expect(find(root, `onboarding-body-${card.id}`).props.children).toBe(
        card.body,
      );
    }
  });
});
