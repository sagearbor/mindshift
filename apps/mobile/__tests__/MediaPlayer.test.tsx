import React from "react";
import renderer, { act } from "react-test-renderer";
import MediaPlayer from "../src/components/MediaPlayer";

// The jest-setup expo-video mock exposes the controllable player here, so we
// can inspect what the component set on it during useVideoPlayer's setup.
const player = (globalThis as Record<string, unknown>).__expoVideoMock as {
  muted: boolean;
  volume: number;
  currentTime: number;
  duration: number;
  playing: boolean;
};

describe("MediaPlayer", () => {
  it("plays audibly: forces muted=false and volume=1 on the player, whatever its prior state", async () => {
    // Simulate a hostile prior state (e.g. a player left muted / at zero volume).
    player.muted = true;
    player.volume = 0;

    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(
        <MediaPlayer uri="https://example.test/a.m4a" mediaType="audio" />,
      );
    });

    // The setup callback ran during useVideoPlayer and unmuted / restored volume.
    expect(player.muted).toBe(false);
    expect(player.volume).toBe(1);

    act(() => comp.unmount());
  });
});
