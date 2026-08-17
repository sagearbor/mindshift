import React from "react";
import renderer, { act } from "react-test-renderer";
import {
  ICONS,
  CHROME_ICONS,
  getIcon,
  type IconProps,
} from "../src/components/icons";
import { DESTINATIONS, type IconId } from "../src/nav/destinations";

// One render-all test per the plan (no per-icon snapshot explosion): every
// destination icon + every chrome glyph, rendered together.
const ALL_ICONS: [string, React.ComponentType<IconProps>][] = [
  ...Object.entries(ICONS),
  ...Object.entries(CHROME_ICONS).map(
    ([name, Icon]) => [`chrome:${name}`, Icon] as [string, React.ComponentType<IconProps>],
  ),
];

/** Flatten a react-test-renderer JSON tree into a flat list of nodes
 *  (including the root), so a prop can be searched for anywhere in the
 *  rendered SVG, regardless of which shape carries it. */
function flatten(node: renderer.ReactTestRendererJSON | null): renderer.ReactTestRendererJSON[] {
  if (!node) return [];
  const out = [node];
  const children = node.children ?? [];
  for (const child of children) {
    if (typeof child !== "string") out.push(...flatten(child));
  }
  return out;
}

describe("icon set", () => {
  it.each(ALL_ICONS)("%s renders without throwing", (_name, Icon) => {
    let tree: renderer.ReactTestRenderer | undefined;
    expect(() => {
      act(() => {
        tree = renderer.create(<Icon testID="icon" />);
      });
    }).not.toThrow();
    expect(tree!.toJSON()).toBeTruthy();
  });

  it.each(ALL_ICONS)("%s respects the size prop", (_name, Icon) => {
    let tree: renderer.ReactTestRenderer;
    act(() => {
      tree = renderer.create(<Icon size={40} testID="icon" />);
    });
    const root = tree!.toJSON() as renderer.ReactTestRendererJSON;
    expect(root.props.width).toBe(40);
    expect(root.props.height).toBe(40);
  });

  it.each(ALL_ICONS)("%s respects the color prop", (_name, Icon) => {
    let tree: renderer.ReactTestRenderer;
    act(() => {
      tree = renderer.create(<Icon color="#123456" testID="icon" />);
    });
    const nodes = flatten(tree!.toJSON() as renderer.ReactTestRendererJSON);
    // The chosen color must show up as a stroke somewhere in the tree — every
    // icon here is stroke-only, so a color prop that's silently ignored would
    // leave the default ink color instead and fail this.
    expect(nodes.some((n) => n.props.stroke === "#123456")).toBe(true);
    expect(nodes.some((n) => n.props.stroke === "#1F2937")).toBe(false);
  });

  it.each(ALL_ICONS)("%s defaults to size 24 and the house ink color", (_name, Icon) => {
    let tree: renderer.ReactTestRenderer;
    act(() => {
      tree = renderer.create(<Icon testID="icon" />);
    });
    const root = tree!.toJSON() as renderer.ReactTestRendererJSON;
    expect(root.props.width).toBe(24);
    expect(root.props.height).toBe(24);
    const nodes = flatten(root);
    expect(nodes.some((n) => n.props.stroke === "#1F2937")).toBe(true);
  });

  it("covers every destination's IconId — a missing icon fails here, not just at compile time", () => {
    const ids: IconId[] = DESTINATIONS.map((d) => d.iconId);
    for (const id of ids) {
      expect(ICONS[id]).toBeDefined();
      expect(getIcon(id)).toBe(ICONS[id]);
    }
  });

  it("has all four chrome glyphs", () => {
    expect(Object.keys(CHROME_ICONS).sort()).toEqual(
      ["back", "camera", "close", "menu"].sort(),
    );
  });
});
