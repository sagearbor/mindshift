import React from "react";
import { Text } from "react-native";
import renderer, { act } from "react-test-renderer";
import AccordionSection, { useAccordionState } from "../src/components/AccordionSection";
import * as SecureStore from "expo-secure-store";
import { accordionKey, loadOpenSections, saveOpenSections } from "../src/utils/accordionPrefs";

const get = SecureStore.getItemAsync as jest.Mock;
const set = SecureStore.setItemAsync as jest.Mock;

function Harness({ uid }: { uid: string | null }) {
  const { open, toggle } = useAccordionState(uid);
  return (
    <AccordionSection
      sectionId="section-x"
      title="X"
      subtitle="sub"
      expanded={!!open.x}
      onToggle={() => toggle("x")}
    >
      <Text testID="inner">inside</Text>
    </AccordionSection>
  );
}

const flush = () => act(async () => { await Promise.resolve(); });
const byId = (c: renderer.ReactTestRenderer, id: string) =>
  c.root.findAll((n) => n.props?.testID === id && typeof n.type === "string")[0];

describe("AccordionSection", () => {
  beforeEach(() => {
    get.mockReset().mockResolvedValue(null);
    set.mockReset().mockResolvedValue(undefined);
  });

  it("starts collapsed but keeps children mounted", async () => {
    let c!: renderer.ReactTestRenderer;
    await act(async () => { c = renderer.create(<Harness uid="u1" />); });
    expect(byId(c, "inner")).toBeTruthy();
    expect(byId(c, "section-x-body").props.style).toEqual({ display: "none" });
    expect(byId(c, "section-x-toggle").props.accessibilityState).toEqual({ expanded: false });
  });

  it("toggles open and persists per user", async () => {
    let c!: renderer.ReactTestRenderer;
    await act(async () => { c = renderer.create(<Harness uid="u1" />); });
    await act(async () => { c.root.findAll((n) => n.props?.testID === "section-x-toggle" && typeof n.props.onPress === "function")[0].props.onPress(); });
    expect(byId(c, "section-x-toggle").props.accessibilityState).toEqual({ expanded: true });
    expect(byId(c, "section-x-body").props.style).not.toEqual({ display: "none" });
    expect(set).toHaveBeenCalledWith(accordionKey("u1"), JSON.stringify({ x: true }));
  });

  it("restores remembered open state", async () => {
    get.mockResolvedValue(JSON.stringify({ x: true }));
    let c!: renderer.ReactTestRenderer;
    await act(async () => { c = renderer.create(<Harness uid="u1" />); });
    await flush();
    expect(byId(c, "section-x-toggle").props.accessibilityState).toEqual({ expanded: true });
  });

  it("fails open to collapsed when storage throws or holds junk", async () => {
    get.mockRejectedValueOnce(new Error("boom"));
    expect(await loadOpenSections("u")).toEqual({});
    get.mockResolvedValueOnce("not json");
    expect(await loadOpenSections("u")).toEqual({});
    get.mockResolvedValueOnce(JSON.stringify({ a: true, b: "yes" }));
    expect(await loadOpenSections("u")).toEqual({ a: true });
    set.mockRejectedValueOnce(new Error("boom"));
    await expect(saveOpenSections("u", { a: true })).resolves.toBeUndefined();
  });
});
