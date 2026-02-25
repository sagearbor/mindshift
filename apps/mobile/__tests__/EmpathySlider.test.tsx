import React from "react";
import renderer from "react-test-renderer";
import EmpathySlider, {
  getEmpathyLabel,
} from "../src/components/EmpathySlider";

describe("EmpathySlider", () => {
  it("renders correctly at default value", () => {
    const tree = renderer
      .create(<EmpathySlider value={50} onValueChange={jest.fn()} />)
      .toJSON();
    expect(tree).toMatchSnapshot();
  });

  it("renders correctly at minimum", () => {
    const tree = renderer
      .create(<EmpathySlider value={0} onValueChange={jest.fn()} />)
      .toJSON();
    expect(tree).toMatchSnapshot();
  });

  it("renders correctly at maximum", () => {
    const tree = renderer
      .create(<EmpathySlider value={100} onValueChange={jest.fn()} />)
      .toJSON();
    expect(tree).toMatchSnapshot();
  });

  describe("getEmpathyLabel", () => {
    it("returns Assertive for 0-20", () => {
      expect(getEmpathyLabel(0)).toBe("Assertive");
      expect(getEmpathyLabel(20)).toBe("Assertive");
    });

    it("returns Balanced for 21-50", () => {
      expect(getEmpathyLabel(21)).toBe("Balanced");
      expect(getEmpathyLabel(50)).toBe("Balanced");
    });

    it("returns Empathetic for 51-80", () => {
      expect(getEmpathyLabel(51)).toBe("Empathetic");
      expect(getEmpathyLabel(80)).toBe("Empathetic");
    });

    it("returns Full Empathy for 81-100", () => {
      expect(getEmpathyLabel(81)).toBe("Full Empathy");
      expect(getEmpathyLabel(100)).toBe("Full Empathy");
    });
  });
});
