import React from "react";
import renderer, { act } from "react-test-renderer";
// Imported by its platform-qualified path ON PURPOSE. This suite runs with
// native module resolution (see LoginScreen.web.test.tsx's header), so a bare
// "../src/components/AppleSignInButton" would resolve to the .native variant
// and the assertion below would silently test the wrong file.
import AppleSignInButtonWeb from "../src/components/AppleSignInButton.web";

/**
 * Sign in with Apple is absent on web by design, not merely unfinished.
 * Guideline 4.8 binds the App Store build; the web dashboard is not
 * distributed through Apple. Offering it here would mean configuring Apple's
 * redirect flow (a Services ID plus a private key in the Firebase provider)
 * for no requirement and one more credential to rotate.
 */
describe("AppleSignInButton (web)", () => {
  it("renders nothing, so LoginScreen can import it unconditionally", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<AppleSignInButtonWeb />);
    });
    expect(comp.toJSON()).toBeNull();
    act(() => comp.unmount());
  });
});
