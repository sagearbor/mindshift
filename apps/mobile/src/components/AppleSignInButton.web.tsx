import React from "react";

/**
 * No Sign in with Apple on web, on purpose.
 *
 * Guideline 4.8 is an App Store rule, so it binds the iOS build; the web
 * dashboard is not distributed through Apple and is not covered. Offering it
 * here would mean configuring Apple's redirect-based OAuth flow — a Services
 * ID plus a private key in the Firebase provider — for no requirement and one
 * more credential to rotate. Web users have Google, email/password and guest.
 *
 * This renders nothing rather than being absent, so LoginScreen can import the
 * button unconditionally and stay platform-free.
 */
export default function AppleSignInButton() {
  return null;
}
