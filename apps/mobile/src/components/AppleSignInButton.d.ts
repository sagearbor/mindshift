/**
 * Type surface for the platform-resolved Apple sign-in button. Metro (web) and
 * jest (native) pick AppleSignInButton.web.tsx / .native.tsx respectively;
 * TypeScript has no notion of RN platform extensions, so this declaration gives
 * the bare `./AppleSignInButton` import its shared type. Mirrors
 * GoogleSignInButton.d.ts.
 */
import type { ComponentType } from "react";

declare const AppleSignInButton: ComponentType<Record<string, never>>;
export default AppleSignInButton;
