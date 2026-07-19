/**
 * Type surface for the platform-resolved Google sign-in button. Metro (web) and
 * jest (native) pick GoogleSignInButton.web.tsx / .native.tsx respectively;
 * TypeScript has no notion of RN platform extensions, so this declaration gives
 * the bare `./GoogleSignInButton` import its shared type. Both variants take no
 * props and wire their own platform's flow from the auth store.
 */
import type { ComponentType } from "react";

declare const GoogleSignInButton: ComponentType<Record<string, never>>;
export default GoogleSignInButton;
