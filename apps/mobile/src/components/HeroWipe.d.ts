/**
 * Type surface for the platform-resolved hero banner. Metro (web) and jest
 * (native, the default test platform) pick HeroWipe.web.tsx / .native.tsx
 * respectively; TypeScript has no notion of RN platform extensions, so this
 * declaration gives the bare `./HeroWipe` import (HomeScreen.tsx) its type —
 * the same pattern GoogleSignInButton.d.ts uses for the same reason. Both
 * variants take no props.
 */
import type { ComponentType } from "react";

declare const HeroWipe: ComponentType<Record<string, never>>;
export default HeroWipe;
