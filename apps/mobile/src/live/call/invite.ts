/**
 * Share an invite: the native share sheet where there is one, the browser's
 * Web Share API where it exists (iOS Safari has it), else the clipboard —
 * and the honest outcome for the screen either way.
 */
import { Platform, Share } from "react-native";
import { inviteMessage } from "../../nav/callLink";
import type { CallRole } from "./types";

export type InviteOutcome = "shared" | "copied" | "shown";

export interface InviteDeps {
  share?: (content: { message: string; url?: string; title?: string }) => Promise<unknown>;
  webShare?: ((data: { text: string; url: string; title?: string }) => Promise<void>) | null;
  copy?: ((text: string) => Promise<void>) | null;
  platform?: string;
}

export async function shareInvite(
  code: string,
  joinUrl: string | null | undefined,
  role: CallRole = "participant",
  deps: InviteDeps = {},
): Promise<{ outcome: InviteOutcome; url: string }> {
  const { message, url } = inviteMessage(code, joinUrl, role);
  const platform = deps.platform ?? Platform.OS;
  if (platform === "web") {
    const nav = typeof navigator !== "undefined" ? (navigator as Navigator & { share?: (d: ShareData) => Promise<void> }) : null;
    const webShare =
      deps.webShare === undefined
        ? nav && typeof nav.share === "function"
          ? (d: { text: string; url: string; title?: string }) => nav.share(d)
          : null
        : deps.webShare;
    if (webShare) {
      try {
        await webShare({ title: "MindShift call", text: message, url });
        return { outcome: "shared", url };
      } catch {
        // Cancelled or refused: fall through to the clipboard.
      }
    }
    const copy =
      deps.copy === undefined
        ? nav && nav.clipboard && typeof nav.clipboard.writeText === "function"
          ? (t: string) => nav.clipboard.writeText(t)
          : null
        : deps.copy;
    if (copy) {
      try {
        await copy(url);
        return { outcome: "copied", url };
      } catch {
        // No clipboard access: show it.
      }
    }
    return { outcome: "shown", url };
  }
  const share = deps.share ?? ((c) => Share.share(c));
  try {
    await share({ message, url, title: "MindShift call" });
    return { outcome: "shared", url };
  } catch {
    return { outcome: "shown", url };
  }
}
