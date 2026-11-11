/**
 * App-start wiring for the Emotion dev tester. The saved switches used to be
 * applied only when Settings > Advanced was mounted, so after an app restart
 * recording stayed silently off until that screen was opened. This loads the
 * persisted settings for the signed-in account and configures the hub, so a
 * restart + Start Listening records with the saved settings.
 */
import type { EmotionDevHub } from "./devHub";
import type { EmotionDevUserSettings } from "./config";

export async function initEmotionDevFromPrefs(
  hub: Pick<EmotionDevHub, "configure">,
  userId: string | null | undefined,
  load: (userId: string | null | undefined) => Promise<EmotionDevUserSettings>,
): Promise<EmotionDevUserSettings> {
  const s = await load(userId);
  await hub.configure(s);
  return s;
}
