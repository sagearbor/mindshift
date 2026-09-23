import { Alert, Platform, type AlertButton } from "react-native";

/**
 * `Alert.alert` that actually shows something on web.
 *
 * react-native-web ships `Alert` as a no-op, and this app renders the same
 * source in the browser (arborfam-hub.web.app). Every confirmation and error
 * that went through `Alert.alert` — "Voice forgotten", "Export failed", the
 * watch-unpair confirm, "Forget this person?" — silently vanished there (UX
 * walk 2026-09-23). On web this falls back to the browser's own dialogs:
 * one-button alerts → `window.alert`, two-or-more → `window.confirm`, where
 * OK runs the first non-cancel button and Cancel runs the cancel button.
 * Native is untouched: it is `Alert.alert`, so existing spies keep working.
 */
export function showAlert(
  title: string,
  message?: string,
  buttons?: AlertButton[],
): void {
  if (Platform.OS !== "web" || typeof window === "undefined") {
    Alert.alert(title, message, buttons);
    return;
  }
  const text = message ? `${title}\n\n${message}` : title;
  if (!buttons || buttons.length <= 1) {
    window.alert(text);
    buttons?.[0]?.onPress?.();
    return;
  }
  const cancel = buttons.find((b) => b.style === "cancel");
  const confirm =
    buttons.find((b) => b.style !== "cancel") ?? buttons[buttons.length - 1];
  if (window.confirm(text)) confirm.onPress?.();
  else cancel?.onPress?.();
}
