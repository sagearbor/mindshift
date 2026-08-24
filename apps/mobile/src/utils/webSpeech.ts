/**
 * Web-only helpers around `window.speechSynthesis` (what expo-speech's web
 * build calls into).
 *
 * iOS Safari refuses to voice an utterance unless speech synthesis was
 * first used inside a user gesture. Coaching suggestions arrive seconds
 * after the Start tap (from the server's `suggestion` event), so the FIRST
 * one would be silently dropped. `unlockWebSpeechSynthesis()` speaks a
 * silent, empty-ish utterance synchronously in the Start handler, which
 * satisfies the gate for the rest of the page's life.
 *
 * Import-safe in Node/Jest: every global is read lazily and guarded.
 */

export function webSpeechSynthesisAvailable(): boolean {
  return (
    typeof window !== "undefined" &&
    typeof window.speechSynthesis !== "undefined" &&
    typeof SpeechSynthesisUtterance !== "undefined"
  );
}

/** Must be called synchronously from a user gesture. Never throws. */
export function unlockWebSpeechSynthesis(): boolean {
  try {
    if (!webSpeechSynthesisAvailable()) return false;
    const synth = window.speechSynthesis;
    const utterance = new SpeechSynthesisUtterance(" ");
    utterance.volume = 0;
    utterance.rate = 10; // over in an instant
    synth.speak(utterance);
    return true;
  } catch {
    return false;
  }
}
