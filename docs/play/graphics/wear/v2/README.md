# Wear OS Play Store screenshots — v2

## Why v1 was rejected

Google Play rejected the `com.sagearbor.gauge.wear` listing on 2026-09-21
under the Metadata policy, for two reasons:

1. **Unclear visuals** — the six screenshots in `docs/play/graphics/wear/`
   (`wear-01-off.png` .. `wear-06-mode-picker.png`) were plain watch-face
   captures with no explanation. The worst offender was `wear-01-off.png`:
   the sentinel not yet armed, showing a `-76 dB` reading and a green ring
   that means nothing without context. The signal/mode picker screens
   (`wear-04-signal-picker.png`, `wear-06-mode-picker.png`) were just lists
   of option labels with no framing either.
2. **Vague mention of another app** — a paragraph in the full description
   about "Companion mode" ("when you are already running a MindShift
   session on your phone... the phone is listening, and the watch is purely
   the thing that buzzes"). See `docs/play/play-answers-mindshift-wear.yaml`
   → `store_listing.full_description.rejection_2026_09_21` for the complete
   before/after of the description fix; that part of the rejection is
   unrelated to these screenshots.

## What v2 changes

Same five source screenshots (four kept, one dropped, one new caption
choice), each with a **caption band added over the lower ~22%** of the
1080×1080 square — solid `#0b0f17` (matching the rest of the canvas, so it
reads as one image, not a sticker), large white Arial Bold, one line,
≤ 34 characters, explaining what's on screen above it:

| File | Source | Caption | Why |
|---|---|---|---|
| `wear-01-listening.png` | `wear-02-calm.png` | "Listening for your raised voice" | Green ring, quiet reading, Volume signal selected — now explained as the idle/listening state. |
| `wear-02-buzz.png` | `wear-03-elevated.png` | "Over your baseline — a gentle buzz" | Red ring, signal over threshold — explains what the color change means and that it corresponds to a haptic cue. Caption avoids naming "voice" specifically because this capture has the Heart Rate signal selected (120 bpm), not Volume — see the parent README's "What we could not reach" note on why a real mic-triggered episode couldn't be captured in the headless emulator. |
| `wear-03-signals.png` | `wear-04-signal-picker.png` | "Voice, heart rate or movement" | The signal picker, now captioned instead of being a bare list of option labels. |
| `wear-04-buzz-strength.png` | `wear-05-settings.png` | "Choose how strong the buzz is" | The "Feel the buzzes" settings screen. |
| `wear-05-modes.png` | `wear-06-mode-picker.png` | "Standard, Battery Saver or Session" | The mode picker. |

`wear-01-off.png` (sentinel off, `-76 dB`, no caption would have made it
meaningful) is dropped entirely, per the fix's instructions.

Generated with Pillow via `tmp/play-shots/caption.py` (not committed —
`tmp/` is gitignored): each source PNG is opened, a solid-color rectangle is
drawn over `y >= 0.78 * height` (covering the partial chip text that used to
peek out at the bottom edge in v1), and the caption is centered in that
band at the largest font size that fits within a 48px margin.

## Uploaded

Uploaded via the Play Developer API in one edit: `images().deleteall`
removed all six v1 `wearScreenshots`, then `images().upload` added these
five in the order listed above, alongside the description fix. Commit used
`changesNotSentForReview=true` (Play refuses a plain commit right after a
rejection); the coordinator sends the edit for review from the Console UI.
