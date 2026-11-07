/**
 * Pure logic for the in-app "Choose TINY/SMALL model file" buttons: decide
 * whether a picked file is plausibly the right model, what to name it, and
 * whether the phone has room. No I/O here (the copy itself is native, in
 * modelImportNative.ts). The weights are PRIVATE and never enter git.
 */
import { MODEL_FILES, MODEL_MIN_BYTES } from "./config";
import type { ModelKind, ModelStatus } from "./modelFiles";

export type ImportCheck = { ok: true; destName: string } | { ok: false; reason: string };

const mb = (b: number) => `${(b / (1024 * 1024)).toFixed(1)} MB`;
const LABEL: Record<ModelKind, string> = { tiny: "TINY", small: "SMALL" };

/** Validate a picked file's name/size before copying. size may be unknown (null): then it is re-checked after the copy. */
export function validatePickedModel(kind: ModelKind, file: { name: string | null; size: number | null }): ImportCheck {
  const name = (file.name ?? "").trim();
  if (name && !/\.onnx$/i.test(name)) {
    return { ok: false, reason: `That file (${name}) is not an .onnx model. Choose the ${LABEL[kind]} .onnx file.` };
  }
  const other: ModelKind = kind === "tiny" ? "small" : "tiny";
  if (name && new RegExp(other, "i").test(name) && !new RegExp(kind, "i").test(name)) {
    return { ok: false, reason: `${name} looks like the ${LABEL[other]} model, but you tapped the ${LABEL[kind]} button.` };
  }
  if (file.size !== null) {
    const sizeProblem = checkModelSize(kind, file.size);
    if (sizeProblem) return { ok: false, reason: sizeProblem };
  }
  return { ok: true, destName: MODEL_FILES[kind] };
}

/** null when the size is plausible for this model; otherwise why not. Tiny is also capped below the small minimum so the big file can't pass as tiny. */
export function checkModelSize(kind: ModelKind, bytes: number): string | null {
  if (bytes < MODEL_MIN_BYTES[kind]) {
    return `Too small for the ${LABEL[kind]} model (${mb(bytes)}; expected at least ${mb(MODEL_MIN_BYTES[kind])}). Truncated copy or wrong file?`;
  }
  if (kind === "tiny" && bytes >= MODEL_MIN_BYTES.small) {
    return `Too big for the TINY model (${mb(bytes)}). That looks like the SMALL model.`;
  }
  return null;
}

/** Room needed for the copy: size plus 10% / 20 MB headroom (whichever is larger). */
export function checkStorage(bytes: number, freeBytes: number | null): string | null {
  if (freeBytes === null) return null; // unknown: let the copy try and report its own failure
  const need = bytes + Math.max(Math.round(bytes * 0.1), 20 * 1024 * 1024);
  if (freeBytes < need) return `Not enough free storage: need about ${mb(need)}, phone has ${mb(freeBytes)} free.`;
  return null;
}

/** One line per prerequisite for the Emotion dev tester, so a "nothing happens" state always names what is missing. */
export function describePrerequisites(p: {
  enabled: boolean;
  tiny: ModelStatus | null;
  small: ModelStatus | null;
  liveSessionActive: boolean;
}): string[] {
  const line = (label: string, m: ModelStatus | null) =>
    m === null ? `${label} model: not checked yet` : m.status === "ready" ? `${label} model: ready (${mb(m.bytes)})` : `${label} model: MISSING (${m.reason})`;
  return [
    p.enabled ? "Developer mode: ON" : "Developer mode: OFF (turn on the switch above)",
    line("TINY", p.tiny) + (p.tiny?.status === "missing" ? " — required" : ""),
    line("SMALL", p.small) + (p.small?.status === "missing" ? " — optional, TINY runs alone" : ""),
    p.liveSessionActive ? "Live Coach session: audio flowing" : "Live Coach session: NOT running (start one; it supplies the microphone audio)",
  ];
}
