/**
 * The swappable inference seam of the Emotion dev tester.
 *
 * Phase 1 runs both models on the PHONE through ONNX Runtime
 * ({@link OnnxEmotionEngine}). Phase 2 (tiny on the watch, no phone) swaps
 * the implementation behind this same interface — a Kotlin engine on the
 * watch has the identical contract: a prepared 80x600 log-mel + valid_frames
 * in, six probabilities (EMOTION_LABELS order) out. Everything above this
 * file (gates, rate caps, flags, store, transport) is engine-agnostic.
 */
import type { OnnxSession } from "../live/ort";
import { float32Tensor } from "../live/ort";
import { EMOTION_LABELS } from "./config";
import { N_FRAMES, N_MELS, type LogMel } from "./logMel";
import { isProbVector } from "./logic";

export type EmotionModelId = "tiny" | "small";

export interface EmotionEngine {
  readonly id: EmotionModelId;
  /** Human-readable runtime ("onnxruntime-react-native cpu", "watch-kotlin", …). */
  readonly runtime: string;
  /** Six probabilities in EMOTION_LABELS order. Throws on a bad model output. */
  classify(mel: LogMel): Promise<number[]>;
  release(): Promise<void>;
}

export class OnnxEmotionEngine implements EmotionEngine {
  readonly runtime = "onnxruntime cpu";
  constructor(readonly id: EmotionModelId, private readonly session: OnnxSession) {}

  async classify(mel: LogMel): Promise<number[]> {
    if (mel.data.length !== N_MELS * N_FRAMES) throw new Error("log-mel has the wrong shape");
    const out = await this.session.run({
      log_mel: float32Tensor(mel.data, [1, N_MELS, N_FRAMES]),
      valid_frames: { type: "int64", data: BigInt64Array.from([BigInt(mel.validFrames)]), dims: [1] },
    });
    const first = out[this.session.outputNames[0]];
    if (!first) throw new Error("model returned no output");
    const p = Array.from(first.data as Float32Array, Number);
    if (!isProbVector(p)) {
      throw new Error(`model output is not a ${EMOTION_LABELS.length}-class probability vector`);
    }
    return p;
  }

  release(): Promise<void> {
    return this.session.release();
  }
}
