/**
 * End-to-end parity with the Python reference on the golden clip, using the
 * REAL private models through onnxruntime-node — but only on a machine that
 * has them. The weights are PRIVATE and never in this repo: point
 * EMOTION_MODEL_DIR at ".../mindshift-emotion-private/results" (the folder
 * holding tiny-finetuned-v1/export and small-finetuned-v1/export). Without
 * it these tests skip, loudly, rather than pass on fabricated numbers.
 */
import * as fs from "fs";
import * as path from "path";
import { OnnxEmotionEngine } from "../src/emotion/engine";
import { computeLogMel, int16ToFloat32 } from "../src/emotion/logMel";
import { nodeOrtSessionFactory } from "../src/live/testing/ortNode";

const DIR = process.env.EMOTION_MODEL_DIR ?? "/Users/sagearbor/projects/githubs/mindshift-emotion-private/results";
const FIX = path.join(__dirname, "..", "src", "emotion", "__fixtures__");
const meta = JSON.parse(fs.readFileSync(path.join(FIX, "clip04-meta.json"), "utf8"));
const have = (k: string) => fs.existsSync(path.join(DIR, `${k}-finetuned-v1`, "export", "emotion-int8.onnx"));

const pcm = (() => {
  const b = fs.readFileSync(path.join(FIX, "clip04-16k-int16.bin"));
  return int16ToFloat32(new Int16Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)));
})();

describe("emotion models: TS frontend + onnxruntime == Python reference", () => {
  for (const k of ["tiny", "small"] as const) {
    const run = have(k) ? it : it.skip;
    run(`${k}: probabilities on clip-04 match the Python reference`, async () => {
      const session = await nodeOrtSessionFactory()(path.join(DIR, `${k}-finetuned-v1`, "export", "emotion-int8.onnx"));
      const engine = new OnnxEmotionEngine(k, session);
      const p = await engine.classify(computeLogMel(pcm));
      await engine.release();
      const ref: number[] = meta.reference_probs[k];
      p.forEach((v, i) => expect(Math.abs(v - ref[i])).toBeLessThan(0.02));
    });
  }
});
