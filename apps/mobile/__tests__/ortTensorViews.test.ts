import { float32Tensor } from "../src/live/ort";

// onnxruntime-react-native ignores byteOffset on typed-array views (measured
// 2026-08-30: every window embedded the recording's first 1.5 s). The tensor
// helper must therefore never hand ORT a view into a larger buffer.
describe("float32Tensor never passes a subarray view to ORT", () => {
  it("copies a view with a byteOffset into an owned zero-offset array with the view's contents", () => {
    const whole = Float32Array.from({ length: 10 }, (_, i) => i);
    const view = whole.subarray(4, 8);
    const t = float32Tensor(view, [1, 4]);
    expect(t.data.byteOffset).toBe(0);
    expect(t.data.byteLength).toBe(t.data.buffer.byteLength);
    expect(Array.from(t.data)).toEqual([4, 5, 6, 7]);
    expect(t.data).not.toBe(view);
  });

  it("copies a zero-offset view that is shorter than its buffer", () => {
    const whole = new Float32Array(10);
    const head = whole.subarray(0, 3);
    const t = float32Tensor(head, [1, 3]);
    expect(t.data.byteLength).toBe(t.data.buffer.byteLength);
  });

  it("passes an owned whole-buffer array through untouched", () => {
    const owned = Float32Array.from([1, 2, 3]);
    expect(float32Tensor(owned, [1, 3]).data).toBe(owned);
  });
});
