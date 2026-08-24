"""Real ECAPA timing: N separate embed_pcm calls vs ONE embed_pcm_batch call
for the same N chunks. Uses the isolated MINDSHIFT_ECAPA_CACHE (set via env
before running). Writes results to a text file (avoid huge stdout)."""
import os
import sys
import time

sys.path.insert(0, "/Users/sophie.arborbot/PROJECTS/github_repos/mindshift/.claude/worktrees/poker6-v3-refine/server")

import numpy as np  # noqa: E402
import speaker_id  # noqa: E402

SR = 16000
WINDOW_S = 1.5
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8

rng = np.random.default_rng(0)
chunks = [rng.standard_normal(int(WINDOW_S * SR)).astype(np.float32) * 0.01 for _ in range(N)]

print(f"ECAPA cache: {os.environ.get('MINDSHIFT_ECAPA_CACHE')}")
print(f"N={N} chunks of {WINDOW_S}s each")

# Warm up the model load (not counted -- this happens once per process in
# production too).
t0 = time.time()
_ = speaker_id.embed_pcm(chunks[0], SR)
print(f"model load + first call: {time.time() - t0:.2f}s")

# Loop timing (N-1 remaining calls, since chunk 0 already warmed/embedded --
# use fresh chunks for a fair per-call cost after warm-up).
loop_chunks = [rng.standard_normal(int(WINDOW_S * SR)).astype(np.float32) * 0.01 for _ in range(N)]
t0 = time.time()
loop_embs = [speaker_id.embed_pcm(c, SR) for c in loop_chunks]
loop_elapsed = time.time() - t0
print(f"LOOP: {N} embed_pcm calls: {loop_elapsed:.2f}s total, {loop_elapsed/N:.2f}s/call")

# Batched timing (same chunks, fresh call since caching isn't a thing here).
t0 = time.time()
batch_embs = speaker_id.embed_pcm_batch(loop_chunks, SR)
batch_elapsed = time.time() - t0
print(f"BATCH: 1 embed_pcm_batch call for {N} chunks: {batch_elapsed:.2f}s total, {batch_elapsed/N:.2f}s/chunk")

print(f"SPEEDUP: {loop_elapsed / batch_elapsed:.2f}x")

# Sanity: batched and looped embeddings should closely match (same model,
# same audio -- only the wav_lens/padding path differs).
diffs = [float(np.dot(a, b)) for a, b in zip(loop_embs, batch_embs)]
print(f"cosine(loop_emb, batch_emb) per chunk (should be ~1.0): min={min(diffs):.6f} max={max(diffs):.6f}")
