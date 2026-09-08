"""Segmentation-only diagnostics from the cached pyannote intermediates:
  vad_miss      fraction of GT speech frames where pyannote hears NOBODY
  unit_purity   active-duration-weighted fraction of each (chunk, local
                speaker) unit that belongs to the unit's majority GT speaker
                (1.0 = pyannote's local segmentation never mixes speakers)
  per-speaker   share of each GT speaker's frames that land in a unit whose
                majority is SOMEONE ELSE (= lost before clustering even runs)
Runs under tmp/venv-pyannote.  Writes results_segdiag.json.
"""
from __future__ import annotations

import json

import numpy as np

from common import HERE, score
from hybrid_lib import Cached


def main():
    out = {}
    for name in score.all_fixtures():
        c = Cached(name)
        C, F, L = c.bin.data.shape
        frame_len = c.bin.sliding_window.duration / F

        def allowed(l):
            return tuple(l) if isinstance(l, (tuple, list)) else (l,)
        labels = sorted({l for g in c.gt for l in allowed(g[2])})
        # VAD miss: GT frames (10 ms) with count == 0
        cnt, csw = c.count.data[:, 0], c.count.sliding_window
        gt_frames = 0
        miss = 0
        lost = {l: [0, 0] for l in labels}  # [frames in wrong-majority unit, frames]
        purity_num = purity_den = 0.0
        for u in c.units:
            ch, s = u["chunk"], u["spk"]
            act = np.nan_to_num(c.bin.data[ch, :, s]) > 0.5
            votes = {l: 0 for l in labels}
            n_act = 0
            for f in np.where(act)[0]:
                t = u["chunk_start"] + (f + 0.5) * frame_len
                for gs, ge, gl in c.gt:
                    if gs <= t < ge:
                        for l in allowed(gl):
                            votes[l] += 1
                        n_act += 1
                        break
            if n_act == 0:
                continue
            maj = max(votes, key=votes.get)
            purity_num += votes[maj]
            purity_den += n_act
            for l in labels:
                if l != maj:
                    lost[l][0] += votes[l]
                lost[l][1] += votes[l]
        for gs, ge, gl in c.gt:
            for t in np.arange(gs, ge, 0.01):
                gt_frames += 1
                i = int((t - csw.start) / csw.step)
                if i < 0 or i >= len(cnt) or cnt[i] == 0:
                    miss += 1
        out[name] = {"vad_miss": round(miss / gt_frames, 3), "unit_purity": round(purity_num / purity_den, 3),
                     "lost_to_other_speakers_unit": {l: round(a / b, 3) if b else None for l, (a, b) in lost.items()},
                     "units": len(c.units), "chunks": C}
        print(name, out[name], flush=True)
    (HERE / "results_segdiag.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
