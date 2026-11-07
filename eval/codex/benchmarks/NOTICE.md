# CREMA-D derived benchmark attribution

Source: [CheyneyComputerScience/CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D),
Crowd-sourced Emotional Multimodal Actors Dataset, source commit
`1658cd342dff90010aa843eaeebd53610a08b1dc`.

Cite Cao, Cooper, Keutmann, Gur, Nenkova and Verma (2014),
“CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset”, IEEE Transactions on Affective Computing.

The source database is available under [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/).
Individual contents are under [DbCL 1.0](https://opendatacommons.org/licenses/dbcl/1-0/).
This derived benchmark index is made available under those same applicable terms.
Retain attribution and comply with ODbL database share-alike obligations when using or redistributing it.
No audio is included in this repository change; obtain it from the official source.

Changes made here: deterministic actor-disjoint partitions, reconstructed audio-only
crowd vote distributions with the original response-time filter, file hashes, and
exclusion of one all-zero WAV. Actor intent is retained separately. No new human
annotations were created; no machine label is presented as human ground truth.

Whisper model weights and code are separately [MIT-licensed](https://github.com/openai/whisper#license).
Qwen2.5-Omni is separately [Apache-2.0-licensed](https://huggingface.co/Qwen/Qwen2.5-Omni-7B/blob/main/LICENSE).
These pretrained weights are downloaded locally and are not part of the benchmark index.
