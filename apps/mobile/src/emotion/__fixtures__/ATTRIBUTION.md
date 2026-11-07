# Fixture attribution

`clip04-16k-int16.bin` is clip-04 of the Morgan Emotional Speech Set (MESS),
resampled to 16 kHz mono and quantised to int16 by `scripts/emotion_golden.py`.

Creator: Shae Morgan, University of Louisville.
Source: https://doi.org/10.5281/zenodo.7378320
License: CC BY 4.0 https://creativecommons.org/licenses/by/4.0/
Morgan, S. D. (2019). Categorical and Dimensional Ratings of Emotional Speech:
Behavioral Findings From the Morgan Emotional Speech Set. Journal of Speech,
Language, and Hearing Research, 62(11), 4015-4029. Modified (resampled,
quantised); no endorsement implied.

`clip04-logmel.f32` is the HF WhisperFeatureExtractor log-mel of that signal
([80][600] little-endian float32). `clip04-meta.json` also records the two
private models' probabilities on it (outputs only, no weights) for the
optional emotionModelParity test.
