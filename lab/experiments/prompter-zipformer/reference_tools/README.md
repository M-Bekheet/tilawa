# Quran-Lab reference eval tools (verbatim copies)

Copied from `Quran-Lab/zipformer_p-arabic-v3` (NPL-1.2). Do not edit these
files; they hardcode the author's Windows icefall checkout and will not run
here as-is.

- `quran_per_eval.py` — PER via PyTorch `model.encode` + JSONL manifest
  (`audio_filepath`, `text`, optional `source`). Gold phonemes from
  `quran_text2phoneme.json` keyed by diacritic-stripped Arabic. Requires
  `zipformer_rnnt_ctc_train.py` from their private recipe.
- `decode_with_confidence.py` — greedy CTC with per-symbol margin_peak.
- `export_quran_streaming_onnx.py` — icefall streaming CTC export,
  `chunk_size=max(context_mix)=24` (1000 ms), `left_context_frames=256`,
  then `quantize_dynamic(..., QInt8, op_types=["MatMul"])`.

ONNX PER on our qlab corpus is `per_onnx_wrapper.py` (does not import the
above). Tables (`tokens.txt`, `quran_text2phoneme.json`,
`ordered_quran_phonemes.json`) live in gitignored
`data/prompter/reference/`.
