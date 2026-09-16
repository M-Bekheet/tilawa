# Known label issues — test_corpus_v3

Do **not** silently rewrite `manifest.json`. These gold problems were confirmed by an independent Gemini 3.1 Pro transcription oracle on 2026-09-16. Full per-clip transcripts, similarities, and verdicts: `artifacts/gemini_oracle/misses_oracle.json` (writeup: `artifacts/gemini_oracle/README.md`).

## Confirmed wrong ayah

| id | labelled | actual | evidence |
|---|---|---|---|
| `tlog_m043_010_043` | 10:43 | **10:42** | Gemini transcript matches 10:42 (`ومنهم من يستمعون إليك…`) exactly; Zipformer predicted 10:42. |
| `tlog_m044_010_043` | 10:43 | **10:42** | Same recitation / same transcript as `tlog_m043_010_043`. |

## Incomplete span

| id | labelled | actual | evidence |
|---|---|---|---|
| `tlog_m008_107_001` | 107:1 | **106:4 then 107:1** | Gemini heard Quraysh 106:4 followed by Māʿūn 107:1; the label omits the leading ayah. |
