# Gemini transcription oracle — missed clips

Independent ASR on the 21 clips our verse-recognition models miss. Model: `gemini-3.1-pro-preview` (latest numbered Pro with `generateContent`; temperature 0). Transcripts scored with Levenshtein ratio of `normalize_arabic(transcript)` vs QuranDB uthmani / `text_clean` / `text_clean_no_bsm` — independent of Gemini’s own surah/ayah ID.

## Verdict counts

- **label_ok_ambiguous**: 17
- **label_wrong**: 2
- **model_wrong**: 0
- **undetermined**: 2

## Per-clip

| id | corpus | expected | predicted | gemini | sim_exp | sim_pred | pair | verdict |
|---|---|---|---|---|---:|---:|---:|---|
| `ea_alafasy_055053` | v3 | 55:53 | 55:13 | 55:13 | 0.933 | 0.933 | 1.000 | label_ok_ambiguous |
| `ea_alafasy_081019` | v3 | 81:19 | 69:40 | 69:40 | 1.000 | 1.000 | 1.000 | label_ok_ambiguous |
| `ea_husary_037082` | v3 | 37:82 | 26:66 | 26:66 | 0.971 | 0.971 | 1.000 | label_ok_ambiguous |
| `ea_alafasy_030001` | v3 | 30:1 | 2:1 | 2:1 | 1.000 | 1.000 | 1.000 | label_ok_ambiguous |
| `ea_husary_026122` | v3 | 26:122 | 26:9 | 26:9 | 1.000 | 1.000 | 1.000 | label_ok_ambiguous |
| `tlog_m000_100_001` | v3 | 100:1 | 100:1 | 100:1–2 | 1.000 | 1.000 | 1.000 | undetermined |
| `tlog_m008_107_001` | v3 | 107:1 | 106:4 | 106:4 | 0.909 | 0.939 | 0.304 | undetermined |
| `tlog_m043_010_043` | v3 | 10:43 | 10:42 | 10:42 | 0.804 | 1.000 | 0.804 | label_wrong |
| `tlog_m044_010_043` | v3 | 10:43 | 10:42 | 10:42 | 0.804 | 1.000 | 0.804 | label_wrong |
| `qul_alnufais__37_43` | qlab | 37:43 | 37:43 | 56:12 | 0.963 | 0.963 | 1.000 | label_ok_ambiguous |
| `qul_alnufais__56_12` | qlab | 56:12 | 37:43 | 56:12 | 0.963 | 0.963 | 1.000 | label_ok_ambiguous |
| `qul_alnufais__21_38` | qlab | 21:38 | 10:48 | 10:48 | 1.000 | 1.000 | 1.000 | label_ok_ambiguous |
| `qul_alnufais__55_40` | qlab | 55:40 | 55:13 | 55:13 | 0.933 | 0.933 | 1.000 | label_ok_ambiguous |
| `qul_alnufais__83_13` | qlab | 83:13 | 68:15 | 68:15 | 0.988 | 0.988 | 1.000 | label_ok_ambiguous |
| `qul_alnufais__8_51` | qlab | 8:51 | 3:182 | 3:182 | 1.000 | 1.000 | 1.000 | label_ok_ambiguous |
| `qul_alnufais__55_30` | qlab | 55:30 | 55:13 | 55:13 | 0.933 | 0.933 | 1.000 | label_ok_ambiguous |
| `tlog_holdout__37_176_undefined_Bc1Te4g` | qlab | 37:176 | 26:204 | 26:204 | 1.000 | 1.000 | 1.000 | label_ok_ambiguous |
| `tlog_holdout__70_29_3740714225` | qlab | 70:29 | 23:5 | 23:5 | 1.000 | 1.000 | 1.000 | label_ok_ambiguous |
| `tlog_holdout__77_45_6585124791` | qlab | 77:45 | 77:15 | 77:15 | 1.000 | 1.000 | 1.000 | label_ok_ambiguous |
| `tlog_holdout__38_73_1028803212` | qlab | 38:73 | 15:30 | 15:30 | 1.000 | 1.000 | 1.000 | label_ok_ambiguous |
| `tlog_holdout__38_79_1059280208` | qlab | 38:79 | 15:36 | 15:36 | 0.967 | 0.967 | 1.000 | label_ok_ambiguous |

`pair` is labelled-vs-predicted verse-text similarity. Gemini IDs on duplicate ayahs are arbitrary (first listed location).

## Conclusions

17/21 are verbatim-identical ayah pairs — undecidable from audio. That is the bulk of the “misses”: Ar-Rahman refrain (55:13/30/40/53, 31 copies), Shuʿarāʾ refrain (26:9/122), muqattaʿāt الم (2:1/30:1 and four more), plus two-location twins (81:19=69:40, 37:82=26:66, 37:43=56:12, 21:38=10:48, 83:13=68:15, 8:51=3:182, 37:176=26:204, 70:29=23:5, 77:45=77:15, 38:73=15:30, 38:79=15:36). 2 label errors: `tlog_m043_010_043` and `tlog_m044_010_043` are labelled 10:43 but Gemini transcribes 10:42 (`ومنهم من يستمعون إليك…`) exactly — the reciter said the previous ayah, which our model predicted. 0 real model errors on distinguishable unique text. 2 undetermined: `tlog_m000_100_001` is unique 100:1 that both label and best model already agree on (Gemini confirms 100:1 plus bismillah and the start of 100:2; the miss is sequence/ref scoring, not identity); `tlog_m008_107_001` is a span — Gemini heard 106:4 then 107:1, so the label omits the leading ayah and the model omits the trailing one.
