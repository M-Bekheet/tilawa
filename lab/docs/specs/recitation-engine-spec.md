# Recitation engine — behavioural specification

Implementation-independent description of what the offline Quran recitation engine computes. An engineer who has never seen the reference source should be able to reimplement from this document plus `docs/specs/vectors/` and match those vectors (and, with the same ONNX + corpus + host loop, the benchmark bar below).

**Acoustic model under test:** `interp-gentle-a0.5` int8 Zipformer2-CTC  
(`web/frontend/public/models/zipformer_interp_gentle_a05.int8.onnx`).  
**Corpus:** `quran.json` v2 (`data/prompter/quran.json`).  
**Acceptance bar (already measured on the reference stack):**

| Corpus | Correct |
|---|---|
| v1 | **53/53** |
| v2 | **43/43** |
| v3 | **248/256** |
| qlab | **572/583** |

Machine-readable oracles: `docs/specs/vectors/`. Dumped from the original engine before its removal at `e172b79` (regenerator `dump_vectors.mjs` is gone).

Clock: **CTC output frames run at 25 Hz** (12 output frames per 48 fbank-frame hop of 480 ms). All engine timers are in those frames.

---

## 0. Pipeline

```
PCM 16 kHz float32 [-1, 1]
  → Kaldi log-mel fbank (80 bins, 10 ms hop)
  → streaming Zipformer2-CTC (window T=61, hop 48)
  → greedy CTC tokens {sym, frame, margin}
  → RecitationEngine (searching | tracking)
       searching: 5-gram index + semi-global alignment → locate
       tracking:  per-surah online DP → cursor + per-word verdicts
  → host emission (50 % gate, optional gap-bridge, whole-ayah fallback)
```

The engine is **pure and frame-clocked**. The host owns PCM chunking, ONNX, and what to do with `idle` / `completed` / `locateFailed`.

---

## 1. Kaldi fbank frontend

Numerical reference already locked by `shared/fbank.py` (`torchaudio.compliance.kaldi.fbank`) and `tests/test_fbank_parity.py` (max abs error `< 1e-3` vs the JS port; streaming `--chunk 7680` is **bitwise equal** to one-shot).

### Contract

| | |
|---|---|
| Input | mono float32 PCM in `[-1, 1]`, **16 000 Hz** |
| Output | frames of 80 log-mel bins, float32 |
| Streaming | `acceptWaveform(samples) → Frame[]` (only frames whose end sample now exists) |
| Flush | `inputFinished() → Frame[]` (kaldi `snip_edges=false` total) |
| Reset | drop buffer, `sampleOffset=0`, `framesProduced=0` |

### Constants (exact)

| Name | Value |
|---|---|
| `sampleRate` | 16000 |
| `frameLength` N | 400 samples (25 ms) |
| `frameShift` | 160 samples (10 ms) |
| `numBins` | 80 |
| `fftSize` | 512 |
| `preemphasis` | 0.97 |
| `dither` | 0 |
| `window` | povey: `hann(i)^{0.85}` with `hann(i) = 0.5 - 0.5·cos(2πi/(N-1))`, i = 0..399 |
| `low_freq` | 20 Hz |
| `high_freq` | −400 → Nyquist − 400 = **7600 Hz** |
| `use_energy` | false |
| `snip_edges` | false |
| `remove_dc_offset` | true |
| log floor | `1.1920929e-7` (IEEE float32 ε). `log(max(e, ε))` |

Mel scale (HTK): `1127 · ln(1 + f/700)`.

80 triangular filters equally spaced in mel between 20 Hz and 7600 Hz. Weights live on FFT bins `k = 0..255` (Nyquist excluded). Bin width = `16000/512 = 31.25 Hz`. A bin `k` contributes to filter `b` iff its mel is strictly between that filter’s left and right; weight is the usual tent `(mel-left)/Δ` or `(right-mel)/Δ`. Stored sparse as `(firstBin, weights[])`.

### Frame geometry (`snip_edges=false`)

Frame `f` (0-based) is **centred** at `f·160 + 80` and spans

```
start(f) = f·160 + 80 − 200 = f·160 − 120
end(f)   = start(f) + 400   = f·160 + 280
```

### Streaming readiness

Frame `f` is emitted from `acceptWaveform` iff

```
end(f) ≤ sampleOffset + bufferedLength
```

i.e. the end sample **exists**. Start-of-stream negative indices are **not** a reason to wait.

Worked: `docs/specs/vectors/fbank_readiness.json`

- 279 samples → 0 streaming frames; 280 samples → frame 0.
- 439 → 1; 440 → 2.
- 16 000 samples → 99 streaming frames.

### `inputFinished` flush

```
total = floor( (n + frameShift/2) / frameShift ) = floor( (n + 80) / 160 )
```

Emit every `f` with `framesProduced ≤ f < total`, reflecting past the **true** stream end `n` (not a padded length).

Worked: 16 000 samples → flush total 100 (one extra frame beyond the 99 streaming-ready). 119 samples → flush total 1 even though streaming-ready is 0.

### Edge reflection (kaldi)

When extracting frame `f` with a sample index `s` and available length `n` (`n = +∞` during streaming, `n = trueLength` during flush):

```
while s < 0 or s ≥ n:
    if s < 0: s = -s - 1      # −1 → 0, −2 → 1, …
    else:     s = 2n - 1 - s  # n → n−1, n+1 → n−2, …
```

Then read `samples[s - sampleOffset]`.

### Per-frame DSP (order is mandatory)

1. Gather 400 reflected samples as float64.
2. Subtract mean (DC).
3. Pre-emphasis **in reverse**: `x[i] -= 0.97 · x[i-1]` for `i = N-1 … 1`; then `x[0] -= 0.97 · x[0]` (i.e. `x[0] *= 0.03`).
4. Multiply by the povey window.
5. Zero-pad to 512, **unscaled** radix-2 FFT, power `re²+im²` on bins 0..255.
6. Apply the 80 mel filters; `log(max(energy, 1.1920929e-7))`.

Trim of the sample buffer is an implementation detail (drop samples before `start(nextFrame)` once more than 1600 extra samples have accumulated). It must not change outputs.

### Vectors

- `fbank_synthetic.json` — first **200** frames of `_synth_wave(duration_s=3.7, seed=0)` from `tests/test_fbank_parity.py` (numpy PCG64), chunked at 7680 then flushed. Frame 0 begins `[-3.95553, -3.17001, -2.84194, …]`.
- `fbank_001002.json` — first **5** frames of `benchmark/test_corpus/001002.mp3`. Full-clip parity is the Python test, not this dump.

---

## 2. Streaming Zipformer runner

Black box: k2/icefall streaming Zipformer2-CTC ONNX. The engine does **not** interpret weights. It only has to drive tensors exactly.

Full I/O manifest: `docs/specs/vectors/zipformer_io.json` (99 inputs). Same layout as `zipformer_interp_gentle_a05.io.json`.

### Constants

| | |
|---|---|
| `T` | 61 fbank frames per forward |
| `hop` | 48 fbank frames consumed per forward |
| `featureDim` | 80 |
| `vocabSize` | 251 |
| overlap left in the window | `T − hop = 13` |
| ONNX session | `graphOptimizationLevel = "all"`; EP `cpu` (Node) or `wasm` (browser) |

### Input tensors

`x`: float32 `[1, 61, 80]`, row-major frames.

Every other input is a **state**. Zero-initialise on `reset` (float32 → zeros, int64 → `0n`). After each `session.run`, replace state `name` with output `new_name`. **Refuse** a model that lacks `new_<name>` for any state input (a frozen cache silently wrecks every chunk after the first).

States (all float32 except `processed_lens`):

The 16 encoder layers `ℓ = 0..15` each have `cached_key_ℓ`, `cached_nonlin_attn_ℓ`, `cached_val1_ℓ`, `cached_val2_ℓ`, `cached_conv1_ℓ`, `cached_conv2_ℓ`. Dims vary by layer; copy them from `zipformer_io.json` — do not invent them.

Plus:

| name | dims | dtype |
|---|---|---|
| `embed_states` | `[1, 128, 3, 19]` | float32 |
| `processed_lens` | `[1]` | int64 |

`processed_lens` is **owned by the model**. The runner copies `new_processed_lens` through. Observed on this export: `0, 24, 48, 72, …` (+24 per forward). Do not increment it in host code; a different export may use a different step.

Outputs: `log_probs` float32 `[1, F, 251]` plus `new_*` for every state. **Observed `F = 12`** on every forward of this export (4:1 feature-to-output; 12/0.48 s = 25 Hz). Concatenate `F` across the windows of one `accept` call.

### Buffering

```
buffer.append(incoming fbank frames)
while buffer.length ≥ T:
    x = buffer[0 : T]          # 61 frames
    run ONNX
    emit log_probs[0, :, :]    # F=12 rows
    carry new_* states
    buffer.splice(0, hop)      # drop 48, keep the rest (always ≥ 13)
```

A call that does not bring the buffer to 61 frames emits **zero** output frames.

**No pad, no flush of a partial window.** Leftover `< T` frames at end-of-stream are discarded. The host must append silence (2.0 s in the shipped loop) so hop-aligned windows keep coming. After the 2 s drain, leftover is still typically 24–35 frames (`ctc_*.json` `leftoverFbankFrames`) — that tail is never scored. That is required behaviour, not a bug.

`reset`: zero all states **and** clear the fbank-frame buffer.

### Worked (480 ms host chunks)

7680 samples → 47 streaming fbank frames (frame 0 needs 280 samples, so a 480 ms chunk is 47 not 48). Second chunk brings the buffer ≥ 61 → first forward, leftover 47. Steady state: each later 48-frame chunk triggers exactly one forward (`F=12`). See `zipformer_windows.json`.

---

## 3. Greedy CTC decoder

Vocab: `docs/specs/vectors/tokens.json`. **Blank id = 250** (last). `sym` is a (possibly multi-character) string: letters, harakat, shadda-as-doubling (`ممم`), madd-as-repetition (`اااا`, `ۦۦۦۦ`), sukun/qalqalah `ڇ`, ghunna `ۜ`, etc.

### Contract

```
consume(logProbs, frames, classes=251) → Token[]
flush() → Token[]          # open run at end-of-stream
reset()                    # previousBest=blank, frameIndex=0, run=null
framesDecoded              # == frameIndex, counts every row including blanks
```

Token: `{ sym: string, frame: int, margin: float }`.

`logProbs` is row-major `frames × 251`, natural log-softmax.

### Per output frame `t`

Argmax and runner-up over the 251 logits. Ties: **strict `>`**, so the **lowest class index** wins.

Let `best` be that class, `p1` / `p2` its log-probs.

Carry `previousBest` (initially blank) and an optional open `run = {id, frame, p1, p2}`.

1. If `best ≠ blank` **and** `best ≠ previousBest`: emit the open run (if any); start a new run at **this** frame with these `p1,p2`.
2. Else if `best ≠ blank` **and** `best === previousBest` **and** a run is open **and** `p1` is strictly greater than the run’s stored `p1`: **replace** the run’s `p1,p2` (peak-frame margin — never take the transition frame).
3. Else if `best === blank` **and** a run is open: emit the run; clear it.
4. Else: no emission (blank continuing, or same-token non-peak).
5. `previousBest ← best`; `frameIndex++`.

A token is therefore emitted when its run **ends** (blank or a different class). Tokens are never revised; the host never retracts CTC output.

`flush`: if a run is open, emit it, clear it, set `previousBest = blank`.

### Margin

```
margin = exp(p1) − exp(p2)     # linear-prob gap at the peak frame of the run
```

### Engine-side splitting (mandatory)

The engine does **not** consume tokens as atoms. For each token `t`:

```
for each UTF-16 code unit ch of t.sym:     # all phonemes are BMP
    push { ch, frame: t.frame, margin: t.margin }
```

So token `"ءَ"` becomes two heard chars sharing frame and margin. `heardTotal` and the tracker see **characters**, not tokens.

### Worked

`ctc_001002.json`: 18 tokens, 180 output frames, transcript

`ءَلحَمدُلِللَااهِرَببِلعَاالَمِۦۦۦۦن`

First token `{sym:"ءَ", frame:7, margin≈0.9976}`. Blank-id 250 never appears as `sym`.

---

## 4. Phoneme corpus

### Input schema (`quran.json` v2)

```
{ "v": 2, "surahs": [ { "n", "name", "nameEn", "ayahs": [
    { "n", "m", "w": [ [mushafGlyphs, phonemes, plain], ... ] }
] } ] }
```

Reject `v ≠ 2`. Reject an ayah whose `n` is not `index+1`. Require 114 surahs.

Every phoneme character is a BMP code unit; `text[i]` is one phoneme. Offsets are in **characters**, not bytes.

### Flattening

Walk surahs 1..114, ayahs in order, words in order. Concatenate each word’s `phonemes` into one string `text`. Parallel tables:

| table | meaning |
|---|---|
| `wordStart[w]` | char offset of word `w`; `wordStart[wordCount] = text.length` |
| `wordSurah[w]`, `wordAyah[w]`, `wordInAyah[w]` | 1-based surah/ayah, 0-based word-in-ayah |
| `mushaf[w]`, `plain[w]` | display / Unicode |
| `ayahFirst[surah-1][ayah-1]` | global word index of that ayah’s first word |
| `ayahWords[surah-1][ayah-1]` | word count |
| `markers[surah-1][ayah-1]` | ayah-end ornament `m` |
| surah record | `{n, name, nameEn, ayahCount, firstWord, endWord}` (`endWord` exclusive) |

Observed on the shipped corpus: **77 433** words, **652 401** phoneme chars (`docs/specs/vectors/corpus.json`).

Fatiha 1:1 phonemes (4 words): `بِسمِ` + `للَااهِ` + `ررَحمَاانِ` + `ررَحِۦۦۦۦم`.

### Lookups

- `wordAt(offset)`: binary search on `wordStart`; clamp `<0 → 0`, `≥ text.length → wordCount-1`. Search invariant: largest `w` with `wordStart[w] ≤ offset`.
- `wordIndex(surah, ayah, word)` = `ayahFirstWord + word` (throws if out of range).
- `hasAyah` is the non-throwing bounds check (surah in 1..114, ayah in 1..ayahCount).

There is **no** separator between words or ayahs or surahs in `text` itself. Surah isolation is the search index’s job (§7).

---

## 5. Phoneme cost table

A dense `size × size` float32 matrix, `size = 52 = 51 + 1 unknown`.

### Alphabet (order = id, exact string)

```
ءابتثجحخدذرزسشصضطظعغفقكلمنهويۥۦں۾ٲأإآؤئٱىَُِڇؙۣ۪ٞۜـ
```

51 characters: 41 base letters + 3 short vowels `َُِ` + 7 other marks `ڇؙۣ۪ٞۜـ`.  
`unknownId = 51`. Characters outside the alphabet (including Latin `x` and `ة`) map to 51.

Store the table in **float32**. `0.1` is `0.10000000149011612`. Oracle: `docs/specs/vectors/cost_table.json` (`matrix[heardId][expectedId]`).

### Canonical map (applied before class tests; cost 0 if equal after)

| from | to |
|---|---|
| `ۦ` | `ي` |
| `ۥ` | `و` |
| `ں` | `ن` |
| `۾` | `م` |
| `ٱ` | `ا` |
| `ى` | `ي` |

### Substitution cost `charCost(heard, expected)`

1. Identical code units → **0**.
2. Canonical forms equal → **0**.
3. If either is a **mark** (`َُِ` or `ڇؙۣ۪ٞۜـ`):
   - both marks and both short vowels → **0.1**
   - both marks otherwise → **0.25**
   - mark vs letter → **1**
4. Both in hamza family `{ء أ إ آ ا ؤ ئ ٲ}` → **0.1**
5. Acoustic neighbours → **0.25**
6. Else → **1**

Unknown id vs anything (including itself) → **1**.

### Neighbour relation (after canonicalisation)

All unordered pairs inside each group, plus the extra pairs.

Groups: `ذدضتط`, `ظزذصسث`, `جزش`, `ةهت`, `قكغ`, `فبم`.

Extra pairs: `(ه,ح) (غ,خ) (ء,ع) (ن,م) (ن,ل) (ظ,ض)`.

**Trap:** `ة` is listed in `ةهت` but is **not** in the alphabet, so `ة` vs `ه` in the dense table is **1** (unknown). The useful pair from that group is `ه`/`ت` = 0.25.

### Insert / delete

Always **1**. Never graded. Used by alignment, search verification, and the tracker.

### Encode

`encode(s)` → `Uint8Array` of ids, one per char. Used everywhere a string enters DP.

### Worked

| heard | expected | cost |
|---|---|---|
| `ا` | `ا` | 0 |
| `ۦ` | `ي` | 0 |
| `أ` | `ء` | 0.1 |
| `ا` | `أ` | 0.1 (hamza family) |
| `َ` | `ُ` | 0.1 |
| `س` | `ص` | 0.25 |
| `ب` | `َ` | 1 |
| `ة` | `ه` | 1 (unknown) |

---

## 6. Alignment

Two algorithms. Insert = delete = 1; substitute = table.

### 6.1 `normalizedDistance(a, b)` — global weighted Levenshtein

```
if |a|=|b|=0: 0
if exactly one empty: 1
else: DP / max(|a|, |b|)
```

Standard NW recurrence on a **float32** row (assign into a `Float32Array` cell, then compare the stored value). Tie-break inside a cell: take the cheapest of diagonal (substitute), up (insert), and left (delete). On a cost tie, left wins over both diagonal and up; up wins over diagonal when both beat left. Only the cost is used; there is no traceback. Oracle for the store width: `alignment.json` (hamza 0.10000000149011612, 1:1-vs-1:2).

Worked pairs: `docs/specs/vectors/alignment.json` (20 pairs). Examples:

- identical `بِسمِ` → 0
- empty vs nonempty → 1
- `أ` vs `ء` → 0.1 (float32)
- `بَ` vs `بُ` → 0.05
- `س` vs `ص` → 0.25
- `بِسمِللَااهِ` vs `بِسمِ` → 0.5833333333333334
- 1:1 vs 1:2 → ≈ 0.4639

### 6.2 `alignGlobal(heard, ref[from, to))` — NW with traceback

Used by verdicts to assign heard chars to reference positions.

Init: first row is prefix-deletes (`C[0][j] = j`); first column is prefix-inserts (`C[i][0] = i`). The cost matrix is **float32**; traceback compares after the store (same as §6.1).

Traceback at each cell prefers diagonal (substitute a heard char onto a ref char), then up if strictly cheaper (insert heard / skip a heard char), then left if strictly cheaper (delete a ref char). Diagonal therefore wins ties, then up, then left.

Traceback yields `assign[i] = refIndex` or `-1` if that heard char was inserted.

### 6.3 `alignSemiGlobal(query, ref[from, to), headSkipCost=0.5)` — search verify

The query must be consumed to its end; it may start anywhere in the ref slice; it may end anywhere; it may skip a **prefix of itself** at `0.5` per char (cheaper than a mismatch, dearer than typical garble — drops an unalignable استعاذة head, keeps a garbled Quran stretch).

State per cell: cost, `refStart` (where this alignment began in the slice, 0-based into the slice), `queryStart` (how many query chars were skipped as head).

Init row i=0: cost 0, `start=j`, `q=0` for all j (free start in the ref).

Column j=0 at query i: `cost = i · 0.5`, `start=0`, `q=i`.

Recurrence at (i, j), query char `h`, ref char `ref[from+j-1]`:

```
diag  = prev[j-1] + cost(h, refchar)     inherit start/q from diag
up    = prev[j]   + 1                    inherit from up
left  = cur[j-1]  + 1                    inherit from left
fresh = i · 0.5                          start = j, q = i   # skip i query chars, begin at ref j
```

Take the minimum; **strict `<`**, so preference on ties: **diag, then up, then left, then fresh**. Costs live in **float32** rows; `i · 0.5` and the three step costs are rounded to float32 before the compare (oracle: `search.json` fatiha@200, 0.4439999771118164).

After the last query row, pick `j* = argmin_j C[n][j]` (ties → **smallest j**).

Return:

```
cost
distance = (n == 0) ? 1 : cost / n     # divide by FULL query length, not the aligned tail
refStart = from + start[j*]
refEnd   = from + j*
queryStart = q[j*]
```

Worked: query = BASMALA, ref = 1:1+1:2, `headSkipCost=0.5` → `{cost:0, distance:0, refStart:0, refEnd:32, queryStart:0}`.

---

## 7. Search / locate

### 7.1 Index

Build a 5-gram inverted index over a **separated** id stream, not raw `text`.

Between consecutive surahs insert **8** copies of `unknownId` (unmatchable). An alignment cannot walk from the end of one surah into the next.

```
GRAM = 5
BUCKET_BITS = 18          # 262 144 buckets
WINDOW_BITS = 5           # 32-char vote windows
MAX_POSTINGS = 400        # skip a gram whose bucket is fatter than this
CANDIDATE_WINDOWS = 24
VERIFY_MARGIN_BEFORE = 16
VERIFY_MARGIN_AFTER = 32
MIN_ALIGNED = 20
SURAH_GAP = 8
SHORT_QUERY = 100         # tail re-search
PREAMBLE_MAX_DISTANCE = 0.3
```

Hash of 5 consecutive unit ids at `at` — **FNV-1a 32-bit** as implemented by JS bitwise ops:

```
h = 2166136261            # ToInt32 → -2128831035
for k in 0..4:
    h = h XOR ids[at+k]   # JS ToInt32
    h = imul32(h, 16777619)
bucket = (h as uint32) AND (2^18 - 1)
```

`imul32` is C `int32_t` multiply (JS `Math.imul`).

Worked (`hash_examples.json`):

| 5-gram | ids | bucket |
|---|---|---|
| `ءَعُۥ` | 0,41,18,42,29 | 117641 |
| `بِسمِ` | 2,43,12,24,43 | 170063 |
| `الحمد` | 1,23,6,24,8 | 62589 |
| `ءَلحَ` | 0,41,23,6,41 | 174630 |

Postings: CSR (`bucketStart`, `postings`) of starting offsets in the separated stream.

Map a separated offset back to a corpus char offset by binary-searching surah starts, then clamping into that surah’s length (offsets that land in a gap clamp to the **end** of the previous surah). If `refEnd ≤ refOffset` after mapping, the alignment sat only on separators — drop it.

### 7.2 Vote → verify

Query shorter than `searchMinChars` (12) → `{hits:[], decisive:false}`.

For each 5-gram of the encoded query, if the bucket has `≤ 400` postings, each posting `p` votes for window

```
w = (p − u) >> 5     # arithmetic shift, can be negative
```

Sort windows by **votes desc, then window index asc**. Keep 24.

For each: `start = w << 5`, verify slice `[max(0,start-16), min(len, start+|q|+32))` with **semi-global** alignment. Drop if `|q| − queryStart < searchMinChars`. Drop separator-only hits.

Sort verified by **distance asc, then wordIndex asc**. Greedily keep hits that do **not** overlap a better hit’s `[refOffset, refEnd)`.

### 7.3 Hint (optional `{surah, ayah}`)

Among hits with `distance ≤ best.distance + searchDecisiveMargin` (0.1), if any share the hinted surah, pick the one whose `wordIndex` is nearest that ayah’s first word (tie → smaller `wordIndex`). That same gate — “at least one near hit shares the hinted surah” — is what switches the rival rule: the rival is then the first hit that is **not** the chosen best **and** sits **outside** the near-cap, so a hinted surah can make the search decisive by silencing same-distance copies. If no near hit shares the hinted surah (or there is no hint), the rival is simply the second-ranked hit.

### 7.4 Decisive

```
aligned = queryLength − best.queryStart + alignedBonus
decisive iff
    best.distance ≤ 0.35
    AND aligned ≥ 20
    AND (no rival OR rival.distance − best.distance ≥ 0.1)
```

Return at most `limit` hits (default 3), best first.

### 7.5 Tail re-search

If the full query is **not** decisive **and** `|query| > 100`, also search `query.slice(-100)`. If **that** is decisive, add `|query|-100` to every hit’s `queryStart` and return it. Otherwise return the full-query result.

### 7.6 Preamble stripping

**Growing istiʿādha.** If `|query| > |ISTIADHA|+4` → not growing. Else compare `query` to `ISTIADHA[:min(|ISTIADHA|,|query|)]` with `normalizedDistance ≤ 0.35`. Growing → **no lock** (`hits=[], decisive=false`). This exists because the phrase nearly matches 16:98 and `أعوذ بالله` is literal Quran at 2:67; a genuine reciter of those diverges within a few chars.

**Strip.** Try phrases in order `[ISTIADHA, BASMALA]`. For each, lengths `L−4 … L+4` clipped to the remaining query; pick the length of **lowest** `normalizedDistance`; accept if that distance `≤ 0.3`. Advance `offset`. If the accepted phrase is BASMALA, record `basmala=true` and `basmalaOffset` (offset **before** consuming it).

Constants (exact phoneme spellings):

```
ISTIADHA = "ءَعُۥۥذُبِللَااهِمِنَششَييطَاانِررَجِۦۦم"     # 40 chars, not Quran text
BASMALA  = "بِسمِللَااهِررَحمَاانِررَحِۦۦۦۦم"           # 32 chars = 1:1
```

After strip, if the remainder `< 12` chars → no lock.

**Basmala-is-Quran special case.** If a basmala was stripped, first search **from `basmalaOffset`** (basmala included). If that is decisive **and** `hits[0].queryStart ≤ 2`, keep it (this is 1:1, not 27:30 / 6:45 / 37:182). Add `basmalaOffset` to every `queryStart`.

Otherwise search the remainder. Then, for each hit: if it is **1:2 word ≤ 3** and a basmala was stripped, collapse the hit to corpus word 0 (1:1), `refOffset=0`, `queryStart=basmalaOffset`. If that hit was the best, `alignedBonus = offset − basmalaOffset` (istiʿādha length). Other hits just add `offset` to `queryStart`. Then `decide(rest.length, bonus)`.

### 7.7 Worked (`search.json`)

| query | decisive? | why |
|---|---|---|
| 001002[:12] `ءَلحَمدُلِلل` | no | `aligned=12 < 20`; also three distance-0 copies (1:2, 6:1, 14:39) |
| 001002[:20] | no | rival gap 0.025 < 0.1. **With Fatiha hint: yes** |
| 001002 full (36 chars, exact 1:2) | no | rivals 6:45 / 37:182 at 0.0139. Hint makes it decisive |
| 001001 full (exact BASMALA) | no | stripped to empty remainder |
| growing istiʿādha 16 chars | no | growing gate |
| Fatiha multi [:100] | **yes** | 1:1 distance 0, rival 27:30 at 0.355 (gap ≥ 0.1), aligned ≥ 20 |

This is why the host needs a whole-ayah fallback for short clips. The engine is a **prompter**: it refuses to lock on an ambiguous 1:2.

---

## 8. Tracker

One contiguous reference: a **whole surah** (`fromAyah=1 … toAyah=ayahCount`). Heard chars update **one column** of an edit-distance table over that reference. The cursor is the cheapest cell (ties → nearest the previous cursor). Nothing “decides” to move: jumps, repeats and skips are just cheaper or dearer paths. Verdicts are traced later from the cursor trail.

### Config used here

| field | value | role |
|---|---|---|
| `jumpCost` | 12 | restart at a word start that is not a same-ayah repeat |
| `repeatCost` | 10 | restart at a word start in the current ayah at or before the cursor |
| `lostWindow` | 120 | chars in the slow cost-rate window |
| `lostRate` | 0.35 | lost if rate ≥ this |
| `holdWindow` | 30 | fast window |
| `holdRate` | 0.45 | held if rate ≥ this |

### Initial column

Let `len` be the surah’s phoneme length, `start` the char position of the start word. The live column is a **float32** vector (oracle: `events_ea_alafasy_multi` cursor.cost / `alignment.json`).

- Word-start positions: `0` if this is the start word, else `jumpCost`.
- Other positions: `column[m-1] + 1` (delete from the previous word start).
- Then `column[len] = min(column[len], column[len-1]+1)`.
- `cursorCell = start`, `cursorLocalWord = -1`, `cursorCost = 0`.

### Feed one heard char `h`

Let `prev` be the current column. `colMin = min(prev)`. `jump = colMin+12`, `repeat = colMin+10`. `cursorAyah` is the ayah of `cursorLocalWord`, or −1 if none.

```
next[0] = prev[0] + 1                          # insert
if word-start at 0:
    next[0] = min(next[0], ayah[0]==cursorAyah ? repeat : jump)

for m = 1..len:
    substitute from prev[m-1] plus the phoneme cost
    insert  from prev[m] plus 1
    delete  from next[m-1] plus 1
    next[m] is the cheapest of those three. Insert wins a tie with
    substitute; delete wins only if strictly cheaper than that pair.
```

Then **restart floor** at every word start, applied *after* the sweep so deletions can propagate from a restart:

```
for each word-start m:
    restart = (m ≤ cursorPos AND ayah[m]==cursorAyah) ? repeat : jump
    if restart < next[m]:
        next[m] = restart
        for j = m+1 .. len while next[j-1]+1 < next[j]:
            next[j] = next[j-1] + 1
```

Before any cursor word, `cursorAyah = -1`, so every restart is a jump.

Argmin of `next`: ties → **nearest** `cursorPos` (`|m − cursorPos|` strictly smaller). Equidistant → **lower index** (scan left to right).

Then:

```
cursorCell = bestCell
cursorLocalWord = (bestCell == 0) ? 0 : wordOfPos[min(bestCell, len) - 1]
```

After the first char, `cursorLocalWord` is never −1. Cursor `wordIndex = firstWord + cursorLocalWord`.

Append `bestCell` to `trail`, `bestCost` to `costs`.

### Cost rate / lost / held

```
costRate(window):
    if n < 24: return null          # n = heard char count
    w = min(window, n)
    before = (n-w > 0) ? costs[n-w-1] : 0
    return (costs[n-1] − before) / w
```

- **lost** (tracker): `costRate(lostWindow=120) ≥ 0.35`
- **held** (engine, same formula): `costRate(holdWindow=30) ≥ 0.45`

`reachedEnd`: `cursorCell ≥ len − 1`.

### Retract `n`

If `n ≤ 0`, do nothing. Otherwise increment `revision`. Target length = `max(0, heard.length−n)`. Restore the latest snapshot with `length ≤ target` (snapshots every **32** chars, keep **16**), else `resetColumn`. Replay the chars between snapshot and target. Result ≡ never having fed the retracted tail.

---

## 9. Verdicts

Trace the trail into per-word states `ok | unsure | wrong | skipped | pending`.

### Segment the trail

A **run** starts wherever `trail[g] < trail[g-1]` (cursor moved backwards: repeat or jump back). Inside a run, cut every **300** heard chars. Each segment records `heardFrom, heardTo, refFrom, refTo, run`.

`refFrom` = start of the word containing the char just before `trail[segStart]` (0 if cell≤0). `refTo` = `trail[end-1]`.

Context: only the first segment of **run 0** (the very first run) uses no extra heard context. Every later run start, and later segments of any run, set `contextFrom = max(prevSegStart, segStart − 6)`.

The last segment is **open** (not cached). Closed segments are cached by `(contextFrom, heardTo, refFrom, refTo)`. A tracker `revision` change (retract) drops the cache.

### Spans

`alignGlobal` the context-extended heard ids onto `ref[refFrom, refTo)`. Each substituted heard char is assigned to `wordOfPos[refIndex]`. Span of a word = `[first assigned index, last+1)` (from the **context** index space). Later segments **overwrite** earlier spans for the same word (last writer wins).

### Per-word judgement

Let `minWord`/`maxWord` be the min/max word that has a span. `cursorPending = !reachedEnd && !settled`. `dwell = settled ? 0 : commitDwell(6)`.

A word is **pending** if any of:

- it is the cursor word and `cursorPending`
- its span ends after `heardLen − dwell`
- its span’s `run < lastRun` **and** `word ≥ cursorWord` (old-run colours at/after the new cursor stay pending)

`heardCount = span.to − span.from` (0 if no span). `expLen` = that word’s phoneme length.

If **not** pending and `heardCount < 0.34 · expLen`:

- if `minWord < w < maxWord` → emit **skipped** (distance 1, margin 0, `heardRatio = heardCount / expLen`)
- else emit nothing (`continue`)

If no span: emit nothing.

Otherwise distance = `normalizedDistance(heardSlice, expectedPhonemes)`.

**Waqf (pausal form)** — only if `distance > okDistance (0.15)` and a pausal encoding exists (§9.1) **and** a real pause follows the word. Then `distance = min(distance, distance_to_pausal)`. This can only **remove** false reds.

Margin of the word = mean of per-char margins on the span.

State:

```
pending if pending
else ok      if distance ≤ 0.15
else unsure  if distance ≤ 0.4  OR  margin < 0.35
else wrong
```

`settled=true` once the decoder has been silent `settleFrames=25` (1 s) since the last non-empty token batch (`framesDecoded − lastCharFrame`). The cursor word and the tail then judge like everything else.

### 9.1 Waqf rules (`pausalPhonemes(phonemes, plain, atAyahEnd)`)

Return the stopped pronunciation, or **null** (= judge flowing form only). Null is the safe answer.

1. **`atAyahEnd` → null.** The corpus already stores ayah-end words paused. No string test may stand in for this flag (e.g. `مُّسْتَمِرٍّ` stored `ممممُستَمِرر` is indistinguishable from an assimilated noon).
2. `|phonemes| < 2` → null.
3. If `plain` contains a tanween mark `ً` / `ٌ` / `ٍ`:
   - Strip a trailing run of **one repeated** symbol that is in cluster `{ن ں م ۾ و ۥ ي ۦ ل ر}`. Mixed runs are **not** stripped (would eat the stem).
   - The stem must end with the tanween’s own vowel: `َ` for `ً`, `ُ` for `ٌ`, `ِ` for `ٍ`. Else null (usually an ayah-end form already paused while plain still shows tanween).
   - If fathatan **and** `plain` does **not** contain `ة`: `stem + "اا"` (نَارًا `نَاارَںںں` → `نَاارَاا`).
   - Else (dammatan, kasratan, or fathatan on `ة`): drop the final vowel (`لَهَبٍ` → stem without kasra; `ةً` likewise — ta marbuta becomes “ah”, left to the ه/ت neighbour cost).
4. Else if the last phoneme is a short vowel `َُِ`: drop it (إسكان الموقوف عليه). `بِسمِ` → `بِسم`.
5. If the result is empty or equal to the flowing form → null.

Waqf is used **only** in verdicts, and only when a pause is detected. The online tracker always scores the flowing form.

### 9.2 Pause detection (`stopBoundary`)

Walk heard chars from `span.from+1` through `min(heard.length, span.to+4)`:

- if `i === heard.length` → `i` (stream ended: the silence now settling it)
- if `heard[i].frame − heard[i-1].frame ≥ settleFrames` (`cfg.settleFrames`, default 25) → `i`
- else −1 (no stop; do not apply pausal)

If the stop index ≠ `span.to`, re-slice uttered chars to `[span.from, stop)` (alignment against the flowing form can glue a following utterance onto the stopped word).

Worked: `docs/specs/vectors/waqf.json`.

---

## 10. Engine state machine

Two states: `searching` | `tracking`. Frame clock = CTC `framesDecoded` (25 Hz).

### `DEFAULT_CONFIG` (every field)

```
jumpCost: 12
repeatCost: 10
commitDwell: 6
okDistance: 0.15
unsureDistance: 0.4
minHeardFraction: 0.34
minMargin: 0.35
lostWindow: 120
lostRate: 0.35
holdWindow: 30
holdRate: 0.45
searchMinChars: 12
searchQueryChars: 250
searchDecisiveDistance: 0.35
searchDecisiveMargin: 0.1
searchEveryFrames: 25          # 1 s
searchEveryChars: 12
locateFailedFrames: 375        # 15 s
relocateEveryFrames: 37        # 1.48 s
relocateQueryChars: 100
relocateMaxDistance: 0.3
relocateRateMargin: 0.12
idleFrames: 200                # 8 s
maxStruggles: 3
settleFrames: 25               # 1 s
```

Also: `BUFFER_CAP = 1000` heard chars (the search buffer, not `heardTotal`).

### Events

| type | payload | when |
|---|---|---|
| `located` | `{surah, ayah, word, replayed}` | first lock from search |
| `relocated` | `{from:{surah,ayah}, to:{surah,ayah,word}}` | lock onto a **different surah** |
| `cursor` | `{surah, ayah, word, wordIndex}` | cursor word index changed, and not `held` |
| `verdicts` | `{changes: WordVerdict[]}` | any word’s state changed vs last emission, and not `held` |
| `lost` | — | tracker.lost became true (edge; cleared when not lost) |
| `idle` | `{reason: "silent"\|"lost"}` | see timers |
| `completed` | `{surah}` | cursor at end of surah **and** last word’s verdict is not pending (once) |
| `locateFailed` | — | still searching after 375 frames (once per search) |

### `feed(tokens, framesDecoded)`

If `framesDecoded < previous` (decoder reset): rebase `searchStartFrame`, `lastSearchFrame`, `lastRelocateFrame`, `lastProgressFrame`, `lastCharFrame` to the new clock. Then store `framesDecoded`.

Expand tokens to chars (§3). If any chars: `lastCharFrame = framesDecoded`, `heardTotal += n`, push onto the 1000-char ring buffer.

Dispatch to searching or tracking.

### Searching

Due when `buffer.length ≥ 12` **and** (`charsSince last search ≥ 12` **or** (`charsSince > 0` and `framesSince ≥ 25`)). Query = last **250** chars of the buffer. If `index.search(query, hint)` is decisive: `lock(hit.wordIndex, buffer from queryStart, "located")`.

`heardTotal` (not `buffer.length`) is what “chars since” uses — once the ring saturates, length is constant.

After 375 frames from `searchStartFrame` with no lock: emit `locateFailed` once. Keep listening.

### `lock(wordIndex, replay, how, from?)`

Build a Tracker on that **whole surah**, start word = `wordIndex`. State → `tracking`. Reset lost/struggle/completed/relocate-candidate/last-cursor/last-states. Emit `located` or `relocated`. **Then** `tracker.feed(replay)` and emit tracking events (cursor/verdicts/completed). `replayed` is the char count, not the token count. `how` is `"located"` or `"relocated"`. On `"relocated"` the host may snapshot the abandoned tracker first (`onBeforeRelocate`).

### Tracking

Feed the new chars. Emit tracking events. Edge-trigger `lost`.

Every 37 frames, a relocate tick (always advance the tick baseline, even when `stay`):

- Snapshot `heardSinceTick = heardTotal − lastStruggleChars`, then set `lastStruggleChars = heardTotal`. This counts chars since the previous **tick**, not chars in the current `feed` call.
- **`stay` on:** zero the struggle counter; do not search; fall through to silent-idle.
- **else `maybeRelocate`:** need `buffer ≥ 12`. Search last **100** chars, `limit=1`, **no hint**. Let `rate = costRate()`. Candidate = hit’s `(surah,ayah)` or null. `agrees` = candidate equals the **previous** tick’s candidate (same surah **and** ayah). Store candidate. Relocate only if **all** of:
  - a hit exists
  - `rate ≠ null` and `rate ≥ 0.35`
  - `hit.surah ≠ current surah` (never relocates inside the loaded surah)
  - `hit.distance ≤ 0.3`
  - `hit.distance + 0.12 ≤ rate` (equivalently not `hit.distance + 0.12 > rate`)
  - `agrees` (must see the same candidate on **two consecutive** ticks)
- If that moved: first keep this tick’s already-computed cursor/verdict/`lost` events for the **abandoned** surah, then append the relocation events (`relocated` + new-surah tracking). Skip struggle/idle on this call.
- Else if `heardSinceTick > 0`: `struggles = (lost || held) ? struggles+1 : 0`. If `maxStruggles > 0` and `struggles ≥ 3`: emit `{idle, reason:"lost"}`, reset struggles, poke `lastProgressFrame` so a silent-idle does not fire on the next feed. (`maxStruggles=0` disables this.)

Silent idle: if `framesDecoded − lastProgressFrame ≥ 200`, emit `{idle, reason:"silent"}` and bump `lastProgressFrame`. `lastProgressFrame` advances on a real cursor move and on a non-pending verdict change that is **not** a settle-on-silence.

### `held` freezes the page

While `held`, `trackingEvents` returns **nothing** — no cursor, no verdicts, no completed. The tracker still consumes audio. `lastCursorWord` / `lastStates` are **not** updated, so when the rate drops the next emission is a single jump to the truth rather than a replay of the wander. Measured intent: talking after recitation must not walk the next ayah.

### `startSearch`

State → searching, drop tracker, clear buffer, `heardTotal=0`, rebase search frames to **current** `framesDecoded`, clear locateFailed/relocate/struggles/lastStates. Does **not** reset `framesDecoded`. The host that calls this **must** also reset fbank+decoder+runner; the next `feed` then sees a smaller frame clock and rebases the rest.

### `track(surah, ayah, word=0)`

Direct lock at that word, empty replay, `how="located"`.

### `setHint({surah, ayah}|null)` / `setStayOnSurah(bool)`

Hint is passed into searching (and **not** into relocation). Stay: locate once, never relocate, never `idle/lost` from struggles. **Silent idle is deliberately left on.**

### Worked

- `events_001002.json` / `events_001001.json`: stay in `searching`, **zero** engine events, 180 / 192 frames (`< 375`, so no `locateFailed`). Required: 1:2 is not decisive (duplicate ayahs); 1:1 is stripped as basmala.
- `events_ea_alafasy_multi_001_001_007.json`: `located` 1:1 word 0, `replayed: 54`, then cursor/verdicts, `completed` 1. Final cursor 1:7 word 8, cost 0. Settled: all 29 Fatiha words `ok`.

---

## 11. Host emission policy

Not inside the engine. Shipped in `zipformer-emission.ts` (browser) and `experiments/zipformer-ctc/harness.ts` (Node benchmark). Vectors: `host_*.json`.

### Chunking

- 7680 samples = **480 ms** per `acceptWaveform`.
- PCM is padded with **2.0 s** of zeros, then `inputFinished`, then CTC `flush`.
- Recognize mode: on `idle` or `completed`, snapshot settled verdicts (`verdicts(true)`), `startSearch()`, reset fbank+decoder+runner. Stay mode ignores `idle` for that reset.
- On `relocated`, snapshot the **discarded** tracker’s tallies before the new lock (so a surah change does not lose scored ayahs).

### Per-ayah tallies

From a verdict snapshot, count `ok/unsure/wrong/skipped/pending` per `surah:ayah`. `words` = corpus ayah word count. `firstSeen` = insertion order.

Live browser path **accumulates** snapshots (relocation dumps add; `mergeTallies` adds the current snapshot on top). The Node harness **replaces** via `tallyAyahs` into a map that already has that ayah (it **adds** counts — same as accumulate). Implementers matching the benchmark must follow the harness: dump on relocate / idle / completed / end, summing snapshots.

### Gate (ayah recited)

```
MIN_WORD_FRACTION = 0.5
ok + unsure ≥ max(1, 0.5 · words)     # one-word ayah: need ≥ 1 ok/unsure
AND wrong ≤ ok + unsure
```

Emit gated ayahs in `firstSeen` order, once.

Confidence of an ayah = `(ok+unsure)/words`. Sequence confidence = mean of gated ayah confidences.

### `PROMPTER_ALLOW_GAPS` (harness; default **off**)

If env `= "1"`, also emit a below-threshold ayah of `words ≤ 3` when:

- it is not already emitted
- `ok+unsure ≥ 1` and `wrong ≤ ok+unsure`
- **both** `surah:ayah-1` **and** `surah:ayah+1` are already in the emitted set (bridge, not a prefix fill)

### Whole-ayah fallback (on by default; `PROMPTER_FALLBACK=0` disables)

Runs only if **no** ayah passed the gate.

1. `stripPreambles(transcript)`.
2. If a basmala was stripped **and** remainder `< 12` → `{surah:1, ayah:1, distance:0, how:"basmala"}`.
3. Else encode remainder if `|rest|≥3`, else the full transcript. Scan all 6236 ayahs with a length gate `|a| ≰ 2.5|q|+8` (either way). Nearest `normalizedDistance`. Accept if `≤ 0.5`.

Worked:

- 001002: engine never locks; fallback `{1,2, distance:0, how:"whole-ayah"}`.
- 001001: engine never locks; fallback `{1,1, distance:0, how:"basmala"}`.
- Fatiha 1–7: all seven ayahs 100 % ok; fallback null.

Browser `shouldRunFallback` is `emitted.length === 0` (same idea). `fallbackConfidence = clamp(1 − distance, 0, 1)`.

---

## Test vectors index

| file | contents |
|---|---|
| `fbank_synthetic.json` | 200 frames, numpy seed 0 |
| `fbank_001002.json` | 5 frames of 001002.mp3 |
| `fbank_readiness.json` | streaming vs flush frame counts |
| `zipformer_io.json` | full ONNX I/O manifest |
| `zipformer_windows.json` | T/hop, leftover rule, processed_lens +12-frame probe |
| `tokens.json` | 251-symbol vocab, blank 250 |
| `ctc_{clip}.json` | full `{sym,frame,margin}` streams |
| `cost_table.json` | alphabet, 52×52 matrix, worked charCosts |
| `alignment.json` | 20 normalized distances + one semi-global |
| `hash_examples.json` | FNV-1a buckets |
| `search.json` | 10 queries, hits, decisive, Fatiha-hint variant |
| `corpus.json` | counts, Fatiha 1:1 words, `wordAt` clamps |
| `waqf.json` | pausal derivations |
| `default_config.json` | engine config |
| `events_{clip}.json` | per-chunk `engine.feed` logs (480 ms + 2 s drain) |
| `host_{clip}.json` | gate, tallies, fallback |
| `meta.json` | paths, acceptance bar |

Clips: `benchmark/test_corpus/001002.mp3`, `001001.mp3` (medium v1), `benchmark/test_corpus_v3/ea_alafasy_multi_001_001_007.wav` (multi).

The vectors are frozen. They were dumped from the original engine before its
removal at `e172b79`; `dump_vectors.mjs` is no longer in the tree.

A clean-room implementation is correct when it bit-matches (or, for fbank vs Python, max-abs `< 1e-3`) these JSON files and, with the same host loop, scores **53/53, 43/43, 248/256, 572/583**.
