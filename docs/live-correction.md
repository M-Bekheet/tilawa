# Live correction

The demo defaults to Tracking and remembers the mode in `tilawa-mode`. Correction
is live, uses Zipformer, and keeps all inference/audio local.
Both modes use the same engine. There is no second feedback-timing picker.

## Evidence

`CorrectionController` and `possibleWordIssues` in `@tilawa/core` use full acoustic
`VerdictTracer` snapshots. They never infer correctness from cursor advancement,
verse matches, accumulated tallies, or whole-ayah fallback recognition.

An interior omission requires zero aligned phonemes and clear immediately
neighboring words in the same ayah. A possible substitution requires normalized
phoneme distance ≥0.6, probability margin ≥0.65 and heard ratio 0.5–1.5, with the
same anchors. Clear anchors require `ok`, distance ≤0.15, margin ≥0.55 and heard
ratio 0.75–1.3. Evidence must persist across 12 advancing CTC frames (480 ms).
Uncertain or revised evidence cancels a candidate. Settling follows actual decoder
silence, never an ayah match. Lost alignment suppresses correction.

A possible harakah (short-vowel) error is a word whose consonant skeleton matches
(distance ≤0.15, heard ratio 0.75–1.3, mean word margin ≥ `vowelWordMargin`) but
where the aligned heard vowel differs from the expected one. The decoder attaches
to every vowel-final token the probabilities of the same token spelled with fatha,
damma and kasra (`CtcToken.vowels`); `WordVerdict.vowelMargin` is
p(heard vowel) − p(expected vowel) at that token's peak frame, minimised over the
word's mismatches. A flag requires `vowelMargin ≥ CorrectionThresholds.vowelMargin`
and the same clear anchors and 12-frame persistence as other kinds. The word's
final vowel is ignored when a stop follows (waqf drops it). A retry that repeats
a confident vowel error does not count as corrected.

These thresholds are conservative heuristics, not calibrated word probabilities.
A gross phoneme mismatch suggests a possible word difference; it cannot prove a
lexical substitution or distinguish every model deletion from a human omission.
The interface says “possible mistake” and supports dismissal. Boundary omissions,
consecutive missing words, unclear audio, consonant near-misses below the
substitution threshold, and unlocated passages are intentionally not flagged. No pronunciation/tajweed grade
is produced.

## Retry lifecycle

Enable with `ZipformerSession.setMode('correction')`. `correct(action)` accepts
`retry`, `stop_retry`, `dismiss`, `review_later`, `continue`, and `close`. Serialize
all session operations; the web worker queues inference and control commands.

A flag saves the actual tracker position and pauses recognition. Retry resets
fbank, CTC and injected inference state, then uses a separate engine locked to the
ayah. Only a fresh, clear, stable prefix from the ayah's beginning through the
flagged word yields success. Silence and forced cursor locks cannot do so. The
main transcript and verse history remain untouched. Continuing/closing discards
practice acoustic context and restores the saved reading position.

`dismissed`, `deferred` and `corrected` are distinct outcomes. “Review later”
resumes without claiming success or creating a reminder. Dismissed/deferred words
are suppressed for the session; reset clears suppression. Attempt IDs prevent
stale observations from satisfying a retry.

The Focus view renders original Uthmani text with diacritics and stop marks,
including an optional display-only bismillah prefix. Unexpected token-count
mismatches cause abstention rather than mis-highlighting. Reading-cursor styling
remains separate. Long ayahs wrap and scroll without truncation.

## Verification and limits

- Deterministic SDK tests cover omissions, substitutions, correct/uncertain
  input, revised/stale evidence, dismissal, retries, and position preservation.
- Injected ONNX timelines exercise fbank → CTC → alignment → flag → isolated
  successful retry → continuation for both omissions and substitutions.
- Run `node --import tsx test/correction-audio.ts` in `web/frontend` for real local
  model regressions on four recordings in both modes plus silence. The script
  uses existing local assets and does not download or upload anything.
- The legacy `test:streaming` script references deleted frontend modules; the new
  harness targets the current SDK/Zipformer pipeline instead.
- Visual checks use Paper JSX/computed styles and screenshots for English/Arabic
  at 390px/1440px. Browser verification includes real WASM inference offline.

The available recordings are correct-recitation regression fixtures, not a
labeled human-error corpus. Successful recognition and zero flags on these
recordings do not establish correction precision or recall. A larger labeled
mistake corpus is needed to calibrate detection; some short or unclear passages
cannot supply sufficient evidence. This is conservative practice assistance,
not certified recitation assessment.
