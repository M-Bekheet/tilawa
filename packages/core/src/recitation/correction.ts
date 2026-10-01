import type { WordVerdict } from './types.js';

export type RecitationMode = 'tracking' | 'correction';
export type CorrectionAction = 'retry' | 'stop_retry' | 'dismiss' | 'review_later' | 'continue' | 'close';
export interface RecitationPosition { surah: number; ayah: number; word: number }
export interface CorrectionIssue extends RecitationPosition {
  wordIndex: number;
  /** Word-level kinds come from {@link possibleWordIssues}. The two ayah-level
   * kinds are raised by the session when ayah N+2 is matched right after ayah N
   * and N+1 never was: `possible_skipped_ayah` when nothing of N+1 was heard,
   * `unclear_ayah` when audio was heard but the model could not follow it. */
  kind: 'possible_omission' | 'possible_substitution' | 'possible_vowel' | 'possible_skipped_ayah' | 'unclear_ayah';
  /** Words the issue covers, starting at `word`. Default 1; ayah-level kinds
   * set it to the ayah length so a retry must clear the whole ayah. */
  words?: number;
}
export const AYAH_ISSUE_KINDS: ReadonlySet<CorrectionIssue['kind']> = new Set(['possible_skipped_ayah', 'unclear_ayah']);

export interface CorrectionThresholds {
  /** Min CTC margin on a mismatched heard vowel before it counts as evidence. */
  vowelMargin: number;
  /** Min mean word margin for a vowel flag (the whole word must be confidently heard). */
  vowelWordMargin: number;
  /** Max heard ratio of a `skipped` word that counts as an omission. 0 = only
   * words with nothing aligned. A `skipped` word is always below the engine's
   * `minHeardFraction`, so 1 accepts every partial omission. */
  omissionMaxHeard?: number;
  /** GOP rule, for verdicts carrying `gop` (correction mode): a `wrong` or
   * `skipped` word whose GOP (nats/token) is at or below this. `-Infinity`
   * disables it. */
  gopFlag?: number;
  /** A neighbour anchors a GOP flag when it is clear, or fits with GOP at or above this. */
  gopAnchor?: number;
}
export const DEFAULT_CORRECTION_THRESHOLDS: Required<CorrectionThresholds> = {
  vowelMargin: 0.05, vowelWordMargin: 0.5, omissionMaxHeard: 1, gopFlag: -3, gopAnchor: -2,
};
const withDefaults = (th: Partial<CorrectionThresholds>): Required<CorrectionThresholds> =>
  ({ ...DEFAULT_CORRECTION_THRESHOLDS, ...th }) as Required<CorrectionThresholds>;
export interface CorrectionState {
  phase: 'idle' | 'error' | 'retrying' | 'corrected';
  issue: CorrectionIssue | null;
  resume: RecitationPosition | null;
  attempt: number;
  outcome: 'dismissed' | 'deferred' | 'corrected' | null;
}

// Margins are differences of token probabilities, not calibrated word confidence.
// Only gross mismatches surrounded by clear aligned words are actionable. These
// are possible word errors, never pronunciation/tajweed grades.
function clearWord(v: WordVerdict | undefined): boolean {
  return !!v && v.state === 'ok' && Number.isFinite(v.distance) && v.distance <= 0.15
    && Number.isFinite(v.margin) && v.margin >= 0.55
    && Number.isFinite(v.heardRatio) && v.heardRatio >= 0.75 && v.heardRatio <= 1.3;
}

const hasGop = (v: WordVerdict | undefined): v is WordVerdict & { gop: number } =>
  !!v && v.state !== 'pending' && typeof v.gop === 'number' && Number.isFinite(v.gop);

/** GOP rule: string alignment and acoustic fit must agree. The aligner calls
 * the word `wrong` or `skipped`, and forcing the expected word over its window
 * costs much more than the free decode. Real clean takes show GOP this low on
 * `ok` and `unsure` words too (window misallocation, elongation), so those are
 * never flagged by GOP alone. Neighbours must fit and the word must be the
 * worst fit locally, or the low score belongs to a boundary smear. */
function gopIssue(v: WordVerdict, before: WordVerdict, after: WordVerdict, th: Required<CorrectionThresholds>): CorrectionIssue['kind'] | null {
  if (!hasGop(v) || v.gop > th.gopFlag || (v.state !== 'wrong' && v.state !== 'skipped')) return null;
  const anchored = (n: WordVerdict) => clearWord(n) || (hasGop(n) && n.gop >= th.gopAnchor && n.state !== 'skipped');
  if (!anchored(before) || !anchored(after)) return null;
  if ((hasGop(before) && before.gop < v.gop) || (hasGop(after) && after.gop < v.gop)) return null;
  // Silence fits the window better than the expected word does.
  const none = v.gopNone ?? -Infinity;
  return v.state === 'skipped' || (none - v.gop >= 2 && none >= -3) ? 'possible_omission' : 'possible_substitution';
}

export function possibleWordIssues(
  verdicts: readonly WordVerdict[],
  thresholds: Partial<CorrectionThresholds> = DEFAULT_CORRECTION_THRESHOLDS,
): CorrectionIssue[] {
  const th = withDefaults(thresholds);
  const byIndex = new Map(verdicts.map(v => [v.wordIndex, v]));
  return verdicts.flatMap(v => {
    const before = byIndex.get(v.wordIndex - 1);
    const after = byIndex.get(v.wordIndex + 1);
    // Do not infer leading/trailing omissions, uncertain audio, or skipped ayahs.
    if (!before || !after || before.surah !== v.surah || after.surah !== v.surah
      || before.ayah !== v.ayah || after.ayah !== v.ayah) return [];
    const gop = gopIssue(v, before, after, th);
    if (!clearWord(before) || !clearWord(after)) {
      return gop ? [{ surah: v.surah, ayah: v.ayah, word: v.word, wordIndex: v.wordIndex, kind: gop }] : [];
    }
    // A partly heard word (the aligner lent it a few chars of its neighbours,
    // or the reciter said only its onset) is still an omission.
    const omission = v.state === 'skipped' && (v.heardRatio === 0 || v.heardRatio <= th.omissionMaxHeard);
    const substitution = v.state === 'wrong' && Number.isFinite(v.distance) && v.distance >= 0.6
      && Number.isFinite(v.margin) && v.margin >= 0.65
      && v.heardRatio >= 0.5 && v.heardRatio <= 1.5;
    // Harakah error: consonant skeleton matches (distance within `ok`), but at
    // least one aligned short vowel differs and the decoder was sure about it.
    const vowel = (v.state === 'ok' || v.state === 'unsure') && Number.isFinite(v.distance) && v.distance <= 0.15
      && (v.vowelErrors ?? 0) >= 1 && Number.isFinite(v.vowelMargin) && v.vowelMargin >= th.vowelMargin
      && Number.isFinite(v.margin) && v.margin >= th.vowelWordMargin
      && v.heardRatio >= 0.75 && v.heardRatio <= 1.3;
    const kind = omission ? 'possible_omission' as const : substitution ? 'possible_substitution' as const
      : vowel ? 'possible_vowel' as const : gop;
    return kind ? [{ surah: v.surah, ayah: v.ayah, word: v.word, wordIndex: v.wordIndex, kind }] : [];
  });
}

/** Pure state machine. Pass full, non-forced-settled acoustic snapshots only.
 * Frames must be monotonic within an observation stream. A new retry gets a new
 * attempt ID, so old audio/results cannot accidentally produce success. */
export class CorrectionController {
  mode: RecitationMode = 'tracking';
  thresholds: CorrectionThresholds = DEFAULT_CORRECTION_THRESHOLDS;
  state: CorrectionState = { phase: 'idle', issue: null, resume: null, attempt: 0, outcome: null };
  private suppressed = new Set<number>();
  private candidates = new Map<number, { kind: CorrectionIssue['kind']; frame: number }>();
  private retryFrame: number | null = null;

  reset(): void {
    this.state = { phase: 'idle', issue: null, resume: null, attempt: this.state.attempt + 1, outcome: null };
    this.suppressed.clear();
    this.clearEvidence();
  }
  clearEvidence(): void { this.candidates.clear(); this.retryFrame = null; }
  setMode(mode: RecitationMode): void { this.mode = mode; this.clearEvidence(); }

  observe(verdicts: readonly WordVerdict[], cursor: RecitationPosition, frame: number, attempt = this.state.attempt): boolean {
    if (this.mode !== 'correction' || !Number.isFinite(frame) || attempt !== this.state.attempt) return false;
    const th = withDefaults(this.thresholds);
    if (this.state.phase === 'retrying') {
      const issue = this.state.issue!;
      // Require a fresh, clear prefix from the start of this ayah through the
      // flagged word; a verse match or cursor advance alone cannot succeed.
      const through = issue.word + Math.max(1, issue.words ?? 1) - 1;
      const prefix = verdicts.filter(v => v.surah === issue.surah && v.ayah === issue.ayah && v.word <= through);
      // A retry that repeats a confident vowel error is not a correction.
      const good = Array.from({ length: through + 1 }, (_, word) =>
        prefix.find(v => v.word === word)).every(v => clearWord(v)
          && ((v!.vowelErrors ?? 0) === 0 || v!.vowelMargin < th.vowelMargin));
      if (!good) { this.retryFrame = null; return false; }
      if (this.retryFrame === null || frame < this.retryFrame) this.retryFrame = frame;
      if (frame - this.retryFrame < 12) return false;
      this.state = { ...this.state, phase: 'corrected', outcome: 'corrected' };
      return true;
    }
    if (this.state.phase !== 'idle') return false;
    const issues = possibleWordIssues(verdicts, th).filter(v => !this.suppressed.has(v.wordIndex));
    const live = new Set(issues.map(v => v.wordIndex));
    for (const key of this.candidates.keys()) if (!live.has(key)) this.candidates.delete(key);
    for (const issue of issues) {
      const old = this.candidates.get(issue.wordIndex);
      if (!old || old.kind !== issue.kind || frame < old.frame) {
        this.candidates.set(issue.wordIndex, { kind: issue.kind, frame });
      } else if (frame - old.frame >= 12) {
        this.state = { phase: 'error', issue, resume: { ...cursor }, attempt: this.state.attempt, outcome: null };
        this.clearEvidence();
        return true;
      }
    }
    return false;
  }

  /** Raise an issue the session inferred outside the word-level rules (the
   * ayah-level kinds). Same gates as a word flag: correction mode, idle, not
   * dismissed/deferred earlier in this session. */
  raise(issue: CorrectionIssue, cursor: RecitationPosition): boolean {
    if (this.mode !== 'correction' || this.state.phase !== 'idle' || this.suppressed.has(issue.wordIndex)) return false;
    this.state = { phase: 'error', issue: { ...issue }, resume: { ...cursor }, attempt: this.state.attempt, outcome: null };
    this.clearEvidence();
    return true;
  }

  act(action: CorrectionAction): boolean {
    const { phase, issue } = this.state;
    if (!issue || phase === 'idle') return false;
    if (action === 'retry' && (phase === 'error' || phase === 'corrected')) {
      this.state = { ...this.state, phase: 'retrying', attempt: this.state.attempt + 1, outcome: null };
    } else if (action === 'stop_retry' && phase === 'retrying') {
      this.state = { ...this.state, phase: 'error', attempt: this.state.attempt + 1 };
    } else if ((action === 'dismiss' && phase === 'error') || action === 'review_later'
      || action === 'close' || (action === 'continue' && phase === 'corrected')) {
      const outcome = action === 'dismiss' ? 'dismissed' : phase === 'corrected' ? 'corrected' : 'deferred';
      this.suppressed.add(issue.wordIndex);
      this.state = { ...this.state, phase: 'idle', attempt: this.state.attempt + 1, outcome };
    } else return false;
    this.clearEvidence();
    return true;
  }
}
