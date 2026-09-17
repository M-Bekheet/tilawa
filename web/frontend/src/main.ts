import "@fontsource/amiri/400.css";
import "@fontsource/amiri/700.css";
import "./style.css";
import "./arabic-font.css";

import { initSurahDropdown, openReportDialog } from "./report-dialog";

import type {
  VerseMatchMessage,
  VerseCandidateMessage,
  FinalSequenceMessage,
  RawTranscriptMessage,
  WordProgressMessage,
  WorkerOutbound,
  QuranVerse,
  DebugMessage,
} from "./lib/types";
import { engineLabel, resolveEngine } from "./lib/engine";
import { DEFAULT_STREAMING_CONFIG } from "./lib/types";

// ---------------------------------------------------------------------------
// Types (UI-only)
// ---------------------------------------------------------------------------
interface SurahVerse {
  ayah: number;
  text_uthmani: string;
}

interface SurahData {
  surah: number;
  surah_name: string;
  surah_name_en: string;
  verses: SurahVerse[];
}

interface VerseGroup {
  surah: number;
  surahName: string;
  surahNameEn: string;
  currentAyah: number;
  verses: SurahVerse[];
  element: HTMLElement;
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
interface DiagnosticEvent {
  timestamp: number;
  type: string;
  data: Record<string, unknown>;
}

const MAX_DIAGNOSTIC_EVENTS = 50;
const MAX_DEBUG_EVENTS = 80;
const DIAGNOSTIC_COOLDOWN_MS = 30_000;
const DEBUG_VIEW_ENABLED = Boolean(import.meta.env.VITE_DEBUG_MODE);

function browserStorage(): Storage | null {
  try {
    return localStorage;
  } catch {
    return null;
  }
}

const ENGINE = resolveEngine(location.search, browserStorage());
const ENGINE_LABEL = engineLabel(ENGINE);

const state = {
  groups: [] as VerseGroup[],
  worker: null as Worker | null,
  audioCtx: null as AudioContext | null,
  stream: null as MediaStream | null,
  isActive: false,
  hasFirstMatch: false,
  modelReady: false,
  surahCache: new Map<number, SurahData>(),
  quranData: null as QuranVerse[] | null,
  sessionAudioChunks: [] as Float32Array[],
  lastModelPrediction: null as { surah: number; ayah: number; confidence: number } | null,
  diagnosticEvents: [] as DiagnosticEvent[],
  debugEvents: [] as DebugMessage[],
  lastDiagnosticSentAt: 0,
  recentVerseMatches: [] as { surah: number; ayah: number; timestamp: number }[],
  finalSequence: [] as { surah: number; ayah: number; confidence: number }[],
  streamingConfig: DEFAULT_STREAMING_CONFIG,
  audioProcessor: null as AudioWorkletNode | null,
};

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------
const $verses = document.getElementById("verses")!;
const $rawTranscript = document.getElementById("raw-transcript")!;
const $indicator = document.getElementById("listening-indicator")!;
const $permissionPrompt = document.getElementById("permission-prompt")!;
const $listeningStatus = document.getElementById("listening-status")!;
const $modelStatus = document.getElementById("model-status")!;
const $engineStatus = document.getElementById("engine-status");
if ($engineStatus) $engineStatus.textContent = ENGINE_LABEL;
const $loadingStatus = document.getElementById("loading-status")!;
const $loadingProgress = document.getElementById("loading-progress")!;
const $loadingDetail = document.getElementById("loading-detail")!;

const $readyState = document.getElementById("ready-state")!;
const $recordingState = document.getElementById("recording-state")!;
const $postRecording = document.getElementById("post-recording")!;
const $btnBeginTest = document.getElementById("btn-begin-test") as HTMLButtonElement;
const $benchIdle = document.getElementById("bench-idle");
const $btnStart = document.getElementById("btn-start")!;
const $btnStop = document.getElementById("btn-stop")!;
const $btnReport = document.getElementById("btn-report")!;
const $btnRestart = document.getElementById("btn-restart")!;
const $candidateStatus = document.getElementById("candidate-status")!;
const $debugPanel = document.getElementById("debug-panel") as HTMLDetailsElement;
const $debugSummary = document.getElementById("debug-summary")!;
const $debugContent = document.getElementById("debug-content")!;
const $debugCopy = document.getElementById("debug-copy") as HTMLButtonElement;
const $debugCopyStatus = document.getElementById("debug-copy-status")!;
const $waveform = document.getElementById("listening-waveform")!;
const $waveformBars = Array.from($waveform.querySelectorAll<HTMLElement>(".waveform-bar"));

type Language = "en" | "ar";
let language: Language = "en";
try { language = localStorage.getItem("tilawa-language") === "ar" ? "ar" : "en"; } catch { /* Storage may be unavailable. */ }
const tr = (en: string, ar: string): string => language === "ar" ? ar : en;
const $demoTitle = document.getElementById("demo-title")!;
const $recordingActions = document.getElementById("recording-actions")!;
const $retryDownload = document.getElementById("btn-retry-download")!;
const $cancelDownload = document.getElementById("btn-cancel-download")!;
let downloadPercent = 0;
let setupFailed = false;
let noMatchTimer: ReturnType<typeof setTimeout> | undefined;
let audioAttempt = 0;
let audioWorkletUrl: string | undefined;

async function prepareAudioWorklet(): Promise<void> {
  if (audioWorkletUrl) return;
  const response = await fetch("/audio-processor.js");
  if (!response.ok) throw new Error(`Audio setup failed: ${response.status}`);
  audioWorkletUrl = URL.createObjectURL(new Blob([await response.text()], { type: "text/javascript" }));
}

function refreshLabels(): void {
  $demoTitle.textContent = state.isActive
    ? (state.hasFirstMatch ? tr("Verse found", "تمّ التعرّف على الآية") : tr("Listening", "نستمع لتلاوتك"))
    : tr("Try Tilawa", "جرّب تلاوة");
  $modelStatus.textContent = state.modelReady ? tr("Offline ready", "جاهز دون إنترنت")
    : setupFailed ? tr("Download interrupted", "انقطع التنزيل")
    : modelInitStarted ? (downloadPercent >= 100 ? tr("Preparing model…", "جارٍ تجهيز النموذج…") : `${downloadPercent}%`)
    : tr("100% offline recognition", "تعرّف دون إنترنت بالكامل");
  $modelStatus.classList.toggle("ready", state.modelReady);
  if (modelInitStarted && !state.modelReady) {
    $loadingDetail.textContent = setupFailed
      ? tr("Check your connection and try again. You only need internet to get the model.", "تحقّق من الاتصال وحاول مجددًا. تحتاج إلى الإنترنت لتنزيل النموذج فقط.")
      : downloadPercent >= 100 ? tr("Preparing model…", "جارٍ تجهيز النموذج…")
      : `${tr("Downloading model", "جارٍ تنزيل النموذج")} — ${downloadPercent}%`;
  }
  for (const group of state.groups) {
    const header = group.element.querySelector<HTMLElement>(".surah-header");
    if (header) header.textContent = language === "ar" ? group.surahName : group.surahNameEn;
    const meta = group.element.querySelector<HTMLElement>(".surah-meta");
    if (meta) meta.textContent = tr(`Surah ${group.surah} · Ayah ${group.currentAyah}`, `سورة ${toArabicNum(group.surah)} · الآية ${toArabicNum(group.currentAyah)}`);
  }
  document.getElementById("recording-note")!.textContent = state.hasFirstMatch
    ? tr("Following word by word · Offline", "نتابع تلاوتك كلمةً بكلمة · دون إنترنت")
    : tr("Microphone on · Audio stays here", "الميكروفون يعمل · صوتك يبقى هنا");
}

function applyLanguage(): void {
  document.documentElement.lang = language;
  document.documentElement.dir = language === "ar" ? "rtl" : "ltr";
  document.querySelectorAll<HTMLElement>("[data-en][data-ar]").forEach(el => {
    el.textContent = el.dataset[language]!;
  });
  const toggle = document.getElementById("language-toggle")!;
  toggle.textContent = tr("العربية", "English");
  toggle.lang = language === "en" ? "ar" : "en";
  refreshLabels();
}

function failSetup(): void {
  setupFailed = true;
  $retryDownload.hidden = false;
  $cancelDownload.hidden = true;
  refreshLabels();
}

function cancelSetup(): void {
  state.worker?.terminate();
  state.worker = null;
  modelInitStarted = false;
  setupFailed = false;
  downloadPercent = 0;
  $loadingStatus.hidden = true;
  $benchIdle!.hidden = false;
  $btnBeginTest.disabled = false;
  refreshLabels();
}

const WAVEFORM_BAR_PHASES = [0.34, 0.72, 0.48, 0.95, 0.58, 1, 0.68, 0.86, 0.42, 0.76, 0.52];

function updateListeningWaveform(rms: number): void {
  const strength = Math.min(1, Math.max(0, (rms - 0.004) * 24));
  const drift = performance.now() / 180;
  $waveform.style.setProperty("--waveform-strength", strength.toFixed(3));

  for (let i = 0; i < $waveformBars.length; i++) {
    const phase = WAVEFORM_BAR_PHASES[i % WAVEFORM_BAR_PHASES.length];
    const motion = 0.55 + 0.45 * Math.sin(drift + i * 0.78);
    const level = 0.18 + strength * (phase * 0.62 + motion * 0.34);
    $waveformBars[i].style.setProperty("--bar-level", Math.min(1, level).toFixed(3));
  }
}

function resetListeningWaveform(): void {
  $waveform.style.setProperty("--waveform-strength", "0");
  for (let i = 0; i < $waveformBars.length; i++) {
    const idleLevel = 0.18 + (i % 3) * 0.035;
    $waveformBars[i].style.setProperty("--bar-level", idleLevel.toFixed(3));
  }
}

function pushStreamingConfig(): void {
  state.worker?.postMessage({ type: "set_config", config: state.streamingConfig });
  state.audioProcessor?.port.postMessage({
    type: "set_config",
    audioChunkMs: state.streamingConfig.audioChunkMs,
  });
}

// ---------------------------------------------------------------------------
// Arabic numeral converter
// ---------------------------------------------------------------------------
const arabicNumerals = ["٠", "١", "٢", "٣", "٤", "٥", "٦", "٧", "٨", "٩"];
function toArabicNum(n: number): string {
  return String(n)
    .split("")
    .map((d) => arabicNumerals[parseInt(d)])
    .join("");
}

// ---------------------------------------------------------------------------
// Surah data (loaded from quran.json, no server needed)
// ---------------------------------------------------------------------------
async function loadQuranData(): Promise<void> {
  if (state.quranData) return;
  const res = await fetch("/quran.json");
  if (!res.ok) throw new Error(`quran.json fetch failed: ${res.status}`);
  state.quranData = await res.json();
  initSurahDropdown(state.quranData!);
}

async function fetchSurah(surahNum: number): Promise<SurahData> {
  const cached = state.surahCache.get(surahNum);
  if (cached) return cached;

  await loadQuranData();
  const verses = state.quranData!.filter((v) => v.surah === surahNum);
  if (!verses.length) throw new Error(`Surah ${surahNum} not found`);

  const data: SurahData = {
    surah: surahNum,
    surah_name: verses[0].surah_name,
    surah_name_en: verses[0].surah_name_en,
    verses: verses.map((v) => ({
      ayah: v.ayah,
      text_uthmani: v.text_uthmani,
    })),
  };
  state.surahCache.set(surahNum, data);
  return data;
}

// ---------------------------------------------------------------------------
// Verse rendering
// ---------------------------------------------------------------------------
const WAQF_MARKS = new Set([
  "\u06D6", "\u06D7", "\u06D8", "\u06D9", "\u06DA", "\u06DB", "\u06DC",
]);

function isWaqfToken(token: string): boolean {
  return token.length <= 2 && [...token].every((c) => WAQF_MARKS.has(c));
}

interface WordToken {
  text: string;
  isRealWord: boolean;
}

function splitUthmaniWords(text: string): WordToken[] {
  const raw = text.split(/\s+/).filter((w) => w.length > 0);
  const result: WordToken[] = [];

  for (const token of raw) {
    if (isWaqfToken(token) && result.length > 0) {
      result[result.length - 1].text += " " + token;
    } else {
      result.push({ text: token, isRealWord: true });
    }
  }

  return result;
}

const BISMILLAH_WORD_COUNT = 4;
const BISMILLAH_BASE = "بسم الله الرحمن الرحيم";

function stripDiacritics(s: string): string {
  return s.replace(/[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]/g, "");
}

function startsWithBismillah(text: string): boolean {
  const stripped = stripDiacritics(text);
  return stripped.startsWith(BISMILLAH_BASE) || stripped.startsWith(stripDiacritics(BISMILLAH_BASE));
}

function createVerseGroupElement(group: VerseGroup): HTMLElement {
  const el = document.createElement("div");
  el.className = "verse-group";
  el.setAttribute("data-surah", String(group.surah));

  const header = document.createElement("div");
  header.className = "surah-header";
  header.textContent = language === "ar" ? group.surahName : group.surahNameEn;
  el.appendChild(header);
  const meta = document.createElement("p");
  meta.className = "surah-meta";
  el.appendChild(meta);

  const hasBismillah =
    group.surah !== 1 &&
    group.surah !== 9 &&
    startsWithBismillah(group.verses[0]?.text_uthmani ?? "");
  if (hasBismillah) {
    const words = group.verses[0].text_uthmani.split(/\s+/);
    const bsmText = words.slice(0, BISMILLAH_WORD_COUNT).join(" ");
    const bsmEl = document.createElement("div");
    bsmEl.className = "bismillah";
    bsmEl.dir = "rtl";
    bsmEl.lang = "ar";
    bsmEl.textContent = bsmText;
    el.appendChild(bsmEl);
  }

  const body = document.createElement("div");
  body.className = "verse-body";
  body.dir = "rtl";
  body.lang = "ar";

  for (const v of group.verses) {
    const verseEl = document.createElement("span");
    verseEl.className = "verse verse--upcoming";
    verseEl.setAttribute("data-ayah", String(v.ayah));

    const allWords = splitUthmaniWords(v.text_uthmani);
    const skipBsm = hasBismillah && v.ayah === 1;
    const startIdx = skipBsm ? BISMILLAH_WORD_COUNT : 0;

    const textEl = document.createElement("span");
    textEl.className = "verse-text";
    for (let i = startIdx; i < allWords.length; i++) {
      const wordEl = document.createElement("span");
      wordEl.className = "word";
      wordEl.setAttribute("data-word-idx", String(i));
      wordEl.textContent = allWords[i].text;
      textEl.appendChild(wordEl);
      if (i < allWords.length - 1) {
        textEl.appendChild(document.createTextNode(" "));
      }
    }
    verseEl.appendChild(textEl);

    const markerEl = document.createElement("span");
    markerEl.className = "verse-marker";
    markerEl.textContent = ` \u06DD${toArabicNum(v.ayah)} `;
    verseEl.appendChild(markerEl);

    body.appendChild(verseEl);
  }

  el.appendChild(body);
  return el;
}

function updateVerseHighlight(group: VerseGroup, newAyah: number): void {
  const el = group.element;
  const oldAyah = group.currentAyah;

  const verses = el.querySelectorAll<HTMLElement>(".verse");
  for (const verseEl of verses) {
    const ayah = parseInt(verseEl.getAttribute("data-ayah") || "0");
    verseEl.hidden = ayah > newAyah || ayah < newAyah - 2;
    if (ayah === newAyah) {
      verseEl.className = "verse verse--active";
    } else if (ayah <= newAyah && (ayah >= oldAyah || ayah < oldAyah)) {
      if (
        verseEl.classList.contains("verse--active") ||
        (ayah > oldAyah && ayah < newAyah) ||
        ayah <= oldAyah
      ) {
        verseEl.className = "verse verse--recited";
      }
    }
  }

  group.currentAyah = newAyah;
  refreshLabels();
  scrollToActiveVerse();
}

function scrollToActiveVerse(): void {
  const active = document.querySelector(".verse--active");
  if (active) {
    const verse = active as HTMLElement;
    const bounds = $verses.getBoundingClientRect();
    const verseBounds = verse.getBoundingClientRect();
    if (verseBounds.bottom > bounds.bottom || verseBounds.top < bounds.top + 80) {
      $verses.scrollTo({ top: Math.max(0, verse.offsetTop - 100), behavior: "instant" });
    }
  }
}

// ---------------------------------------------------------------------------
// Message handlers
// ---------------------------------------------------------------------------
async function handleVerseMatch(msg: VerseMatchMessage): Promise<void> {
  $rawTranscript.textContent = "";
  $rawTranscript.classList.remove("visible");
  $candidateStatus.hidden = true;

  state.lastModelPrediction = { surah: msg.surah, ayah: msg.ayah, confidence: msg.confidence };

  if (!state.hasFirstMatch) {
    state.hasFirstMatch = true;
    $listeningStatus.hidden = true;
    $indicator.classList.add("has-verses");
    clearTimeout(noMatchTimer);
    refreshLabels();
  }

  const lastGroup = state.groups[state.groups.length - 1];

  if (lastGroup && lastGroup.surah === msg.surah) {
    updateVerseHighlight(lastGroup, msg.ayah);
    return;
  }

  if (lastGroup) {
    lastGroup.element.classList.add("verse-group--exiting");
    const oldEl = lastGroup.element;
    setTimeout(() => oldEl.remove(), 400);
  }

  const surahData = await fetchSurah(msg.surah);

  const group: VerseGroup = {
    surah: msg.surah,
    surahName: surahData.surah_name,
    surahNameEn: surahData.surah_name_en,
    currentAyah: 0,
    verses: surahData.verses,
    element: document.createElement("div"),
  };
  group.element = createVerseGroupElement(group);
  state.groups.push(group);
  $verses.appendChild(group.element);

  updateVerseHighlight(group, msg.ayah);
}

let _matchedWordIndices = new Set<number>();
let _trackingKey = "";

function handleWordProgress(msg: WordProgressMessage): void {
  const lastGroup = state.groups[state.groups.length - 1];
  if (!lastGroup || lastGroup.surah !== msg.surah) return;

  const verseEl = lastGroup.element.querySelector<HTMLElement>(
    `.verse[data-ayah="${msg.ayah}"]`,
  );
  if (!verseEl) return;

  if (!verseEl.classList.contains("verse--active")) {
    updateVerseHighlight(lastGroup, msg.ayah);
  }

  const key = `${msg.surah}:${msg.ayah}`;
  if (key !== _trackingKey) {
    _matchedWordIndices = new Set<number>();
    _trackingKey = key;
  }

  for (const idx of msg.matched_indices) {
    _matchedWordIndices.add(idx);
  }

  let contiguousMax = -1;
  for (let i = 0; i <= msg.total_words; i++) {
    if (_matchedWordIndices.has(i)) {
      contiguousMax = i;
    } else {
      break;
    }
  }

  const wordEls = verseEl.querySelectorAll<HTMLElement>(".word");
  for (const wordEl of wordEls) {
    const idx = parseInt(wordEl.getAttribute("data-word-idx") || "-1");
    wordEl.classList.toggle("word--spoken", idx < contiguousMax);
    wordEl.classList.toggle("word--current", idx === contiguousMax);
  }
}

function handleRawTranscript(msg: RawTranscriptMessage): void {
  if (state.hasFirstMatch && !DEBUG_VIEW_ENABLED) return;
  $rawTranscript.textContent = msg.text;
  $rawTranscript.classList.add("visible");
}

async function handleVerseCandidate(msg: VerseCandidateMessage): Promise<void> {
  const best = msg.candidates[0];
  if (!best) {
    return;
  }
  if (state.hasFirstMatch && best.source !== "tracking") return;

  const surah = await fetchSurah(best.surah);
  const range =
    best.ayah_end && best.ayah_end > best.ayah
      ? `${best.ayah}-${best.ayah_end}`
      : String(best.ayah);
  const label = best.source === "tracking"
    ? tr("Pending next", "الآية التالية المحتملة")
    : msg.stable ? tr("Likely", "على الأرجح") : tr("Listening near", "نستمع قرب");

  $candidateStatus.textContent =
    `${label}: ${language === "ar" ? surah.surah_name : surah.surah_name_en} ${range} (${Math.round(best.confidence * 100)}%)`;
  $candidateStatus.classList.toggle("candidate-status--stable", msg.stable);
  $candidateStatus.classList.toggle("candidate-status--pending", best.source === "tracking");
  $candidateStatus.hidden = false;
  $listeningStatus.hidden = true;
}

async function handleFinalSequence(msg: FinalSequenceMessage): Promise<void> {
  state.finalSequence = msg.verses;
  if (!msg.verses.length) return;

  const first = msg.verses[0];
  const last = msg.verses[msg.verses.length - 1];
  const surah = await fetchSurah(first.surah);
  const range =
    first.surah === last.surah && first.ayah !== last.ayah
      ? `${first.ayah}-${last.ayah}`
      : String(first.ayah);

  $candidateStatus.textContent =
    `${tr("Recited", "التلاوة")}: ${language === "ar" ? surah.surah_name : surah.surah_name_en} ${range} (${Math.round(msg.confidence * 100)}%)`;
  $candidateStatus.classList.add("candidate-status--stable");
  $candidateStatus.hidden = false;
}

function handleDebugMessage(msg: DebugMessage): void {
  state.debugEvents.push(msg);
  if (state.debugEvents.length > MAX_DEBUG_EVENTS) {
    state.debugEvents.shift();
  }
  renderDebugPanel();
}

function syncDebugEnabled(): void {
  state.worker?.postMessage({ type: "set_debug", enabled: $debugPanel.open });
  renderDebugPanel();
}

function buildDebugBundle() {
  const totalSamples = state.sessionAudioChunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const activeGroup = state.groups[state.groups.length - 1] ?? null;
  return {
    schema: "tilawa-debug-bundle/v1",
    createdAt: new Date().toISOString(),
    pageUrl: location.href,
    userAgent: navigator.userAgent,
    engine: ENGINE,
    modelReady: state.modelReady,
    isActive: state.isActive,
    streamingConfig: state.streamingConfig,
    audio: {
      sampleRate: 16000,
      chunkCount: state.sessionAudioChunks.length,
      totalSamples,
      durationSec: Math.round((totalSamples / 16000) * 1000) / 1000,
    },
    ui: {
      hasFirstMatch: state.hasFirstMatch,
      lastModelPrediction: state.lastModelPrediction,
      activeGroup: activeGroup
        ? {
            surah: activeGroup.surah,
            surahNameEn: activeGroup.surahNameEn,
            currentAyah: activeGroup.currentAyah,
          }
        : null,
      candidateStatus: {
        text: $candidateStatus.textContent ?? "",
        hidden: $candidateStatus.hidden,
      },
      rawTranscript: {
        text: $rawTranscript.textContent ?? "",
        visible: $rawTranscript.classList.contains("visible"),
      },
      finalSequence: state.finalSequence,
      recentVerseMatches: state.recentVerseMatches,
    },
    diagnostics: state.diagnosticEvents,
    debugEvents: state.debugEvents,
  };
}

async function copyDebugBundle(): Promise<void> {
  const json = JSON.stringify(buildDebugBundle(), null, 2);
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(json);
    } else {
      const textarea = document.createElement("textarea");
      textarea.value = json;
      textarea.style.position = "fixed";
      textarea.style.left = "-9999px";
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand("copy");
      textarea.remove();
    }
    $debugCopyStatus.textContent = "Copied";
  } catch (err) {
    console.error("Failed to copy debug bundle:", err);
    $debugCopyStatus.textContent = "Copy failed";
  }

  setTimeout(() => {
    $debugCopyStatus.textContent = "";
  }, 1800);
}

function formatDebugValue(value: unknown): string {
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(3);
  if (typeof value === "string") return value;
  if (typeof value === "boolean") return value ? "true" : "false";
  if (value === null || value === undefined) return "null";
  return JSON.stringify(value);
}

function truncateDebugText(value: unknown, max = 70): string {
  const text = formatDebugValue(value).replace(/\s+/g, " ").trim();
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}

function debugChip(label: string, value: unknown, variant = ""): HTMLElement {
  const chip = document.createElement("span");
  chip.className = `debug-chip ${variant}`.trim();
  chip.textContent = `${label}: ${truncateDebugText(value)}`;
  return chip;
}

function refsFromDebugList(value: unknown, limit = 4): string {
  if (!Array.isArray(value)) return "";
  return value
    .slice(0, limit)
    .map((entry) => {
      if (!entry || typeof entry !== "object") return formatDebugValue(entry);
      const obj = entry as Record<string, unknown>;
      const score = typeof obj.score === "number" ? ` ${obj.score.toFixed(2)}` : "";
      const fusion = typeof obj.fusion === "number" ? ` ${obj.fusion.toFixed(2)}` : "";
      const ref = obj.ref ?? obj.top_ref ?? "?";
      return `${ref}${score}${fusion}`;
    })
    .join("  ");
}

function summarizeDebugEvent(event: DebugMessage): { label: string; chips: HTMLElement[] } {
  const data = event.data;
  const chips: HTMLElement[] = [];

  if (event.event === "transcribe") {
    chips.push(debugChip("sec", data.audioSec));
    chips.push(debugChip("text", data.text, "debug-chip--wide"));
    chips.push(debugChip("phon", data.rawPhonemes, "debug-chip--wide"));
    const beam = refsFromDebugList(data.beam);
    if (beam) chips.push(debugChip("beam", beam, "debug-chip--wide"));
    return { label: "asr", chips };
  }

  const trackerType = typeof data.type === "string" ? data.type : "tracker";
  if (trackerType === "discovery_cycle") {
    chips.push(debugChip("text", data.text, "debug-chip--wide"));
    chips.push(debugChip("cands", refsFromDebugList(data.candidates), "debug-chip--wide"));
    return { label: "discover", chips };
  }
  if (trackerType === "tracking_cycle") {
    chips.push(debugChip("ref", data.ref));
    chips.push(debugChip("words", `${data.word_position}/${data.total_words}`));
    chips.push(debugChip("cov", data.coverage));
    chips.push(debugChip("primary", data.word_matches));
    chips.push(debugChip("advanced", data.advanced));
    chips.push(debugChip("pending", data.pending));
    chips.push(debugChip("final", data.final_flush));
    return { label: "track", chips };
  }
  if (trackerType === "advance_decision") {
    chips.push(debugChip("from", data.from_ref));
    chips.push(debugChip("to", data.to_ref));
    chips.push(debugChip("action", data.action, data.action === "armed" ? "debug-chip--strong" : ""));
    chips.push(debugChip("why", data.reason, "debug-chip--wide"));
    chips.push(debugChip("words", `${data.word_position}/${data.total_words}`));
    chips.push(debugChip("target", data.completion_target));
    chips.push(debugChip("margin", data.margin));
    chips.push(debugChip("strict", data.strict_margin));
    return { label: "advance", chips };
  }
  if (trackerType === "commit") {
    chips.push(debugChip("ref", data.ref, "debug-chip--strong"));
    chips.push(debugChip("why", data.reason));
    chips.push(debugChip("conf", data.confidence));
    chips.push(debugChip("rank", data.selected_rank));
    return { label: "commit", chips };
  }
  if (trackerType === "pending_emission") {
    chips.push(debugChip("action", data.action));
    chips.push(debugChip("ref", data.ref));
    chips.push(debugChip("margin", data.margin));
    chips.push(debugChip("fresh", data.fresh_samples));
    return { label: "pending", chips };
  }
  if (trackerType === "rollback" || trackerType === "stale_exit" || trackerType === "flush") {
    for (const [key, value] of Object.entries(data)) {
      if (key !== "type") chips.push(debugChip(key, value));
    }
    return { label: trackerType, chips };
  }

  for (const [key, value] of Object.entries(data).slice(0, 5)) {
    if (key !== "type") chips.push(debugChip(key, value));
  }
  return { label: trackerType, chips };
}

function renderDebugPanel(): void {
  $debugSummary.textContent = `${ENGINE_LABEL} · ${state.debugEvents.length} events`;
  if (!$debugPanel.open) return;

  $debugContent.textContent = "";
  for (const event of state.debugEvents.slice().reverse()) {
    const summary = summarizeDebugEvent(event);
    const item = document.createElement("div");
    item.className = `debug-row debug-row--${summary.label}`;

    const time = document.createElement("span");
    time.className = "debug-time";
    time.textContent = new Date(event.at).toLocaleTimeString();

    const label = document.createElement("span");
    label.className = "debug-label";
    label.textContent = summary.label;

    item.append(time, label, ...summary.chips);

    $debugContent.appendChild(item);
  }
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------
function pushDiagnosticEvent(type: string, data: Record<string, unknown>): void {
  state.diagnosticEvents.push({ timestamp: Date.now(), type, data });
  if (state.diagnosticEvents.length > MAX_DIAGNOSTIC_EVENTS) {
    state.diagnosticEvents.shift();
  }
}

function recordAnomaly(msg: VerseMatchMessage): void {
  const now = Date.now();

  // Track recent verse matches for rapid switching detection
  state.recentVerseMatches.push({ surah: msg.surah, ayah: msg.ayah, timestamp: now });
  // Keep only last 10 seconds
  state.recentVerseMatches = state.recentVerseMatches.filter(
    (m) => now - m.timestamp < 10_000,
  );

  let trigger: string | null = null;

  // Surah jump: different surah than previous match
  const prev = state.lastModelPrediction;
  if (prev && prev.surah !== msg.surah) {
    trigger = "surah_jump";
  }

  // Rapid switching: 3+ different verses in 10 seconds
  if (!trigger) {
    const unique = new Set(
      state.recentVerseMatches.map((m) => `${m.surah}:${m.ayah}`),
    );
    if (unique.size >= 3) {
      trigger = "rapid_switching";
    }
  }

  if (!trigger) return;

  // Cooldown
  if (now - state.lastDiagnosticSentAt < DIAGNOSTIC_COOLDOWN_MS) return;
  state.lastDiagnosticSentAt = now;

  pushDiagnosticEvent("anomaly", { trigger });
}

// ---------------------------------------------------------------------------
// Worker message handler
// ---------------------------------------------------------------------------
function handleWorkerMessage(msg: WorkerOutbound): void {
  if (msg.type === "loading") {
    downloadPercent = Math.max(0, Math.min(100, msg.percent));
    $loadingProgress.style.setProperty("--progress", String(downloadPercent / 100));
    $loadingProgress.parentElement!.setAttribute("aria-valuenow", String(downloadPercent));
    refreshLabels();
  } else if (msg.type === "loading_status") {
    // The worker emits internal setup stages; show localized product language.
    if (/Creating|Initializing/i.test(msg.message)) downloadPercent = 100;
    refreshLabels();
  } else if (msg.type === "error") {
    console.error("Worker reported error:", msg.message);
    failSetup();
  } else if (msg.type === "ready") {
    state.modelReady = true;
    $loadingStatus.hidden = true;
    $readyState.hidden = false;
    refreshLabels();
  } else if (msg.type === "verse_match") {
    pushDiagnosticEvent("verse_match", {
      surah: msg.surah, ayah: msg.ayah, confidence: msg.confidence,
    });
    recordAnomaly(msg);
    handleVerseMatch(msg);
  } else if (msg.type === "verse_candidate") {
    pushDiagnosticEvent("verse_candidate", {
      best: msg.candidates[0] ? `${msg.candidates[0].surah}:${msg.candidates[0].ayah}` : null,
      confidence: msg.candidates[0]?.confidence ?? 0,
      stable: msg.stable,
    });
    handleVerseCandidate(msg);
  } else if (msg.type === "final_sequence") {
    pushDiagnosticEvent("final_sequence", {
      verses: msg.verses.map((v) => `${v.surah}:${v.ayah}`),
      confidence: msg.confidence,
    });
    handleFinalSequence(msg);
  } else if (msg.type === "word_progress") {
    pushDiagnosticEvent("word_progress", {
      surah: msg.surah, ayah: msg.ayah,
      word_index: msg.word_index, total_words: msg.total_words,
    });
    handleWordProgress(msg);
  } else if (msg.type === "raw_transcript") {
    pushDiagnosticEvent("raw_transcript", {
      text: msg.text, confidence: msg.confidence,
    });
    handleRawTranscript(msg);
  } else if (msg.type === "debug") {
    handleDebugMessage(msg);
  }
}

// ---------------------------------------------------------------------------
// Audio capture
// ---------------------------------------------------------------------------
async function startAudio(): Promise<boolean> {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
      },
    });
    state.stream = stream;
    $permissionPrompt.hidden = true;

    const audioCtx = new AudioContext();
    state.audioCtx = audioCtx;

    await audioCtx.audioWorklet.addModule(audioWorkletUrl!);
    const source = audioCtx.createMediaStreamSource(stream);
    const processor = new AudioWorkletNode(audioCtx, "audio-stream-processor");
    state.audioProcessor = processor;
    processor.port.postMessage({
      type: "set_config",
      audioChunkMs: state.streamingConfig.audioChunkMs,
    });

    processor.port.onmessage = (e: MessageEvent) => {
      const samples = new Float32Array(e.data as ArrayBuffer);
      // Save copy to session buffer
      state.sessionAudioChunks.push(samples.slice());
      // Send to worker for recognition
      if (state.worker) {
        state.worker.postMessage(
          { type: "audio", samples },
          [samples.buffer],
        );
      }
    };

    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 256;
    source.connect(analyser);
    source.connect(processor);

    const levelBuf = new Float32Array(analyser.fftSize);
    state.isActive = true;
    refreshLabels();
    $indicator.classList.add("active");
    resetListeningWaveform();

    const checkLevel = () => {
      if (!state.isActive) return;
      analyser.getFloatTimeDomainData(levelBuf);
      let sum = 0;
      for (let i = 0; i < levelBuf.length; i++) {
        sum += levelBuf[i] * levelBuf[i];
      }
      const rms = Math.sqrt(sum / levelBuf.length);
      updateListeningWaveform(rms);
      if (rms > 0.01) {
        $indicator.classList.add("audio-detected");
        $indicator.classList.remove("silence");
      } else {
        $indicator.classList.remove("audio-detected");
        $indicator.classList.add("silence");
      }
      requestAnimationFrame(checkLevel);
    };
    checkLevel();

    return true;
  } catch (err) {
    console.error("Failed to start audio:", err);
    stopAudio();
    $permissionPrompt.hidden = false;
    resetListeningWaveform();
    return false;
  }
}

// ---------------------------------------------------------------------------
// Stop audio capture
// ---------------------------------------------------------------------------
function stopAudio(): void {
  if (state.stream) {
    state.stream.getTracks().forEach((t) => t.stop());
    state.stream = null;
  }
  if (state.audioCtx) {
    state.audioCtx.close();
    state.audioCtx = null;
  }
  state.audioProcessor = null;
  state.isActive = false;
  clearTimeout(noMatchTimer);
  refreshLabels();
  $indicator.classList.remove("active", "audio-detected", "silence", "has-verses");
  resetListeningWaveform();
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
let modelInitStarted = false;

function initializeModel(): void {
  if (modelInitStarted) return;
  modelInitStarted = true;

  setupFailed = false;
  downloadPercent = 0;
  $retryDownload.hidden = true;
  $cancelDownload.hidden = false;
  $loadingProgress.style.setProperty("--progress", "0");
  if ($benchIdle) $benchIdle.hidden = true;
  $loadingStatus.hidden = false;
  $debugPanel.hidden = !DEBUG_VIEW_ENABLED;
  $modelStatus.textContent = `Loading ${ENGINE_LABEL}...`;
  $loadingDetail.textContent = "Starting download";
  refreshLabels();

  const worker = ENGINE === "zipformer"
    ? new Worker(new URL("./worker/zipformer-backend.ts", import.meta.url), { type: "module" })
    : new Worker(new URL("./worker/inference.ts", import.meta.url), { type: "module" });
  state.worker = worker;

  worker.onmessage = (e: MessageEvent<WorkerOutbound>) => {
    handleWorkerMessage(e.data);
  };

  worker.onerror = (e) => {
    console.error("Worker error:", e);
    failSetup();
  };

  // Load the UI's Quran data before promising that this tab can work offline.
  void Promise.all([loadQuranData(), prepareAudioWorklet()]).then(() => {
    if (state.worker === worker) worker.postMessage({ type: "init" });
  }).catch(error => {
    if (state.worker !== worker) return;
    console.error("Quran data setup failed:", error);
    failSetup();
  });
  pushStreamingConfig();
  syncDebugEnabled();
}

document.addEventListener("DOMContentLoaded", () => {
  $debugPanel.addEventListener("toggle", syncDebugEnabled);
  $debugCopy.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    copyDebugBundle();
  });

  $btnBeginTest.addEventListener("click", () => {
    $btnBeginTest.disabled = true;
    initializeModel();
  });

  syncDebugEnabled();

  // Button handlers
  $btnStart.addEventListener("click", async () => {
    if (state.isActive || $btnStart.getAttribute("aria-busy") === "true") return;
    $btnStart.setAttribute("aria-busy", "true");
    const attempt = ++audioAttempt;
    $readyState.hidden = true;
    $recordingState.hidden = false;
    $recordingActions.hidden = false;
    $postRecording.hidden = true;
    _matchedWordIndices.clear();
    _trackingKey = "";
    $listeningStatus.hidden = false;
    state.sessionAudioChunks = [];
    state.lastModelPrediction = null;
    state.hasFirstMatch = false;
    applyLanguage();
    state.groups = [];
    state.diagnosticEvents = [];
    state.debugEvents = [];
    state.recentVerseMatches = [];
    state.finalSequence = [];
    $verses.innerHTML = "";
    $rawTranscript.textContent = "";
    $rawTranscript.classList.remove("visible");
    $candidateStatus.textContent = "";
    $candidateStatus.hidden = true;
    $candidateStatus.classList.remove("candidate-status--stable", "candidate-status--pending");
    renderDebugPanel();
    // Reset tracker in worker
    state.worker?.postMessage({ type: "reset" });
    pushStreamingConfig();
    const started = await startAudio();
    $btnStart.removeAttribute("aria-busy");
    if (attempt !== audioAttempt) { stopAudio(); return; }
    if (started) noMatchTimer = setTimeout(() => {
      if (!state.isActive || state.hasFirstMatch) return;
      document.getElementById("listening-title")!.textContent = tr("We haven’t found the verse yet.", "لم نتعرّف على الآية بعد.");
      document.getElementById("listening-help")!.textContent = tr("Keep reciting a few more words. Try moving closer to the microphone.", "واصل التلاوة لبضع كلمات أخرى، وحاول الاقتراب من الميكروفون.");
    }, 15000);
    if (!started) {
      $recordingState.hidden = true;
      $recordingActions.hidden = true;
      $listeningStatus.hidden = true;
      $readyState.hidden = false;
    }
  });

  $btnStop.addEventListener("click", () => {
    audioAttempt++;
    stopAudio();
    if (ENGINE === "zipformer") {
      state.worker?.postMessage({ type: "stop" });
    }
    $recordingState.hidden = true;
    $recordingActions.hidden = true;
    $listeningStatus.hidden = true;
    $postRecording.hidden = false;
  });

  $btnRestart.addEventListener("click", () => {
    state.sessionAudioChunks = [];
    state.lastModelPrediction = null;
    state.hasFirstMatch = false;
    applyLanguage();
    state.groups = [];
    state.debugEvents = [];
    state.finalSequence = [];
    $verses.innerHTML = "";
    $rawTranscript.textContent = "";
    $rawTranscript.classList.remove("visible");
    $candidateStatus.textContent = "";
    $candidateStatus.hidden = true;
    $candidateStatus.classList.remove("candidate-status--stable", "candidate-status--pending");
    renderDebugPanel();
    $postRecording.hidden = true;
    $listeningStatus.hidden = true;
    $readyState.hidden = false;
  });

  $btnReport.addEventListener("click", async () => {
    await loadQuranData();
    openReportDialog({
      audioChunks: state.sessionAudioChunks,
      modelPrediction: state.lastModelPrediction,
      quranData: state.quranData!,
      debugBundle: buildDebugBundle(),
    });
  });
});

applyLanguage();
document.getElementById("language-toggle")!.addEventListener("click", () => {
  language = language === "en" ? "ar" : "en";
  try { localStorage.setItem("tilawa-language", language); } catch { /* Optional preference. */ }
  applyLanguage();
});
$cancelDownload.addEventListener("click", cancelSetup);
$retryDownload.addEventListener("click", () => { cancelSetup(); initializeModel(); });
document.getElementById("btn-retry-mic")!.addEventListener("click", () => $btnStart.click());
