export type VerdictState = "ok" | "unsure" | "wrong" | "skipped" | "pending";

export interface CtcToken {
  sym: string;
  frame: number;
  margin: number;
}

export interface HeardChar {
  ch: string;
  frame: number;
  margin: number;
}

export interface WordVerdict {
  surah: number;
  ayah: number;
  word: number;
  wordIndex: number;
  state: VerdictState;
  distance: number;
  heardRatio: number;
  margin: number;
}

export interface SearchHint {
  surah: number;
  ayah: number;
}

export interface SearchHit {
  surah: number;
  ayah: number;
  word: number;
  wordIndex: number;
  refOffset: number;
  refEnd: number;
  queryStart: number;
  distance: number;
}

export interface SearchResult {
  hits: SearchHit[];
  decisive: boolean;
}

export interface CursorPos {
  surah: number;
  ayah: number;
  word: number;
  wordIndex: number;
  cost: number;
}

export type EngineState = "searching" | "tracking";

export type IdleReason = "silent" | "lost";

export interface LocatedEvent {
  type: "located";
  surah: number;
  ayah: number;
  word: number;
  replayed: number;
}

export interface RelocatedEvent {
  type: "relocated";
  from: { surah: number; ayah: number };
  to: { surah: number; ayah: number; word: number };
}

export interface CursorEvent {
  type: "cursor";
  surah: number;
  ayah: number;
  word: number;
  wordIndex: number;
}

export interface VerdictsEvent {
  type: "verdicts";
  changes: WordVerdict[];
}

export interface LostEvent {
  type: "lost";
}

export interface IdleEvent {
  type: "idle";
  reason: IdleReason;
}

export interface CompletedEvent {
  type: "completed";
  surah: number;
}

export interface LocateFailedEvent {
  type: "locateFailed";
}

export type EngineEvent =
  | LocatedEvent
  | RelocatedEvent
  | CursorEvent
  | VerdictsEvent
  | LostEvent
  | IdleEvent
  | CompletedEvent
  | LocateFailedEvent;

export interface StripResult {
  offset: number;
  basmala: boolean;
  basmalaOffset: number;
}

export interface FallbackHit {
  surah: number;
  ayah: number;
  distance: number;
  how: "basmala" | "whole-ayah";
}

export interface SurahRecord {
  n: number;
  name: string;
  nameEn: string;
  ayahCount: number;
  firstWord: number;
  endWord: number;
}
