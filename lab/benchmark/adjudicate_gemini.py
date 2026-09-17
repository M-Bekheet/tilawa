#!/usr/bin/env python3
"""Adjudicate Zipformer miss clips with Gemini transcription.

Two calls per miss: blind ayah ID, then A/B match against expected vs predicted
texts (unlabelled as such). Classifies LABEL_OK_MODEL_WRONG / IDENTICAL_TEXT /
LABEL_WRONG / BAD_CLIP / UNCLEAR.

Usage:
    set -a && source /Users/rock/generations/.env && set +a
    .venv/bin/python benchmark/adjudicate_gemini.py \
        --misses /tmp/misses.json \
        --out benchmark/results/gemini_adjudication_2026-09-16.json
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from datetime import date
from pathlib import Path

import requests
from Levenshtein import distance as lev_distance
from Levenshtein import ratio as lev_ratio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.normalizer import normalize_arabic  # noqa: E402
from shared.quran_db import QuranDB  # noqa: E402

MAIN_CHECKOUT = Path("/Users/rock/ai/projects/offline-tarteel")
DEFAULT_QURAN = MAIN_CHECKOUT / "lab" / "data" / "quran.json"

CORPUS_DIRS = {
    "v3": PROJECT_ROOT / "benchmark" / "test_corpus_v3",
    "qlab": PROJECT_ROOT / "benchmark" / "test_corpus_qlab",
    "v2": PROJECT_ROOT / "benchmark" / "test_corpus_v2",
}
CORPUS_FALLBACKS = {
    "qlab": MAIN_CHECKOUT / "lab" / "benchmark" / "test_corpus_qlab",
    "v3": MAIN_CHECKOUT / "lab" / "benchmark" / "test_corpus_v3",
    "v2": MAIN_CHECKOUT / "lab" / "benchmark" / "test_corpus_v2",
}

MIME = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
}

DEFAULT_MODEL = "gemini-3.1-pro-preview"
FALLBACK_MODEL = "gemini-pro-latest"
# Paid/preview Pro often has free_tier limit 0 on AI Studio keys; Flash works.
QUOTA_FALLBACK_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.8-flash",
    "gemini-3.1-flash-lite",
]
MIN_CALL_GAP_S = 4.0
API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"

IDENTICAL_RATIO = 0.97
IDENTICAL_MAX_DIST = 3

BAD_CLIP_RE = re.compile(
    r"\b(truncat|partial|cut[- ]?off|incomplete|repeat|hesitat|stutter|"
    r"noise|inaudible|unintelligib|non[- ]quran|not quran|background|"
    r"garbled|too short|clipped)\b",
    re.I,
)

DEFAULT_CONTROLS = [
    "ea_alafasy_056058",
    "ea_husary_081008",
    "ea_husary_106003",
]

BLIND_PROMPT = (
    "Transcribe this Quran recitation verbatim in Arabic with full tashkeel. "
    "Then identify the surah and ayah(s) recited (start–end). "
    "Note any repeated words, hesitations, mistakes, truncation, or non-Quranic speech. "
    "Respond as JSON "
    '{"transcript": str, "surah": int, "ayah_start": int, "ayah_end": int, '
    '"notes": str, "confidence": float}.'
)

INFORMED_PROMPT = (
    "You previously transcribed this Quran recitation as:\n"
    "{transcript}\n\n"
    "Here are two candidate ayah texts. They are NOT labelled as gold vs model; "
    "just two possibilities.\n\n"
    "Candidate A:\n{text_a}\n\n"
    "Candidate B:\n{text_b}\n\n"
    "Which candidate does the AUDIO match?\n"
    "Respond as JSON "
    '{{"verdict": "A"|"B"|"both-identical"|"neither", "reason": str}} '
    "with a one-line reason. verdict must be exactly one of those four strings."
)

BLIND_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "transcript": {"type": "STRING"},
        "surah": {"type": "INTEGER"},
        "ayah_start": {"type": "INTEGER"},
        "ayah_end": {"type": "INTEGER"},
        "notes": {"type": "STRING"},
        "confidence": {"type": "NUMBER"},
    },
    "required": ["transcript", "surah", "ayah_start", "ayah_end", "notes", "confidence"],
}

INFORMED_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "verdict": {"type": "STRING"},
        "reason": {"type": "STRING"},
    },
    "required": ["verdict", "reason"],
}


def _load_manifest(corpus: str) -> dict[str, dict]:
    d = CORPUS_DIRS[corpus]
    path = d / "manifest.json"
    if not path.exists() and corpus in CORPUS_FALLBACKS:
        path = CORPUS_FALLBACKS[corpus] / "manifest.json"
    with open(path) as f:
        samples = json.load(f)["samples"]
    return {s["id"]: s for s in samples}


def resolve_audio(corpus: str, sample: dict) -> Path:
    filename = sample["file"]
    primary = CORPUS_DIRS[corpus] / filename
    if primary.exists():
        return primary
    fb = CORPUS_FALLBACKS.get(corpus)
    if fb is not None:
        alt = fb / filename
        if alt.exists():
            return alt
    raise FileNotFoundError(f"audio missing for {sample['id']}: {primary}")


def fmt_refs(verses: list[dict] | None) -> str:
    if not verses:
        return "—"
    parts = []
    i = 0
    while i < len(verses):
        s = verses[i]["surah"]
        a0 = verses[i]["ayah"]
        a1 = a0
        j = i + 1
        while (
            j < len(verses)
            and verses[j]["surah"] == s
            and verses[j]["ayah"] == a1 + 1
        ):
            a1 = verses[j]["ayah"]
            j += 1
        parts.append(f"{s}:{a0}" if a0 == a1 else f"{s}:{a0}–{a1}")
        i = j
    return ",".join(parts)


def verse_text(db: QuranDB, verses: list[dict] | None, field: str = "text_uthmani") -> str:
    if not verses:
        return ""
    chunks = []
    for v in verses:
        row = db.get_verse(v["surah"], v["ayah"])
        if row is None:
            chunks.append("")
        else:
            chunks.append(row[field].lstrip("\ufeff"))
    return " ".join(chunks)


def text_similarity(a: str, b: str) -> tuple[float, int]:
    na = normalize_arabic(a)
    nb = normalize_arabic(b)
    if not na and not nb:
        return 1.0, 0
    return lev_ratio(na, nb), lev_distance(na, nb)


def texts_identical(sim: float, dist: int) -> bool:
    return sim >= IDENTICAL_RATIO or dist <= IDENTICAL_MAX_DIST


def parse_json_loose(text: str) -> dict:
    if not text:
        return {}
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
                return obj if isinstance(obj, dict) else {}
            except json.JSONDecodeError:
                return {"_raw": text}
        return {"_raw": text}


def extract_text(resp: dict) -> str:
    cands = resp.get("candidates") or []
    if not cands:
        return json.dumps(resp.get("error") or resp, ensure_ascii=False)[:4000]
    parts = (((cands[0] or {}).get("content") or {}).get("parts")) or []
    return "".join(p.get("text") or "" for p in parts)


_DEAD_MODELS: set[str] = set()
_WORKING_MODEL: str | None = None
_LAST_CALL_TS = 0.0
_RETRY_IN_RE = re.compile(r"retry in ([0-9.]+)s", re.I)


def gemini_generate(
    model: str,
    fallback: str,
    api_key: str,
    prompt: str,
    audio_path: Path | None,
    schema: dict,
    timeout: int = 45,
) -> tuple[str, dict, dict]:
    """Return (model_used, parsed_json, raw_response)."""
    parts: list[dict] = []
    if audio_path is not None:
        raw = audio_path.read_bytes()
        mime = MIME.get(audio_path.suffix.lower(), "audio/wav")
        parts.append(
            {
                "inline_data": {
                    "mime_type": mime,
                    "data": base64.b64encode(raw).decode("ascii"),
                }
            }
        )
    parts.append({"text": prompt})

    body = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": schema,
        },
    }

    global _WORKING_MODEL, _LAST_CALL_TS
    models: list[str] = []
    if _WORKING_MODEL and _WORKING_MODEL not in _DEAD_MODELS:
        models.append(_WORKING_MODEL)
    for extra in (model, fallback, *QUOTA_FALLBACK_MODELS):
        if extra and extra not in models and extra not in _DEAD_MODELS:
            models.append(extra)

    last_raw: dict = {}
    for mi, name in enumerate(models):
        url = f"{API_ROOT}/{name}:generateContent"
        delay = 2.0
        skip_model = False
        for attempt in range(10):
            gap = MIN_CALL_GAP_S - (time.time() - _LAST_CALL_TS)
            if gap > 0:
                time.sleep(gap)
            try:
                r = requests.post(
                    url,
                    params={"key": api_key},
                    json=body,
                    timeout=timeout,
                )
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                wait = min(delay, 20)
                print(f"  ! {name} {type(e).__name__}, retry {attempt + 1}/10 in {wait:.1f}s")
                time.sleep(wait)
                delay = min(delay * 2, 60)
                continue
            _LAST_CALL_TS = time.time()
            try:
                last_raw = r.json()
            except ValueError:
                last_raw = {"status": r.status_code, "text": r.text[:2000]}

            if r.status_code == 200:
                parsed = parse_json_loose(extract_text(last_raw))
                _WORKING_MODEL = name
                return name, parsed, last_raw

            err = last_raw.get("error") or {}
            msg = str(err.get("message") or "")
            quota_zero = r.status_code == 429 and "limit: 0" in msg
            if quota_zero or (r.status_code in (400, 404) and mi < len(models) - 1):
                _DEAD_MODELS.add(name)
                nxt = models[mi + 1] if mi < len(models) - 1 else None
                print(f"  ! {name} HTTP {r.status_code} ({msg[:80]}), falling back to {nxt}")
                skip_model = True
                break

            if r.status_code in (429, 500, 502, 503, 504):
                wait = delay
                for det in err.get("details") or []:
                    if not isinstance(det, dict):
                        continue
                    rd = det.get("retryDelay")
                    if isinstance(rd, str) and rd.endswith("s"):
                        try:
                            wait = max(wait, float(rd[:-1]))
                        except ValueError:
                            pass
                    elif isinstance(rd, dict) and "seconds" in rd:
                        wait = max(wait, float(rd["seconds"]))
                m = _RETRY_IN_RE.search(msg)
                if m:
                    wait = max(wait, float(m.group(1)))
                wait = min(wait, 70)
                print(f"  ! {name} HTTP {r.status_code}, retry {attempt + 1}/10 in {wait:.1f}s")
                time.sleep(wait)
                delay = min(delay * 2, 60)
                continue

            raise RuntimeError(
                f"Gemini {name} HTTP {r.status_code}: {json.dumps(last_raw)[:800]}"
            )
        if skip_model:
            continue
    raise RuntimeError(f"Gemini failed: {json.dumps(last_raw)[:800]}")


def find_v2_v31_miss(results_dir: Path) -> tuple[list[dict], list[dict], str]:
    """Locate retasy_v2_012 expected/predicted from a v3.1 v2 result JSON."""
    hits = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        exps = data if isinstance(data, list) else [data]
        for exp in exps:
            if not isinstance(exp, dict):
                continue
            samples = exp.get("per_sample") or []
            if len(samples) != 43:
                continue
            size = exp.get("model_size")
            sha = None
            row = None
            for s in samples:
                sha = s.get("model_sha256_prefix") or sha
                if s.get("id") == "retasy_v2_012":
                    row = s
            if row is None:
                continue
            sha_ok = isinstance(sha, str) and sha.startswith("31755836")
            size_ok = size == 72705392
            if not (sha_ok or size_ok):
                continue
            if row.get("sequence_accuracy", 1) == 1.0 and row.get("recall", 1) == 1.0:
                continue
            hits.append((path.name, row, sha, size))
    if not hits:
        return (
            [{"surah": 1, "ayah": 3}],
            [{"surah": 55, "ayah": 1, "score": 1.0}],
            "fallback (1:3→55:1 from EXPERIMENTS.md; no matching result JSON)",
        )
    name, row, sha, size = hits[0]
    return row["expected"], row["predicted"], f"{name} sha={sha} size={size}"


def load_v2_entry(results_dir: Path) -> dict:
    expected, predicted, src = find_v2_v31_miss(results_dir)
    return {
        "corpus": "v2",
        "id": "retasy_v2_012",
        "v31": [expected, predicted],
        "gentle": None,
        "_v2_source": src,
    }


def pair_for(entry: dict) -> tuple[list[dict], list[dict]]:
    pair = entry.get("v31") if entry.get("v31") is not None else entry.get("gentle")
    if not pair:
        raise ValueError(f"no expected/predicted pair for {entry.get('id')}")
    return pair[0], pair[1]


def normalize_verdict(v: str | None) -> str:
    s = (v or "").strip().lower().replace("_", "-").replace(" ", "")
    if s in ("a", "candidatea", "cand-a"):
        return "A"
    if s in ("b", "candidateb", "cand-b"):
        return "B"
    if "both" in s or s in ("identical", "same"):
        return "both-identical"
    if s in ("neither", "none", "no"):
        return "neither"
    return v or ""


def refs_match_blind(verses: list[dict], blind: dict) -> bool:
    if not verses or not blind:
        return False
    surah = blind.get("surah")
    start = blind.get("ayah_start")
    end = blind.get("ayah_end") or start
    try:
        surah = int(surah)
        start = int(start)
        end = int(end)
    except (TypeError, ValueError):
        return False
    exp_s = verses[0]["surah"]
    exp_a0 = verses[0]["ayah"]
    exp_a1 = verses[-1]["ayah"]
    if any(v["surah"] != exp_s for v in verses):
        return False
    return surah == exp_s and start == exp_a0 and end == exp_a1


def classify(row: dict) -> str:
    sim = row["text_similarity"]
    dist = row["text_distance"]
    informed = row.get("informed_parsed") or {}
    blind = row.get("blind_parsed") or {}
    verdict = normalize_verdict(informed.get("verdict"))
    notes = " ".join(
        [
            str(blind.get("notes") or ""),
            str(informed.get("reason") or ""),
        ]
    )
    clip_bad = bool(BAD_CLIP_RE.search(notes))

    if texts_identical(sim, dist) or verdict == "both-identical":
        return "IDENTICAL_TEXT"

    match_exp = verdict == "A" or refs_match_blind(row["expected"], blind)
    match_pred = verdict == "B" or refs_match_blind(row["predicted"], blind)

    if clip_bad and verdict == "neither":
        return "BAD_CLIP"
    if match_exp and not match_pred:
        return "LABEL_OK_MODEL_WRONG"
    if match_pred and not match_exp:
        return "LABEL_WRONG"
    if clip_bad:
        return "BAD_CLIP"
    return "UNCLEAR"


def print_table(rows: list[dict]) -> None:
    headers = [
        "id",
        "corpus",
        "expected",
        "predicted",
        "sim",
        "blind",
        "informed",
        "class",
        "notes",
    ]
    table = [headers]
    for r in rows:
        blind = r.get("blind_parsed") or {}
        b_s = blind.get("surah")
        b_a0 = blind.get("ayah_start")
        b_a1 = blind.get("ayah_end") or b_a0
        if b_s is None:
            blind_s = "—"
        elif b_a0 == b_a1:
            blind_s = f"{b_s}:{b_a0}"
        else:
            blind_s = f"{b_s}:{b_a0}–{b_a1}"
        informed = r.get("informed_parsed") or {}
        notes = (informed.get("reason") or blind.get("notes") or "")[:60]
        notes = notes.replace("\n", " ")
        table.append(
            [
                r["id"],
                r["corpus"],
                fmt_refs(r.get("expected")),
                fmt_refs(r.get("predicted")),
                f"{r.get('text_similarity', 0):.3f}",
                blind_s,
                normalize_verdict(informed.get("verdict")) or "—",
                r.get("class") or "—",
                notes,
            ]
        )
    widths = [max(len(str(row[i])) for row in table) for i in range(len(headers))]
    for i, row in enumerate(table):
        line = " | ".join(str(c).ljust(widths[j]) for j, c in enumerate(row))
        print(line)
        if i == 0:
            print("-+-".join("-" * w for w in widths))


def _strip_thought(obj):
    if isinstance(obj, dict):
        return {k: _strip_thought(v) for k, v in obj.items() if k != "thoughtSignature"}
    if isinstance(obj, list):
        return [_strip_thought(x) for x in obj]
    return obj


def save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_strip_thought(payload), ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def build_miss_row(entry: dict, db: QuranDB, manifests: dict[str, dict]) -> dict:
    corpus = entry["corpus"]
    sid = entry["id"]
    sample = manifests[corpus][sid]
    audio = resolve_audio(corpus, sample)
    expected, predicted = pair_for(entry)
    t_exp = verse_text(db, expected)
    t_pred = verse_text(db, predicted)
    sim, dist = text_similarity(t_exp, t_pred)
    return {
        "id": sid,
        "corpus": corpus,
        "audio": str(audio),
        "expected": expected,
        "predicted": predicted,
        "expected_text": t_exp,
        "predicted_text": t_pred,
        "text_similarity": sim,
        "text_distance": dist,
        "v31_miss": entry.get("v31") is not None,
        "gentle_miss": entry.get("gentle") is not None,
        "v2_source": entry.get("_v2_source"),
        "kind": "miss",
    }


def build_control_row(sid: str, db: QuranDB, manifests: dict[str, dict]) -> dict:
    sample = manifests["v3"][sid]
    audio = resolve_audio("v3", sample)
    expected = sample.get("expected_verses") or [
        {"surah": sample["surah"], "ayah": sample["ayah"]}
    ]
    t_exp = verse_text(db, expected)
    return {
        "id": sid,
        "corpus": "v3",
        "audio": str(audio),
        "expected": expected,
        "predicted": expected,
        "expected_text": t_exp,
        "predicted_text": t_exp,
        "text_similarity": 1.0,
        "text_distance": 0,
        "v31_miss": False,
        "gentle_miss": False,
        "kind": "control",
    }


def adjudicate_row(
    row: dict,
    model: str,
    fallback: str,
    api_key: str,
) -> None:
    audio = Path(row["audio"])
    kind = row.get("kind", "miss")

    if "blind_parsed" not in row or not row.get("blind_parsed"):
        print(f"  blind {row['id']} …", flush=True)
        used, parsed, raw = gemini_generate(
            model, fallback, api_key, BLIND_PROMPT, audio, BLIND_SCHEMA
        )
        row["blind_model"] = used
        row["blind_parsed"] = parsed
        row["blind_raw"] = raw
        print(
            f"    → {parsed.get('surah')}:{parsed.get('ayah_start')}–{parsed.get('ayah_end')} "
            f"conf={parsed.get('confidence')}",
            flush=True,
        )

    if kind != "control" and (
        "informed_parsed" not in row or not row.get("informed_parsed")
    ):
        transcript = (row.get("blind_parsed") or {}).get("transcript") or ""
        prompt = INFORMED_PROMPT.format(
            transcript=transcript,
            text_a=row["expected_text"],
            text_b=row["predicted_text"],
        )
        print(f"  informed {row['id']} …", flush=True)
        used, parsed, raw = gemini_generate(
            model, fallback, api_key, prompt, audio, INFORMED_SCHEMA
        )
        row["informed_model"] = used
        row["informed_parsed"] = parsed
        row["informed_raw"] = raw
        print(
            f"    → {normalize_verdict(parsed.get('verdict'))}: {parsed.get('reason')}",
            flush=True,
        )

    if kind == "control":
        row["class"] = (
            "CONTROL_HIT"
            if refs_match_blind(row["expected"], row.get("blind_parsed") or {})
            else "CONTROL_MISS"
        )
    else:
        row["class"] = classify(row)


def control_hit_rate(rows: list[dict]) -> tuple[int, int]:
    ctrls = [r for r in rows if r.get("kind") == "control"]
    hits = sum(1 for r in ctrls if r.get("class") == "CONTROL_HIT")
    return hits, len(ctrls)


def class_counts(rows: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        if r.get("kind") == "control":
            continue
        out[r.get("class") or "UNCLEAR"] = out.get(r.get("class") or "UNCLEAR", 0) + 1
    return out


def markdown_table(rows: list[dict]) -> str:
    lines = [
        "| id | corpus | expected | predicted | text-similarity | Gemini blind surah:ayah | Gemini informed verdict | class | notes |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r.get("kind") == "control":
            continue
        blind = r.get("blind_parsed") or {}
        b_s, b_a0, b_a1 = blind.get("surah"), blind.get("ayah_start"), blind.get("ayah_end") or blind.get("ayah_start")
        if b_s is None:
            blind_s = "—"
        elif b_a0 == b_a1:
            blind_s = f"{b_s}:{b_a0}"
        else:
            blind_s = f"{b_s}:{b_a0}–{b_a1}"
        informed = r.get("informed_parsed") or {}
        notes = (informed.get("reason") or blind.get("notes") or "").replace("|", "/").replace("\n", " ")
        lines.append(
            "| {id} | {corpus} | {exp} | {pred} | {sim:.3f} | {blind} | {verdict} | {cls} | {notes} |".format(
                id=r["id"],
                corpus=r["corpus"],
                exp=fmt_refs(r.get("expected")),
                pred=fmt_refs(r.get("predicted")),
                sim=r.get("text_similarity") or 0,
                blind=blind_s,
                verdict=normalize_verdict(informed.get("verdict")) or "—",
                cls=r.get("class") or "—",
                notes=notes,
            )
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--misses", default="/tmp/misses.json")
    parser.add_argument(
        "--out",
        default=str(
            PROJECT_ROOT
            / "benchmark"
            / "results"
            / f"gemini_adjudication_{date.today().isoformat()}.json"
        ),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fallback-model", default=FALLBACK_MODEL)
    parser.add_argument("--quran", default=str(DEFAULT_QURAN))
    parser.add_argument("--controls", nargs="*", default=DEFAULT_CONTROLS)
    parser.add_argument("--no-controls", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY") or ""
    if not api_key:
        print("GEMINI_API_KEY is not set", file=sys.stderr)
        return 2

    db = QuranDB(Path(args.quran))
    manifests = {c: _load_manifest(c) for c in CORPUS_DIRS}

    misses = json.loads(Path(args.misses).read_text())
    if not any(m.get("id") == "retasy_v2_012" for m in misses):
        misses.append(load_v2_entry(PROJECT_ROOT / "benchmark" / "results"))

    out_path = Path(args.out)
    payload: dict = {"model": args.model, "samples": [], "controls": []}
    done: dict[str, dict] = {}
    if out_path.exists():
        try:
            payload = json.loads(out_path.read_text())
            for r in payload.get("samples", []) + payload.get("controls", []):
                done[f"{r.get('kind','miss')}:{r['id']}"] = r
            print(f"resuming from {out_path} ({len(done)} cached)")
        except json.JSONDecodeError:
            payload = {"model": args.model, "samples": [], "controls": []}

    samples: list[dict] = []
    for entry in misses:
        key = f"miss:{entry['id']}"
        if key in done and done[key].get("blind_parsed") and (
            done[key].get("kind") == "control" or done[key].get("informed_parsed")
        ):
            row = done[key]
            row["class"] = classify(row)
            samples.append(row)
            print(f"cached {entry['id']}")
            continue
        row = done.get(key) or build_miss_row(entry, db, manifests)
        adjudicate_row(row, args.model, args.fallback_model, api_key)
        samples.append(row)
        payload["model"] = row.get("blind_model") or args.model
        payload["samples"] = samples
        payload["controls"] = payload.get("controls") or []
        save(out_path, payload)

    controls: list[dict] = []
    if not args.no_controls:
        for sid in args.controls:
            key = f"control:{sid}"
            if key in done and done[key].get("blind_parsed"):
                row = done[key]
                row["class"] = (
                    "CONTROL_HIT"
                    if refs_match_blind(row["expected"], row.get("blind_parsed") or {})
                    else "CONTROL_MISS"
                )
                controls.append(row)
                print(f"cached control {sid}")
                continue
            row = done.get(key) or build_control_row(sid, db, manifests)
            adjudicate_row(row, args.model, args.fallback_model, api_key)
            controls.append(row)
            payload["samples"] = samples
            payload["controls"] = controls
            save(out_path, payload)

    for r in samples:
        r["class"] = classify(r)

    hits, nctrl = control_hit_rate(controls)
    counts = class_counts(samples)
    payload = {
        "model": payload.get("model") or args.model,
        "fallback_model": args.fallback_model,
        "date": date.today().isoformat(),
        "n_misses": len(samples),
        "class_counts": counts,
        "control_hit_rate": {"hits": hits, "n": nctrl},
        "markdown_table": markdown_table(samples),
        "samples": samples,
        "controls": controls,
    }
    save(out_path, payload)

    print()
    print_table(samples + controls)
    print()
    print("class counts:", json.dumps(counts, sort_keys=True))
    print(f"control hit rate: {hits}/{nctrl}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
