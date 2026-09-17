"""Phoneme training labels for the reference Zipformer CTC vocab.

Turns `tokens.txt` (251 symbols, `<blank>` last) and `quran.json` per-word
phoneme strings into icefall `tokens.txt` and CTC target id sequences.
Tokenisation is greedy longest-match over the non-blank inventory; the
vocab is closed under the corpus by construction.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from .paths import resolve_data_file
except ImportError:  # script / Modal partial package
    from shared.paths import resolve_data_file

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOKENS_TXT = (
    REPO_ROOT / "experiments" / "zipformer-ctc" / "tokens.txt"
)

_BLANK = "<blank>"


def resolve_quran_json() -> Path:
    """Locate prompter `quran.json`: TILAWA_PROMPTER_DATA, then per-file roots."""
    env = os.environ.get("TILAWA_PROMPTER_DATA")
    if env:
        p = Path(env)
        if p.is_dir():
            p = p / "quran.json"
        if p.is_file():
            return p
    try:
        return resolve_data_file("prompter/quran.json")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "quran.json not found; set TILAWA_PROMPTER_DATA or place it at "
            "data/prompter/quran.json"
        ) from e


def load_tokens(tokens_path: str | Path | None = None) -> list[str]:
    """Parse icefall-style `tokens.txt`: one `<sym> <id>` per line, ordered by id."""
    path = Path(tokens_path) if tokens_path is not None else DEFAULT_TOKENS_TXT
    text = path.read_text(encoding="utf-8")
    by_id: dict[int, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        sym, sep, idx_s = line.rpartition(" ")
        if not sep:
            raise ValueError(f"bad tokens.txt line: {line!r}")
        by_id[int(idx_s)] = sym
    if not by_id:
        raise ValueError(f"no tokens in {path}")
    n = max(by_id) + 1
    missing = [i for i in range(n) if i not in by_id]
    if missing:
        raise ValueError(f"token id gaps in {path}: {missing[:8]}")
    tokens = [by_id[i] for i in range(n)]
    if len(tokens) != 251:
        raise ValueError(f"expected 251 tokens, got {len(tokens)}")
    if tokens[-1] != _BLANK:
        raise ValueError(f"last token must be {_BLANK!r}, got {tokens[-1]!r}")
    return tokens


def write_tokens_txt(tokens: list[str], path: str | Path) -> None:
    """Write icefall `tokens.txt`: one `"<sym> <id>"` per line, id = index."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, sym in enumerate(tokens):
            f.write(f"{sym} {i}\n")


class OOVError(ValueError):
    """No vocab token matches at `position` in `text`."""

    def __init__(self, text: str, position: int):
        self.text = text
        self.position = position
        self.substring = text[position : position + 8]
        super().__init__(
            f"OOV at position {position}: {self.substring!r} in {text!r}"
        )


class PhonemeCorpus:
    """quran.json v2: per-ayah concatenated word phonemes, space-free."""

    def __init__(self, path: str | Path | None = None):
        path = Path(path) if path is not None else resolve_quran_json()
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        if data.get("v") != 2:
            raise ValueError(f"quran.json v{data.get('v')} unsupported")
        self.path = path
        self._ayah_words: dict[tuple[int, int], list[str]] = {}
        self._words: list[tuple[int, int, int, str]] = []
        self.n_chars = 0
        for s in data["surahs"]:
            sn = s["n"]
            for a in s["ayahs"]:
                phs = [w[1] for w in a["w"]]
                self._ayah_words[(sn, a["n"])] = phs
                for wi, ph in enumerate(phs):
                    self._words.append((sn, a["n"], wi, ph))
                    self.n_chars += len(ph)

    @property
    def n_words(self) -> int:
        return len(self._words)

    def _words_of(self, surah: int, ayah: int) -> list[str]:
        try:
            return self._ayah_words[(surah, ayah)]
        except KeyError as e:
            raise ValueError(f"no ayah {surah}:{ayah}") from e

    def word_phonemes(self, surah: int, ayah: int) -> list[str]:
        return list(self._words_of(surah, ayah))

    def ayah_phonemes(self, surah: int, ayah: int) -> str:
        return "".join(self._words_of(surah, ayah))

    def span_phonemes(self, surah: int, ayah_start: int, ayah_end: int) -> str:
        parts = [self.ayah_phonemes(surah, a) for a in range(ayah_start, ayah_end + 1)]
        return "".join(parts)

    def basmala(self) -> str:
        return self.ayah_phonemes(1, 1)

    def iter_words(self):
        yield from self._words


class PhonemeTokenizer:
    """Greedy longest-match over the non-blank token inventory."""

    def __init__(self, tokens: list[str]):
        self.tokens = tokens
        self._id_to_tok = tokens
        self._tok_to_id = {t: i for i, t in enumerate(tokens) if t != _BLANK}
        self._max_len = max(len(t) for t in self._tok_to_id) if self._tok_to_id else 0

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        i = 0
        n = len(text)
        table = self._tok_to_id
        max_len = self._max_len
        while i < n:
            matched = None
            for L in range(min(max_len, n - i), 0, -1):
                tid = table.get(text[i : i + L])
                if tid is not None:
                    matched = tid
                    i += L
                    break
            if matched is None:
                raise OOVError(text, i)
            ids.append(matched)
        return ids

    def decode(self, ids: list[int]) -> str:
        parts = []
        for i in ids:
            t = self._id_to_tok[i]
            if t == _BLANK:
                continue
            parts.append(t)
        return "".join(parts)


def oov_report(corpus: PhonemeCorpus, tokenizer: PhonemeTokenizer) -> dict:
    """Count words that tokenise vs those that raise OOVError."""
    ok = 0
    oov = 0
    examples: list[dict] = []
    for surah, ayah, wi, ph in corpus.iter_words():
        try:
            tokenizer.encode(ph)
        except OOVError as e:
            oov += 1
            if len(examples) < 64:
                examples.append(
                    {
                        "surah": surah,
                        "ayah": ayah,
                        "word": wi,
                        "phoneme": ph,
                        "position": e.position,
                        "substring": e.substring,
                    }
                )
        else:
            ok += 1
    return {
        "total_words": ok + oov,
        "ok": ok,
        "oov": oov,
        "examples": examples,
    }


def target_for_clip(
    corpus: PhonemeCorpus,
    tokenizer: PhonemeTokenizer,
    surah: int,
    ayah_start: int,
    ayah_end: int | None = None,
    with_basmala: bool = False,
) -> list[int]:
    """CTC target ids for an ayah span; optionally prepend the 1:1 basmala."""
    if ayah_end is None:
        ayah_end = ayah_start
    text = corpus.span_phonemes(surah, ayah_start, ayah_end)
    if with_basmala:
        text = corpus.basmala() + text
    return tokenizer.encode(text)


def _default_tokens_txt() -> Path:
    try:
        return resolve_quran_json().parent / "tokens.txt"
    except FileNotFoundError:
        return REPO_ROOT / "data" / "prompter" / "tokens.txt"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="tokens.txt output path (default: data/prompter/tokens.txt)",
    )
    parser.add_argument(
        "--tokens",
        "--tokens-js",
        dest="tokens",
        default=None,
        help="path to tokens.txt (default: experiments/zipformer-ctc/tokens.txt)",
    )
    args = parser.parse_args(argv)
    tokens = load_tokens(args.tokens)
    out = Path(args.path) if args.path else _default_tokens_txt()
    write_tokens_txt(tokens, out)
    corpus = PhonemeCorpus()
    tokenizer = PhonemeTokenizer(tokens)
    report = oov_report(corpus, tokenizer)
    print(f"wrote {out} ({len(tokens)} tokens, blank id 250)")
    print(f"words: {report['total_words']}")
    print(f"chars: {corpus.n_chars}")
    print(f"ok: {report['ok']}")
    print(f"oov: {report['oov']}")
    if report["examples"]:
        print("oov examples:")
        for ex in report["examples"][:20]:
            print(
                f"  {ex['surah']}:{ex['ayah']} w{ex['word']} "
                f"pos={ex['position']} {ex['substring']!r}"
            )


if __name__ == "__main__":
    main()
