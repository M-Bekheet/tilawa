"""Phoneme targets + 251-token tokenizer for the Zipformer CTC recipe."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shared.phoneme_labels import (  # noqa: E402
    OOVError,
    PhonemeCorpus,
    PhonemeTokenizer,
    load_tokens,
    target_for_clip,
    write_tokens_txt,
)
from shared.paths import resolve_data_file  # noqa: E402


def _find_quran_json() -> Path | None:
    env = os.environ.get("TILAWA_ZIPFORMER_DATA")
    if env:
        p = Path(env)
        if p.is_dir():
            p = p / "quran.json"
        if p.is_file():
            return p
    try:
        return resolve_data_file("zipformer/quran.json")
    except FileNotFoundError:
        return None


QURAN_JSON = _find_quran_json()
skip_no_quran = pytest.mark.skipif(QURAN_JSON is None, reason="quran.json not found")


@pytest.fixture(scope="module")
def tokens() -> list[str]:
    return load_tokens()


@pytest.fixture(scope="module")
def tokenizer(tokens: list[str]) -> PhonemeTokenizer:
    return PhonemeTokenizer(tokens)


@pytest.fixture(scope="module")
def corpus() -> PhonemeCorpus:
    if QURAN_JSON is None:
        pytest.skip("quran.json not found")
    return PhonemeCorpus(QURAN_JSON)


def test_tokens_count_blank_last_unique(tokens: list[str]):
    assert len(tokens) == 251
    assert tokens[-1] == "<blank>"
    assert len(set(tokens)) == 251


def test_write_tokens_txt_icefall_ids(tokens: list[str], tmp_path: Path):
    out = tmp_path / "tokens.txt"
    write_tokens_txt(tokens, out)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 251
    assert lines[0] == f"{tokens[0]} 0"
    assert lines[-1] == "<blank> 250"


def test_longest_match_prefers_three_char_token(tokens: list[str], tokenizer: PhonemeTokenizer):
    # Vocab has "ااۜ" (id 148), plus the shorter "اا" and "ۜ".
    assert "ااۜ" in tokens
    assert "اا" in tokens
    assert "ۜ" in tokens
    ids = tokenizer.encode("ااۜ")
    assert ids == [tokens.index("ااۜ")]
    assert tokenizer.decode(ids) == "ااۜ"
    assert 250 not in ids


def test_partition_convention_alef_madd(tokenizer: PhonemeTokenizer):
    assert tokenizer.encode("ااا") == [45, 2]


@skip_no_quran
@pytest.mark.parametrize(
    "surah,ayah",
    [
        (1, 1),
        (1, 7),  # madd-heavy: contains اااااا
        (2, 255),
        (112, 1),
    ],
)
def test_tokenizer_roundtrip_ayahs(corpus: PhonemeCorpus, tokenizer: PhonemeTokenizer, surah: int, ayah: int):
    text = corpus.ayah_phonemes(surah, ayah)
    assert text
    ids = tokenizer.encode(text)
    assert 250 not in ids
    assert tokenizer.decode(ids) == text


@skip_no_quran
def test_span_phonemes_fatiha(corpus: PhonemeCorpus):
    joined = "".join(corpus.ayah_phonemes(1, a) for a in range(1, 8))
    assert corpus.span_phonemes(1, 1, 7) == joined


@skip_no_quran
def test_target_for_clip_with_basmala(corpus: PhonemeCorpus, tokenizer: PhonemeTokenizer):
    # 2:1 is not the basmala; prepend must be exact concat then encode.
    ayah = corpus.ayah_phonemes(2, 1)
    expected = tokenizer.encode(corpus.basmala() + ayah)
    got = target_for_clip(corpus, tokenizer, 2, 1, with_basmala=True)
    assert got == expected
    assert target_for_clip(corpus, tokenizer, 2, 1) == tokenizer.encode(ayah)


@skip_no_quran
def test_zero_oov_whole_corpus(corpus: PhonemeCorpus, tokenizer: PhonemeTokenizer):
    from shared.phoneme_labels import oov_report

    report = oov_report(corpus, tokenizer)
    # Plan asserts zero OOV across all ~77k words. If this fails, do not
    # paper over it — document the true count in the task report.
    assert report["oov"] == 0, report
    assert report["total_words"] == report["ok"]
    assert report["total_words"] > 77000


def test_oov_error_names_position(tokenizer: PhonemeTokenizer):
    with pytest.raises(OOVError) as ei:
        tokenizer.encode("xyz")
    err = ei.value
    assert err.position == 0
    assert "xyz" in str(err) or "x" in str(err)


@skip_no_quran
def test_native_corpus_surah1_matches_python(corpus: PhonemeCorpus):
    """Same fact as docs/specs/vectors/corpus.json (Fatiha span), via native corpus.ts."""
    tsx = ROOT / "web" / "frontend" / "node_modules" / ".bin" / "tsx"
    corpus_ts = ROOT / "web" / "frontend" / "src" / "lib" / "recitation" / "corpus.ts"
    if not tsx.is_file() or not corpus_ts.is_file():
        pytest.skip("tsx or corpus.ts missing")
    script = (
        "import { readFileSync } from 'node:fs';\n"
        f"import {{ QuranCorpus }} from {json.dumps(str(corpus_ts))};\n"
        f"const data = JSON.parse(readFileSync({json.dumps(str(QURAN_JSON))}, 'utf8'));\n"
        "const c = new QuranCorpus(data);\n"
        "const s = c.surahs[0];\n"
        "process.stdout.write(c.text.slice(c.wordStart[s.firstWord], c.wordStart[s.endWord]));\n"
    )
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "dump_s1.ts"
        p.write_text(script, encoding="utf-8")
        r = subprocess.run(
            [str(tsx), str(p)],
            capture_output=True,
            timeout=30,
            check=False,
        )
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    ts_text = r.stdout.decode("utf-8")
    assert ts_text == corpus.span_phonemes(1, 1, 7)
