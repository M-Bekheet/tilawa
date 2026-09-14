"""Pure helpers for icefall Zipformer-CTC export (no k2 / icefall / GPU)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from zipformer_ctc_utils import (  # noqa: E402
    ARCH_FLAGS,
    DEFAULT_TRAIN_SOURCES,
    VOCAB_SIZE,
    compute_T_hop,
    icefall_to_ref_ids,
    io_inputs_match,
    io_json_from_session,
    permute_ctc_head,
    ref_to_icefall_ids,
    write_icefall_tokens,
)

REF_IO = (
    ROOT
    / "experiments"
    / "prompter-zipformer"
    / "engine"
    / "model"
    / "zipformer-io.json"
)
TOKENS_JS = (
    ROOT
    / "experiments"
    / "prompter-zipformer"
    / "engine"
    / "model"
    / "tokens.js"
)


def test_ref_icefall_id_roundtrip_blank_250_to_0():
    ids = np.arange(VOCAB_SIZE, dtype=np.int64)
    ice = ref_to_icefall_ids(ids)
    assert ice[250] == 0
    assert ice[0] == 1
    assert ice[249] == 250
    back = icefall_to_ref_ids(ice)
    np.testing.assert_array_equal(back, ids)
    # list / scalar
    assert ref_to_icefall_ids(250) == 0
    assert icefall_to_ref_ids(0) == 250
    assert icefall_to_ref_ids(ref_to_icefall_ids([3, 250, 0])) == [3, 250, 0]


def test_permute_ctc_head_softmax_alignment():
    rng = np.random.default_rng(0)
    hidden = 32
    x = rng.standard_normal((7, hidden)).astype(np.float32)
    w = rng.standard_normal((VOCAB_SIZE, hidden)).astype(np.float32)
    b = rng.standard_normal(VOCAB_SIZE).astype(np.float32)
    w_perm, b_perm = permute_ctc_head(w, b)

    def softmax(logits: np.ndarray) -> np.ndarray:
        z = logits - logits.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    p_ice = softmax(x @ w.T + b)
    p_ref = softmax(x @ w_perm.T + b_perm)
    for ref_id in range(VOCAB_SIZE):
        ice_id = int(ref_to_icefall_ids(ref_id))
        np.testing.assert_allclose(
            p_ref[:, ref_id], p_ice[:, ice_id], rtol=1e-5, atol=1e-6
        )


def test_compute_T_hop_reference_chunk_24():
    assert compute_T_hop(24) == (61, 48)
    t, hop = compute_T_hop(16)
    assert t == 16 * 2 + 13
    assert hop == 32


def test_write_icefall_tokens_blk_first_roundtrip(tmp_path: Path):
    sys.path.insert(0, str(ROOT))
    from shared.prompter_labels import load_tokens

    tokens = load_tokens(TOKENS_JS)
    path = tmp_path / "tokens_icefall.txt"
    write_icefall_tokens(tokens, path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "<blk> 0"
    assert tokens[-1] == "<blank>"
    assert f"{tokens[0]} 1" == lines[1]
    assert f"{tokens[249]} 250" == lines[-1]
    assert len(lines) == VOCAB_SIZE
    parsed = {}
    for line in lines:
        sym, idx = line.rsplit(" ", 1)
        parsed[int(idx)] = sym
    assert parsed[0] == "<blk>"
    assert 250 in parsed
    assert "<blank>" not in parsed.values()


def test_io_json_from_session_matches_reference_schema():
    ref = json.loads(REF_IO.read_text(encoding="utf-8"))
    inputs_meta = [
        SimpleNamespace(name=inp["name"], shape=list(inp["dims"]), type="tensor(float)")
        if inp["dtype"] == "float32"
        else SimpleNamespace(
            name=inp["name"], shape=list(inp["dims"]), type="tensor(int64)"
        )
        for inp in ref["inputs"]
    ]
    outputs_meta = [
        SimpleNamespace(name="log_probs", shape=[1, 24, 251], type="tensor(float)"),
        SimpleNamespace(
            name="new_embed_states", shape=[1, 128, 3, 19], type="tensor(float)"
        ),
        SimpleNamespace(name="new_processed_lens", shape=[1], type="tensor(int64)"),
    ]
    built = io_json_from_session(
        inputs_meta,
        outputs_meta,
        model=ref["model"],
        T=ref["T"],
        hop=ref["hop"],
        feature_dim=ref["featureDim"],
        vocab_size=ref["vocabSize"],
    )
    for key in ref:
        assert key in built, f"missing key {key}"
    assert built["T"] == 61
    assert built["hop"] == 48
    assert built["featureDim"] == 80
    assert built["vocabSize"] == 251
    assert {i["name"] for i in built["inputs"]} == {i["name"] for i in ref["inputs"]}
    for a, b in zip(built["inputs"], ref["inputs"]):
        assert a["name"] == b["name"]
        assert a["dims"] == b["dims"]
        assert a["dtype"] == b["dtype"]
    assert built["outputs"][0]["dtype"] == "float32"
    assert built["outputs"][-1]["dtype"] == "int64"
    assert io_inputs_match(built, ref)
    broken = json.loads(json.dumps(built))
    broken["inputs"][0]["dims"] = [1, 60, 80]
    assert not io_inputs_match(broken, ref)


def test_arch_flags_reference_cnn_kernels():
    flags = list(ARCH_FLAGS)
    i = flags.index("--cnn-module-kernel")
    assert flags[i + 1] == "31,31,15,15,15,31"


def test_default_train_sources_exclude_qurantts():
    parts = DEFAULT_TRAIN_SOURCES.split(",")
    assert "qurantts" not in parts
    assert parts == ["everyayah", "qua", "iqra", "retasy", "tlog"]
    train_py = (ROOT / "scripts" / "train_zipformer_ctc_modal.py").read_text()
    data_py = (ROOT / "scripts" / "zipformer_asr_datamodule.py").read_text()
    assert train_py.count('sources: str = "everyayah,qua,iqra,retasy,tlog"') == 2
    assert 'default="everyayah,qua,iqra,retasy,tlog"' in data_py
    assert "--sources everyayah,qua,qurantts" not in train_py
