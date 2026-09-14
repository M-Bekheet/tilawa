"""Pure helpers for icefall Zipformer2-CTC training/export (no k2, icefall, or GPU).

Blank permutation
-----------------
The reference Zipformer CTC vocab (`tokens.js` / `tokens.txt`) is 251 symbols with
``<blank>`` last (id 250). icefall CTC assumes blank id 0.

Train-time mapping is the rotation ``icefall_id = (ref_id + 1) % 251`` (so blank
250 → 0, phoneme 0 → 1, …, phoneme 249 → 250). After training, permute the CTC
output Linear's rows (weight + bias) with the inverse rotation so the exported
ONNX emits logits in reference order with blank at 250.

``permute_ctc_head`` is that row permutation: ``W_perm[ref] = W_ice[(ref+1)%251]``.
Verified by ``softmax(x @ W_perm.T + b_perm)[ref] == softmax(x @ W.T + b)[ice]``.

Streaming T / hop
-----------------
icefall ``export-onnx-streaming-ctc.py``::

    pad_length = 7 + 2 * 3   # Conv2dSubsampling (T-7)//2 plus ConvNeXt right pad
    hop = chunk_size * 2     # decode_chunk_len at 10 ms fbank rate
    T = hop + pad_length     # encoder_embed consumes T frames per chunk

``chunk_size=24`` → hop=48, T=61, matching the reference ``zipformer-io.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

VOCAB_SIZE = 251
REF_BLANK_ID = 250
ICEFALL_BLANK_ID = 0
# Conv2dSubsampling left context 7 frames + ConvNeXt right pad 3 (×2 at 100 Hz).
PAD_LENGTH = 7 + 2 * 3  # 13
# chunk_size at 50 Hz after encoder_embed; 24 → T=61, hop=48 (reference I/O).
DEFAULT_EXPORT_CHUNK_SIZE = 24
DEFAULT_LEFT_CONTEXT_FRAMES = 256

ARCH_FLAGS = [
    "--causal",
    "1",
    "--use-ctc",
    "1",
    "--use-transducer",
    "0",
    "--num-encoder-layers",
    "2,2,3,4,3,2",
    "--feedforward-dim",
    "512,768,1024,1536,1024,768",
    "--encoder-dim",
    "192,256,384,512,384,256",
    "--encoder-unmasked-dim",
    "192,192,256,256,256,192",
    "--cnn-module-kernel",
    "15,15,15,7,7,7",
    "--downsampling-factor",
    "1,2,4,8,4,2",
    "--num-heads",
    "4,4,4,8,4,4",
    "--query-head-dim",
    "32",
    "--value-head-dim",
    "12",
]

_ORT_DTYPE = {
    "tensor(float)": "float32",
    "tensor(float32)": "float32",
    "float": "float32",
    "float32": "float32",
    "tensor(int64)": "int64",
    "tensor(int32)": "int32",
    "int64": "int64",
    "int32": "int32",
}


def ref_to_icefall_ids(ids: int | Sequence[int] | np.ndarray) -> int | list[int] | np.ndarray:
    """Rotate reference ids so blank 250 maps to icefall blank 0."""
    if isinstance(ids, (int, np.integer)):
        return int((int(ids) + 1) % VOCAB_SIZE)
    arr = np.asarray(ids)
    out = (arr + 1) % VOCAB_SIZE
    if isinstance(ids, np.ndarray):
        return out.astype(ids.dtype, copy=False)
    return out.tolist()


def icefall_to_ref_ids(ids: int | Sequence[int] | np.ndarray) -> int | list[int] | np.ndarray:
    """Inverse of :func:`ref_to_icefall_ids`."""
    if isinstance(ids, (int, np.integer)):
        return int((int(ids) - 1) % VOCAB_SIZE)
    arr = np.asarray(ids)
    out = (arr - 1) % VOCAB_SIZE
    if isinstance(ids, np.ndarray):
        return out.astype(ids.dtype, copy=False)
    return out.tolist()


def permute_ctc_head(
    weight: np.ndarray, bias: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Permute CTC Linear rows from icefall order (blank=0) to reference (blank=250).

    ``weight`` is ``[vocab, hidden]``. After permutation,
    ``softmax(x @ W_perm.T + b_perm)[ref_id] == softmax(x @ W.T + b)[icefall_id]``.
    """
    w = np.asarray(weight)
    b = np.asarray(bias)
    if w.shape[0] != VOCAB_SIZE:
        raise ValueError(f"expected weight dim0={VOCAB_SIZE}, got {w.shape}")
    if b.shape[0] != VOCAB_SIZE:
        raise ValueError(f"expected bias dim0={VOCAB_SIZE}, got {b.shape}")
    perm = (np.arange(VOCAB_SIZE) + 1) % VOCAB_SIZE
    return w[perm].copy(), b[perm].copy()


def compute_T_hop(chunk_size: int, pad_length: int = PAD_LENGTH) -> tuple[int, int]:
    """Return ``(T, hop)`` for icefall streaming Zipformer export.

    ``hop = chunk_size * 2`` (fbank frames consumed per decode step).
    ``T = hop + pad_length`` with ``pad_length = 7 + 2*3 = 13``.
    ``chunk_size=24`` yields the reference ``(61, 48)``.
    """
    hop = int(chunk_size) * 2
    t = hop + int(pad_length)
    return t, hop


def write_icefall_tokens(tokens: Sequence[str], path: str | Path) -> None:
    """Write icefall-ordered ``tokens.txt``: ``<blk> 0`` then phonemes at 1..250."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["<blk> 0"]
    n = 1
    for tok in tokens:
        if tok in ("<blank>", "<blk>"):
            continue
        lines.append(f"{tok} {n}")
        n += 1
    if n != VOCAB_SIZE:
        raise ValueError(
            f"expected {VOCAB_SIZE - 1} non-blank tokens, wrote {n - 1}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def icefall_token_list(ref_tokens: Sequence[str]) -> list[str]:
    """``['<blk>', *phonemes]`` — icefall order, length 251."""
    phonemes = [t for t in ref_tokens if t not in ("<blank>", "<blk>")]
    if len(phonemes) != VOCAB_SIZE - 1:
        raise ValueError(f"expected {VOCAB_SIZE - 1} phonemes, got {len(phonemes)}")
    return ["<blk>", *phonemes]


def _as_list(shape: Any) -> list[int]:
    if shape is None:
        return []
    out = []
    for d in list(shape):
        if d is None or d == "N" or isinstance(d, str):
            out.append(1 if d in (None, "N") else d)
        else:
            out.append(int(d))
    return out


def _dtype_str(typ: Any) -> str:
    if typ is None:
        return "float32"
    s = str(typ).lower()
    if s in _ORT_DTYPE:
        return _ORT_DTYPE[s]
    if "int64" in s:
        return "int64"
    if "int32" in s:
        return "int32"
    return "float32"


def _node_meta(node: Any) -> dict:
    name = getattr(node, "name", None) or node["name"]
    shape = getattr(node, "shape", None)
    if shape is None and isinstance(node, dict):
        shape = node.get("shape") or node.get("dims")
    typ = getattr(node, "type", None)
    if typ is None and isinstance(node, dict):
        typ = node.get("type") or node.get("dtype")
    return {"name": name, "dims": _as_list(shape), "dtype": _dtype_str(typ)}


def io_json_from_session(
    inputs_meta: Iterable[Any],
    outputs_meta: Iterable[Any],
    *,
    model: str,
    T: int,
    hop: int,
    feature_dim: int = 80,
    vocab_size: int = VOCAB_SIZE,
) -> dict:
    """Build a ``zipformer-io.json`` dict from ONNX Runtime session metadata."""
    inputs = [_node_meta(n) for n in inputs_meta]
    outputs = [_node_meta(n) for n in outputs_meta]
    return {
        "model": model,
        "T": int(T),
        "hop": int(hop),
        "featureDim": int(feature_dim),
        "vocabSize": int(vocab_size),
        "inputs": inputs,
        "outputs": outputs,
    }


def diff_io_json(ours: dict, ref: dict) -> list[str]:
    """Human-readable name/dims diffs. Names-only mismatches are 'NAME' prefixed."""
    diffs: list[str] = []
    our_in = {i["name"]: i for i in ours.get("inputs", [])}
    ref_in = {i["name"]: i for i in ref.get("inputs", [])}
    extra = sorted(set(our_in) - set(ref_in))
    missing = sorted(set(ref_in) - set(our_in))
    if extra:
        diffs.append(f"NAME extra inputs: {extra}")
    if missing:
        diffs.append(f"NAME missing inputs: {missing}")
    for name in sorted(set(our_in) & set(ref_in)):
        a, b = our_in[name], ref_in[name]
        if a.get("dims") != b.get("dims"):
            diffs.append(f"dims {name}: {a.get('dims')} vs ref {b.get('dims')}")
        if a.get("dtype") != b.get("dtype"):
            diffs.append(f"dtype {name}: {a.get('dtype')} vs ref {b.get('dtype')}")
    for key in ("T", "hop", "featureDim", "vocabSize"):
        if key in ref and ours.get(key) != ref.get(key):
            diffs.append(f"{key}: {ours.get(key)} vs ref {ref.get(key)}")
    return diffs


def io_names_match(ours: dict, ref: dict) -> bool:
    our_names = [i["name"] for i in ours.get("inputs", [])]
    ref_names = [i["name"] for i in ref.get("inputs", [])]
    return our_names == ref_names


class IcefallPhonemeEncoder:
    """Duck-types SentencePieceProcessor.encode for icefall ``train.py``.

    Wraps :class:`shared.prompter_labels.PhonemeTokenizer` (reference ids) and
    rotates to icefall ids so CTC blank is 0.
    """

    def __init__(self, ref_tokens: Sequence[str]):
        from shared.prompter_labels import PhonemeTokenizer

        self.ref_tokens = list(ref_tokens)
        self.icefall_tokens = icefall_token_list(ref_tokens)
        self._tok = PhonemeTokenizer(self.ref_tokens)
        self.blank_id = ICEFALL_BLANK_ID
        self.vocab_size = VOCAB_SIZE

    def get_piece_size(self) -> int:
        return VOCAB_SIZE

    def piece_to_id(self, piece: str) -> int:
        if piece in (" ", "<blk>", "<blank>"):
            return ICEFALL_BLANK_ID
        try:
            return self.icefall_tokens.index(piece)
        except ValueError:
            return ICEFALL_BLANK_ID

    def encode(self, texts, out_type=int):
        def one(text: str):
            ids = ref_to_icefall_ids(self._tok.encode(text))
            if not isinstance(ids, list):
                ids = list(ids)
            if out_type is str or out_type == str:
                return [self.icefall_tokens[i] for i in ids]
            return ids

        if isinstance(texts, str):
            return one(texts)
        return [one(t) for t in texts]


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
