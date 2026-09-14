"""icefall streaming Zipformer2-CTC on Modal + streaming ONNX export.

App ``zipformer-ctc-train``. Trains with icefall's ``zipformer/train.py`` as a
subprocess (copy recipe into ``/vol/work/zipformer``, overwrite
``asr_datamodule.py``, patch the two ``sp`` uses: SentencePiece load /
``vocab_size`` + ``encode``). Own-script alternative was rejected so we stay
on icefall's optimiser, Eden schedule, DDP, checkpoint averaging, and
streaming export without reimplementing them.

Blank handling: train in icefall order (blank id 0 = ``(ref_id + 1) % 251``);
permute the CTC Linear rows at export so ONNX logits match the reference
vocab with ``<blank>`` at 250. See ``scripts/zipformer_ctc_utils.py``.

Architecture (verbatim from the reference ONNX ``metadata_props``):
``--cnn-module-kernel 31,31,15,15,15,31`` (cache last-dim = kernel//2).
Export runs on a CPU function (``cpu=8``, no GPU).

Pinned icefall: ``3f848bb6d0acc970c9b294a30ca0a04a7c9c78d1`` (master HEAD at
implementation). k2 wheel:
``k2==1.24.4.dev20250715+cuda12.4.torch2.4.1`` from ``https://k2-fsa.github.io/k2/cuda.html``
(cp311 manylinux; Hugging Face ``csukuangfj/k2`` ubuntu-cuda artifact, HTTP 302 verified).
Image is ``nvidia/cuda:12.4.1-devel-ubuntu22.04`` + Modal ``add_python=3.11`` + pip
``torch==2.4.1+cu124`` (the pytorch/pytorch conda image fights Modal's Python).

Usage::

    ZIPFORMER_GPU=H100 modal run --detach scripts/train_zipformer_ctc_modal.py \\
        --run-name smoke --smoke --synthetic

    modal run scripts/train_zipformer_ctc_modal.py \\
        --run-name v1 --export-only --epoch 40 --avg 10

Do not download checkpoints/ONNX into git. After export::

    modal volume get zipformer-ctc-training /exports/<run_name> /tmp/zipformer-<run_name>
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

WORKTREE = Path(__file__).resolve().parent.parent
MAIN_CHECKOUT = Path("/Users/rock/ai/projects/offline-tarteel")

ICEFALL_SHA = "3f848bb6d0acc970c9b294a30ca0a04a7c9c78d1"
K2_VERSION = "1.24.4.dev20250715+cuda12.4.torch2.4.1"
K2_WHEEL = (
    "https://huggingface.co/csukuangfj/k2/resolve/main/ubuntu-cuda/"
    "k2-1.24.4.dev20250715+cuda12.4.torch2.4.1-cp311-cp311-manylinux2014_x86_64"
    ".manylinux_2_17_x86_64.whl"
)
K2_INDEX = "https://k2-fsa.github.io/k2/cuda.html"

_GPU_SPEC = os.environ.get("ZIPFORMER_GPU", "H100")


def _world_size(spec: str) -> int:
    if ":" in spec:
        tail = spec.rsplit(":", 1)[-1]
        if tail.isdigit():
            return int(tail)
    return 1


WORLD_SIZE = _world_size(_GPU_SPEC)


def _client_file(rel: str) -> Path | None:
    for root in (WORKTREE, MAIN_CHECKOUT):
        p = root / rel
        if p.is_file():
            return p
    return None


def _build_image() -> modal.Image:
    img = (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.1-devel-ubuntu22.04",
            add_python="3.11",
        )
        .apt_install("git", "libsndfile1", "ffmpeg", "sox")
        .pip_install(
            "numpy<2",
            "torch==2.4.1+cu124",
            "torchaudio==2.4.1+cu124",
            extra_index_url="https://download.pytorch.org/whl/cu124",
        )
        .pip_install(
            f"k2=={K2_VERSION}",
            find_links=K2_INDEX,
            extra_options="--no-deps",
        )
        .run_commands(
            "python -c 'import torch; print(torch.__version__, torch.version.cuda); assert \"cu124\" in torch.__version__ or str(torch.version.cuda).startswith(\"12.4\")'",
            "python -c 'import importlib.metadata as m; print(\"k2\", m.version(\"k2\"))'",
        )
        .pip_install(
            "lhotse",
            "urllib3",
            "requests",
            "kaldi-native-fbank",
            "sentencepiece>=0.1.96",
            "tensorboard",
            "onnx>=1.15.0",
            "onnxruntime>=1.16.3",
            "soundfile",
            "lilcom",
            "kaldialign",
            "typeguard",
            "dill",
            "packaging",
            "pypinyin==0.50.0",
            "num2words",
        )
        .pip_install("numpy<2")
        .run_commands(
            "mkdir -p /app/shared && touch /app/shared/__init__.py",
            "rm -rf /opt/icefall && git clone https://github.com/k2-fsa/icefall /opt/icefall "
            f"&& git -C /opt/icefall checkout {ICEFALL_SHA} "
            f"&& git -C /opt/icefall rev-parse HEAD > /opt/icefall.sha "
            f"&& test \"$(cat /opt/icefall.sha)\" = '{ICEFALL_SHA}'",
        )
        .env(
            {
                "PYTHONPATH": "/opt/icefall:/app",
                "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
            }
        )
    )
    # add_local_file only on the client (these paths do not exist in the container).
    mounts = [
        ("shared/prompter_labels.py", "/app/shared/prompter_labels.py"),
        ("shared/fbank.py", "/app/shared/fbank.py"),
        ("scripts/zipformer_ctc_utils.py", "/app/zipformer_ctc_utils.py"),
        ("scripts/zipformer_asr_datamodule.py", "/app/zipformer_asr_datamodule.py"),
        (
            "experiments/prompter-zipformer/engine/model/tokens.js",
            "/app/tokens.js",
        ),
        (
            "experiments/prompter-zipformer/engine/model/zipformer-io.json",
            "/app/zipformer-io.json",
        ),
        ("data/prompter/quran.json", "/app/data/prompter/quran.json"),
    ]
    found = [(rel, remote, _client_file(rel)) for rel, remote in mounts]
    if any(src is not None for _, _, src in found):
        missing = [rel for rel, _, src in found if src is None]
        if missing:
            raise FileNotFoundError(f"missing local files for image: {missing}")
        for _, remote, src in found:
            img = img.add_local_file(str(src), remote_path=remote)
    return img


image = _build_image()
app = modal.App("zipformer-ctc-train")
vol = modal.Volume.from_name("zipformer-ctc-training", create_if_missing=True)

_TRAIN_SP_OLD = '''    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    # <blk> is defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.sos_id = params.eos_id = sp.piece_to_id("<sos/eos>")
    params.vocab_size = sp.get_piece_size()
'''

_TRAIN_SP_NEW = '''    import sys as _sys
    _sys.path.insert(0, "/app")
    from zipformer_ctc_utils import IcefallPhonemeEncoder
    from shared.prompter_labels import load_tokens as _load_tokens
    sp = IcefallPhonemeEncoder(_load_tokens("/app/tokens.js"))
    params.blank_id = 0
    params.sos_id = params.eos_id = 0
    params.vocab_size = 251
'''

_EXPORT_PERM_OLD = '''    convert_scaled_to_non_scaled(model, inplace=True)

    model = OnnxModel(
'''
_EXPORT_PERM_NEW = '''    convert_scaled_to_non_scaled(model, inplace=True)

    import sys as _sys
    _sys.path.insert(0, "/app")
    from zipformer_ctc_utils import permute_ctc_head as _permute_ctc_head
    _linear = model.ctc_output[1]
    _w, _b = _permute_ctc_head(
        _linear.weight.detach().cpu().numpy(),
        _linear.bias.detach().cpu().numpy(),
    )
    import torch as _torch
    with _torch.no_grad():
        _linear.weight.copy_(_torch.from_numpy(_w).to(_linear.weight.dtype))
        _linear.bias.copy_(_torch.from_numpy(_b).to(_linear.bias.dtype))
    logging.info("permuted CTC head to reference blank=250 order")

    model = OnnxModel(
'''

_TRAIN_VALID_STASH_OLD = '''            logging.info(f"Epoch {params.cur_epoch}, validation: {valid_info}")
'''
_TRAIN_VALID_STASH_NEW = '''            logging.info(f"Epoch {params.cur_epoch}, validation: {valid_info}")
            params.last_valid_loss = valid_info["loss"] / valid_info["frames"]
'''

_TRAIN_EPOCH_T0_OLD = '''    for epoch in range(params.start_epoch, params.num_epochs + 1):
        scheduler.step_epoch(epoch - 1)
'''
_TRAIN_EPOCH_T0_NEW = '''    import time as _epoch_time
    for epoch in range(params.start_epoch, params.num_epochs + 1):
        _epoch_t0 = _epoch_time.time()
        scheduler.step_epoch(epoch - 1)
'''

_TRAIN_METRICS_OLD = '''        save_checkpoint(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=train_dl.sampler,
            scaler=scaler,
            rank=rank,
        )

    logging.info("Done!")
'''
_TRAIN_METRICS_NEW = '''        save_checkpoint(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=train_dl.sampler,
            scaler=scaler,
            rank=rank,
        )

        if rank == 0:
            import json as _json
            import logging as _logging
            _train = float(getattr(params, "train_loss", float("nan")))
            _valid = float(getattr(params, "last_valid_loss", float("nan")))
            _lr = float(max(scheduler.get_last_lr()))
            _elapsed = float(_epoch_time.time() - _epoch_t0)
            _m = Path(params.exp_dir) / "metrics.jsonl"
            with _m.open("a", encoding="utf-8") as _f:
                _f.write(
                    _json.dumps(
                        {
                            "epoch": int(params.cur_epoch),
                            "train_loss": _train,
                            "valid_loss": _valid,
                            "lr": _lr,
                            "elapsed_s": _elapsed,
                        }
                    )
                    + "\\n"
                )
            _logging.info("wrote epoch metrics %s", _m)
            try:
                import modal as _modal
                _modal.Volume.from_name("zipformer-ctc-training").commit()
            except Exception as _e:
                _logging.warning("vol.commit after epoch failed: %s", _e)

    logging.info("Done!")
'''


def _patch_file(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new.strip() in text and old not in text:
        print(f"patch already applied: {label}")
        return
    if old not in text:
        raise RuntimeError(
            f"patch {label!r} did not match {path}; "
            f"icefall recipe may have changed (SHA {ICEFALL_SHA})"
        )
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    written = path.read_text(encoding="utf-8")
    if old not in new:
        assert old not in written, f"patch {label!r} left original text in {path}"
    assert new.strip() in written
    print(f"patched {label} in {path}")


def _prepare_recipe(work: Path, smoke: bool) -> Path:
    import shutil

    src = Path("/opt/icefall/egs/librispeech/ASR/zipformer")
    dst = work / "zipformer"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    shutil.copy("/app/zipformer_asr_datamodule.py", dst / "asr_datamodule.py")
    _patch_file(dst / "train.py", _TRAIN_SP_OLD, _TRAIN_SP_NEW, "train.py sp/vocab")
    _patch_file(
        dst / "train.py",
        _TRAIN_VALID_STASH_OLD,
        _TRAIN_VALID_STASH_NEW,
        "train.py last_valid_loss",
    )
    _patch_file(
        dst / "train.py",
        _TRAIN_EPOCH_T0_OLD,
        _TRAIN_EPOCH_T0_NEW,
        "train.py epoch wall clock",
    )
    _patch_file(
        dst / "train.py",
        _TRAIN_METRICS_OLD,
        _TRAIN_METRICS_NEW,
        "train.py per-epoch metrics+commit",
    )
    train_txt = (dst / "train.py").read_text(encoding="utf-8")
    if smoke:
        if '"log_interval": 50,' not in train_txt:
            raise RuntimeError(
                "smoke log_interval patch: original '\"log_interval\": 50,' not found"
            )
        train_txt = train_txt.replace('"log_interval": 50,', '"log_interval": 1,')
        train_txt = train_txt.replace(
            '"valid_interval": 3000,  # For the 100h subset, use 800',
            '"valid_interval": 5,',
        )
        (dst / "train.py").write_text(train_txt, encoding="utf-8")
        patched = (dst / "train.py").read_text(encoding="utf-8")
        assert '"log_interval": 1,' in patched, "smoke log_interval patch did not apply"
        assert '"log_interval": 50,' not in patched
        assert '"valid_interval": 5,' in patched
        print("asserted smoke log_interval=1 valid_interval=5")
    # export-onnx-streaming-ctc.py already uses token_table["<blk>"] at this SHA.
    _patch_file(
        dst / "export-onnx-streaming-ctc.py",
        _EXPORT_PERM_OLD,
        _EXPORT_PERM_NEW,
        "export ctc permute",
    )
    return dst


def _build_synthetic_cuts(n: int = 20) -> Path:
    import numpy as np
    import soundfile as sf
    from lhotse import (
        CutSet,
        Fbank,
        FbankConfig,
        MonoCut,
        Recording,
        SupervisionSegment,
    )
    from lhotse.features.io import LilcomChunkyWriter

    sys.path.insert(0, "/app")
    from shared.fbank import LHOTSE_FBANK_CONFIG
    from shared.prompter_labels import PhonemeCorpus

    audio_dir = Path("/vol/synthetic/audio")
    feat_dir = Path("/vol/fbank/synthetic")
    man_dir = Path("/vol/manifests")
    audio_dir.mkdir(parents=True, exist_ok=True)
    feat_dir.mkdir(parents=True, exist_ok=True)
    man_dir.mkdir(parents=True, exist_ok=True)

    corpus = PhonemeCorpus("/app/data/prompter/quran.json")
    rng = np.random.default_rng(0)
    cuts = []
    ayahs = [(1, a) for a in range(1, 8)]
    for i in range(n):
        dur = float(rng.uniform(2.0, 6.0))
        n_samples = int(round(dur * 16000))
        wav = (0.02 * rng.standard_normal(n_samples)).astype(np.float32)
        wav_path = audio_dir / f"syn_{i:04d}.wav"
        sf.write(str(wav_path), wav, 16000, subtype="FLOAT")
        rec = Recording.from_file(wav_path)
        surah, ayah = ayahs[i % len(ayahs)]
        text = corpus.ayah_phonemes(surah, ayah)
        sup = SupervisionSegment(
            id=rec.id,
            recording_id=rec.id,
            start=0.0,
            duration=rec.duration,
            channel=0,
            text=text,
            language="quran-phonemes",
            custom={
                "surah": surah,
                "ayah": ayah,
                "source": "synthetic",
            },
        )
        cuts.append(
            MonoCut(
                id=rec.id,
                start=0.0,
                duration=rec.duration,
                channel=0,
                recording=rec,
                supervisions=[sup],
            )
        )
    cs = CutSet.from_cuts(cuts)
    cfg = dict(LHOTSE_FBANK_CONFIG)
    try:
        extractor = Fbank(FbankConfig(**cfg))
    except TypeError:
        cfg.pop("torchaudio_compatible_mel_scale", None)
        extractor = Fbank(FbankConfig(**cfg))
    cs = cs.compute_and_store_features(
        extractor=extractor,
        storage_path=str(feat_dir),
        storage_type=LilcomChunkyWriter,
        num_jobs=1,
    )
    out = man_dir / "synthetic_cuts_fbank.jsonl.gz"
    cs.to_file(out)
    print(f"wrote {n} synthetic cuts -> {out}")
    return out


def _arch_train_flags() -> list[str]:
    from zipformer_ctc_utils import ARCH_FLAGS

    return [
        *ARCH_FLAGS,
        "--chunk-size",
        "16,32,64,-1",
        "--left-context-frames",
        "64,128,256,-1",
    ]


def _write_tokens() -> Path:
    sys.path.insert(0, "/app")
    from shared.prompter_labels import load_tokens
    from zipformer_ctc_utils import write_icefall_tokens

    path = Path("/vol/tokens_icefall.txt")
    write_icefall_tokens(load_tokens("/app/tokens.js"), path)
    return path


def _parse_metrics(exp_dir: Path) -> list[dict]:
    """Read per-epoch ``metrics.jsonl`` written by the train.py patch. Do not overwrite."""
    import json

    path = exp_dir / "metrics.jsonl"
    rows: list[dict] = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append({"raw": line[-400:]})
        print(f"read {len(rows)} epoch metric rows <- {path}")
        return rows
    print(f"no {path}; train.py patch may not have run")
    return rows


def _latest_epoch(exp_dir: Path) -> int:
    epochs = []
    for p in exp_dir.glob("epoch-*.pt"):
        try:
            epochs.append(int(p.stem.split("-")[1]))
        except (IndexError, ValueError):
            continue
    if not epochs:
        raise FileNotFoundError(f"no epoch-*.pt in {exp_dir}")
    return max(epochs)


def _write_k2_cpu_stub() -> Path:
    """CUDA k2 needs libcuda.so.1. Export runs on CPU, so shadow ``k2`` for import
    + ``SymbolTable``; ``convert_scaled_to_non_scaled`` removes Swoosh k2 ops
    before the ONNX trace.
    """
    root = Path("/tmp/k2_cpu_stub")
    pkg = root / "k2"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text(
        """from pathlib import Path

class Fsa:  # noqa: D401
    pass

class RaggedTensor:
    pass

class DeterminizeWeightPushingType:
    pass

class SymbolTable(dict):
    @property
    def symbols(self):
        return list(self.keys())

    @classmethod
    def from_file(cls, path):
        table = cls()
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            sym, idx = line.rsplit(" ", 1)
            table[sym] = int(idx)
        return table

def __getattr__(name):
    return type(name, (), {})
""",
        encoding="utf-8",
    )
    (pkg / "version.py").write_text(
        "\n".join(
            [
                '__version__ = "cpu-stub"',
                '__build_type__ = "Release"',
                '__git_sha1__ = "stub"',
                "def __getattr__(name):",
                '    return "stub"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    return root


def _run_export_impl(
    run_name: str,
    epoch: int,
    avg: int,
    chunk_size: int,
    left_context: int,
) -> dict:
    import json
    import shutil
    import subprocess

    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    sys.path.insert(0, "/app")
    from zipformer_ctc_utils import (
        ARCH_FLAGS,
        compute_T_hop,
        diff_io_json,
        io_inputs_match,
        io_json_from_session,
        load_json,
    )

    work = Path("/vol/work")
    work.mkdir(parents=True, exist_ok=True)
    recipe = _prepare_recipe(work, smoke=False)
    tokens = _write_tokens()
    exp_dir = Path(f"/vol/exp/{run_name}")
    if not exp_dir.exists():
        raise FileNotFoundError(exp_dir)
    if epoch <= 0:
        epoch = _latest_epoch(exp_dir)
    use_avg_model = 0 if avg <= 1 else 1
    if use_avg_model and epoch - avg < 1:
        print(f"avg={avg} too large for epoch={epoch}; falling back to avg=1")
        avg = 1
        use_avg_model = 0

    cmd = [
        sys.executable,
        str(recipe / "export-onnx-streaming-ctc.py"),
        "--exp-dir",
        str(exp_dir),
        "--tokens",
        str(tokens),
        "--epoch",
        str(epoch),
        "--avg",
        str(avg),
        "--use-averaged-model",
        str(use_avg_model),
        "--chunk-size",
        str(chunk_size),
        "--left-context-frames",
        str(left_context),
        "--enable-int8-quantization",
        "0",
        "--dynamic-batch",
        "0",
        *ARCH_FLAGS,
    ]
    env = os.environ.copy()
    stub = _write_k2_cpu_stub()
    env["PYTHONPATH"] = f"{stub}:/opt/icefall:/app:" + env.get("PYTHONPATH", "")
    print("export cmd:", " ".join(cmd))
    print("k2 cpu stub on PYTHONPATH (no libcuda on export workers)")
    subprocess.run(cmd, cwd=str(recipe), env=env, check=True)

    produced = exp_dir / (
        f"ctc-epoch-{epoch}-avg-{avg}-chunk-{chunk_size}-left-{left_context}.onnx"
    )
    if not produced.is_file():
        cands = sorted(exp_dir.glob("ctc-*.onnx"))
        if not cands:
            raise FileNotFoundError("export produced no ctc-*.onnx")
        produced = cands[-1]
        print(f"using {produced}")

    export_dir = Path(f"/vol/exports/{run_name}")
    export_dir.mkdir(parents=True, exist_ok=True)
    model_path = export_dir / "model.onnx"
    shutil.copy2(produced, model_path)

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    t_expected, hop = compute_T_hop(chunk_size)
    x_shape = None
    for inp in sess.get_inputs():
        if inp.name == "x":
            x_shape = list(inp.shape)
    t_from_x = int(x_shape[1]) if x_shape and x_shape[1] not in (None, "N") else t_expected
    print(f"T/hop derivation: chunk_size={chunk_size} -> compute_T_hop={t_expected}/{hop}; x dim={x_shape}")
    io = io_json_from_session(
        sess.get_inputs(),
        sess.get_outputs(),
        model="model.onnx",
        T=t_from_x,
        hop=hop,
        feature_dim=80,
        vocab_size=251,
    )
    io_path = export_dir / "model.io.json"
    io_path.write_text(json.dumps(io, indent=4) + "\n", encoding="utf-8")
    ref = load_json("/app/zipformer-io.json")
    diffs = diff_io_json(io, ref)
    print("io.json diff vs reference:")
    if diffs:
        for d in diffs:
            print(" ", d)
    else:
        print("  (names+dims match)")
    if not io_inputs_match(io, ref):
        raise RuntimeError(
            f"ONNX input names/dims differ from reference: {diffs}"
        )

    int8_path = export_dir / "model.int8.onnx"
    quantize_dynamic(
        model_input=str(model_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul"],
    )
    fp32_bytes = model_path.stat().st_size
    int8_bytes = int8_path.stat().st_size
    print(f"fp32={fp32_bytes} bytes ({fp32_bytes / 1e6:.1f} MB)")
    print(f"int8={int8_bytes} bytes ({int8_bytes / 1e6:.1f} MB)")

    sha = Path("/opt/icefall.sha").read_text().strip()
    ckpt = exp_dir / f"epoch-{epoch}.pt"
    param_count = None
    try:
        import torch

        blob = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        sd = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        param_count = int(sum(v.numel() for v in sd.values() if hasattr(v, "numel")))
    except Exception as exc:
        print(f"param count failed: {exc}")
    print(f"param_count={param_count}")

    meta = {
        "run": run_name,
        "epoch": epoch,
        "avg": avg,
        "chunk": chunk_size,
        "left": left_context,
        "icefall_sha": sha,
        "k2": K2_VERSION,
        "param_count": param_count,
        "fp32_bytes": fp32_bytes,
        "int8_bytes": int8_bytes,
        "T": io["T"],
        "hop": io["hop"],
        "io_diff": diffs,
    }
    (export_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"\nDownload (do not auto-fetch):\n"
        f"  modal volume get zipformer-ctc-training /exports/{run_name} "
        f"/tmp/zipformer-{run_name}\n"
    )
    return meta


@app.function(
    image=image,
    gpu=_GPU_SPEC,
    cpu=max(16, 8 * WORLD_SIZE),
    memory=65536,
    timeout=24 * 3600,
    volumes={"/vol": vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def train(
    run_name: str,
    num_epochs: int = 40,
    max_duration: int = 1200,
    sources: str = "everyayah,qua,iqra,retasy,tlog",
    smoke: bool = False,
    synthetic: bool = False,
    limit_cuts: int = 0,
    do_export: bool = False,
    chunk_size: int = 24,
    left_context_frames: int = 256,
    avg: int = 10,
    start_epoch: int = 1,
) -> dict:
    import subprocess
    import time

    sys.path.insert(0, "/app")
    t0 = time.time()
    work = Path("/vol/work")
    work.mkdir(parents=True, exist_ok=True)
    Path("/vol/exp").mkdir(parents=True, exist_ok=True)
    Path("/vol/manifests").mkdir(parents=True, exist_ok=True)

    src_list = [s.strip() for s in sources.split(",") if s.strip()]
    if smoke and not synthetic:
        missing = [
            s
            for s in src_list
            if not (Path("/vol/manifests") / f"{s}_cuts_fbank.jsonl.gz").is_file()
        ]
        if missing:
            print(f"smoke cuts missing {missing}; falling back to --synthetic")
            synthetic = True
    if synthetic:
        _build_synthetic_cuts(20)
        src_list = ["synthetic"]
        sources = "synthetic"
        if smoke and limit_cuts <= 0:
            limit_cuts = 800

    _write_tokens()
    recipe = _prepare_recipe(work, smoke=smoke)
    exp_dir = Path(f"/vol/exp/{run_name}")
    exp_dir.mkdir(parents=True, exist_ok=True)

    if smoke:
        num_epochs = 1
        max_duration = min(max_duration, 200)
        if limit_cuts <= 0:
            limit_cuts = 800

    cmd = [
        sys.executable,
        str(recipe / "train.py"),
        "--world-size",
        str(WORLD_SIZE),
        "--num-epochs",
        str(num_epochs),
        "--start-epoch",
        str(start_epoch),
        "--exp-dir",
        str(exp_dir),
        "--bpe-model",
        "/app/tokens.js",
        "--use-fp16",
        "1",
        "--base-lr",
        "0.045",
        "--max-duration",
        str(max_duration),
        "--full-libri",
        "1",
        "--enable-musan",
        "0",
        "--num-workers",
        "8" if not smoke else "2",
        "--drop-last",
        "0" if smoke else "1",
        "--num-buckets",
        "4" if smoke else "30",
        "--manifest-dir",
        "/vol/manifests",
        "--sources",
        sources,
        "--limit-cuts",
        str(limit_cuts),
        *_arch_train_flags(),
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = "/opt/icefall:/app:" + env.get("PYTHONPATH", "")
    print("train cmd:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(recipe), env=env, check=True)
    rows = _parse_metrics(exp_dir)
    elapsed = time.time() - t0
    print(f"train finished in {elapsed:.1f}s, {len(rows)} metric rows")

    result = {
        "run_name": run_name,
        "exp_dir": str(exp_dir),
        "num_epochs": num_epochs,
        "elapsed_s": elapsed,
        "synthetic": synthetic,
        "sources": sources,
        "metrics_head": rows[:8],
        "metrics_tail": rows[-8:],
        "icefall_sha": Path("/opt/icefall.sha").read_text().strip(),
    }
    if do_export:
        epoch = _latest_epoch(exp_dir)
        export_avg = 1 if smoke or epoch < 2 else min(avg, epoch)
        result["export"] = _run_export_impl(
            run_name,
            epoch=epoch,
            avg=export_avg,
            chunk_size=chunk_size,
            left_context=left_context_frames,
        )
    vol.commit()
    return result


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=2 * 3600,
    volumes={"/vol": vol},
)
def export_onnx(
    run_name: str,
    epoch: int = 0,
    avg: int = 10,
    chunk_size: int = 24,
    left_context_frames: int = 256,
) -> dict:
    meta = _run_export_impl(run_name, epoch, avg, chunk_size, left_context_frames)
    vol.commit()
    return meta


@app.local_entrypoint()
def main(
    run_name: str,
    num_epochs: int = 40,
    max_duration: int = 1200,
    sources: str = "everyayah,qua,iqra,retasy,tlog",
    smoke: bool = False,
    synthetic: bool = False,
    export_only: bool = False,
    epoch: int = 0,
    avg: int = 10,
    chunk_size: int = 24,
    left_context_frames: int = 256,
    start_epoch: int = 1,
):
    """Train and/or export. Smoke implies 1 epoch, max-duration 200, limit-cuts 800."""
    full_cmd = (
        "ZIPFORMER_GPU=H100:4 modal run --detach scripts/train_zipformer_ctc_modal.py "
        "--run-name trackA-v1 --num-epochs 40 --max-duration 1200 "
        "--sources everyayah,qua,iqra,retasy,tlog"
    )
    export_cmd = (
        "modal run --detach scripts/train_zipformer_ctc_modal.py "
        f"--run-name trackA-v1 --export-only --epoch 40 --avg 10 "
        f"--chunk-size {chunk_size} --left-context-frames {left_context_frames}"
    )
    if export_only:
        meta = export_onnx.remote(
            run_name=run_name,
            epoch=epoch,
            avg=avg,
            chunk_size=chunk_size,
            left_context_frames=left_context_frames,
        )
        print("export metadata:", meta)
        return
    result = train.remote(
        run_name=run_name,
        num_epochs=1 if smoke else num_epochs,
        max_duration=200 if smoke else max_duration,
        sources="everyayah" if smoke and not synthetic else sources,
        smoke=smoke,
        synthetic=synthetic,
        limit_cuts=800 if smoke else 0,
        do_export=False,
        chunk_size=chunk_size,
        left_context_frames=left_context_frames,
        avg=avg,
        start_epoch=start_epoch,
    )
    print("train result:", result)
    if smoke:
        export_avg = 1
        export_epoch = 1
        meta = export_onnx.remote(
            run_name=run_name,
            epoch=export_epoch,
            avg=export_avg,
            chunk_size=chunk_size,
            left_context_frames=left_context_frames,
        )
        print("export metadata:", meta)
        result["export"] = meta
    print("\nFull (detached) training command (do NOT run from this smoke job):")
    print(" ", full_cmd)
    print("Then export:")
    print(" ", export_cmd)
    print(
        f"  modal volume get zipformer-ctc-training /exports/{run_name} "
        f"/tmp/zipformer-{run_name}"
    )
