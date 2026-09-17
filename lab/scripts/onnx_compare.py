"""Compare two ONNX graphs: file sha256, per-initializer tensor bytes, metadata_props.

stdlib + onnx + numpy. Optional dump-io uses onnxruntime + zipformer_ctc_utils
(io_json_from_session) when those are importable.

Usage:
  .venv/bin/python scripts/onnx_compare.py a.onnx b.onnx
  .venv/bin/python scripts/onnx_compare.py --vendored data/zipformer/quran_phoneme_zipformer.onnx \\
      --candidates data/zipformer/reference/v3.onnx data/zipformer/reference/v3.1.onnx
  .venv/bin/python scripts/onnx_compare.py dump-io model.onnx --out io.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_raw_bytes(tensor: onnx.TensorProto) -> bytes:
    arr = numpy_helper.to_array(tensor)
    return np.ascontiguousarray(arr).tobytes()


def initializer_hashes(model: onnx.ModelProto) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for init in model.graph.initializer:
        raw = tensor_raw_bytes(init)
        out[init.name] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "nbytes": len(raw),
            "dtype": int(init.data_type),
            "dims": list(init.dims),
        }
    return out


def metadata_props(model: onnx.ModelProto) -> dict[str, str]:
    return {p.key: p.value for p in model.metadata_props}


def load_model(path: Path) -> onnx.ModelProto:
    return onnx.load(str(path), load_external_data=True)


def compare(a_path: Path, b_path: Path) -> dict:
    a_sha = sha256_file(a_path)
    b_sha = sha256_file(b_path)
    a = load_model(a_path)
    b = load_model(b_path)
    a_init = initializer_hashes(a)
    b_init = initializer_hashes(b)
    a_names = set(a_init)
    b_names = set(b_init)
    shared = sorted(a_names & b_names)
    identical = [n for n in shared if a_init[n]["sha256"] == b_init[n]["sha256"]]
    differing = [n for n in shared if a_init[n]["sha256"] != b_init[n]["sha256"]]
    a_meta = metadata_props(a)
    b_meta = metadata_props(b)
    meta_keys = sorted(set(a_meta) | set(b_meta))
    meta_diff = {
        k: {"a": a_meta.get(k), "b": b_meta.get(k)}
        for k in meta_keys
        if a_meta.get(k) != b_meta.get(k)
    }
    return {
        "a": {"path": str(a_path), "size": a_path.stat().st_size, "sha256": a_sha},
        "b": {"path": str(b_path), "size": b_path.stat().st_size, "sha256": b_sha},
        "file_identical": a_sha == b_sha,
        "n_init_a": len(a_init),
        "n_init_b": len(b_init),
        "n_shared": len(shared),
        "n_identical": len(identical),
        "n_differing": len(differing),
        "only_in_a": sorted(a_names - b_names),
        "only_in_b": sorted(b_names - a_names),
        "metadata_a": a_meta,
        "metadata_b": b_meta,
        "metadata_diff": meta_diff,
        "differing_names": differing,
        "param_identical": (
            a_names == b_names
            and not differing
            and not (a_names - b_names)
            and not (b_names - a_names)
        ),
        "initializers_a": a_init,
        "initializers_b": b_init,
    }


def dump_io(model_path: Path, out_path: Path | None, t: int, hop: int) -> dict:
    import onnxruntime as ort

    from zipformer_ctc_utils import io_json_from_session

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    io = io_json_from_session(
        sess.get_inputs(),
        sess.get_outputs(),
        model=model_path.name,
        T=t,
        hop=hop,
    )
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(io, indent=2) + "\n", encoding="utf-8")
    return io


def _print_compare(report: dict, *, verbose: bool) -> None:
    print(f"A  {report['a']['size']:>12}  {report['a']['sha256']}  {report['a']['path']}")
    print(f"B  {report['b']['size']:>12}  {report['b']['sha256']}  {report['b']['path']}")
    print(f"file_identical={report['file_identical']}  param_identical={report['param_identical']}")
    print(
        f"initializers shared={report['n_shared']} identical={report['n_identical']} "
        f"differing={report['n_differing']} only_a={len(report['only_in_a'])} only_b={len(report['only_in_b'])}"
    )
    if report["metadata_a"] or report["metadata_b"]:
        print("metadata_a:", json.dumps(report["metadata_a"], ensure_ascii=False))
        print("metadata_b:", json.dumps(report["metadata_b"], ensure_ascii=False))
    if report["metadata_diff"]:
        print("metadata_diff:", json.dumps(report["metadata_diff"], ensure_ascii=False))
    if report["only_in_a"]:
        print("only_in_a:", report["only_in_a"][:20], ("..." if len(report["only_in_a"]) > 20 else ""))
    if report["only_in_b"]:
        print("only_in_b:", report["only_in_b"][:20], ("..." if len(report["only_in_b"]) > 20 else ""))
    if verbose and report["differing_names"]:
        print("differing:")
        for name in report["differing_names"][:40]:
            a = report["initializers_a"][name]
            b = report["initializers_b"][name]
            print(f"  {name}  a={a['sha256'][:12]} ({a['nbytes']})  b={b['sha256'][:12]} ({b['nbytes']})")
        if len(report["differing_names"]) > 40:
            print(f"  ... {len(report['differing_names']) - 40} more")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd")

    dump = sub.add_parser("dump-io", help="Write zipformer-io.json from an ONNX session")
    dump.add_argument("model")
    dump.add_argument("--out", type=str, default="")
    dump.add_argument("--T", type=int, default=61)
    dump.add_argument("--hop", type=int, default=48)

    parser.add_argument("a", nargs="?", help="first onnx (or vendored with --vendored)")
    parser.add_argument("b", nargs="?", help="second onnx")
    parser.add_argument("--vendored", type=str, default="")
    parser.add_argument("--candidates", nargs="+", default=[])
    parser.add_argument("--json", type=str, default="")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.cmd == "dump-io":
        out = Path(args.out) if args.out else None
        io = dump_io(Path(args.model), out, args.T, args.hop)
        print(json.dumps({k: io[k] for k in ("model", "T", "hop", "featureDim", "vocabSize")}, indent=2))
        print(f"inputs={len(io['inputs'])} outputs={len(io.get('outputs') or [])}")
        if out:
            print(f"wrote {out}")
        return

    if args.vendored or args.candidates:
        vendored = Path(args.vendored or args.a)
        candidates = [Path(p) for p in (args.candidates or ([args.b] if args.b else []))]
        reports = []
        best = None
        for cand in candidates:
            report = compare(vendored, cand)
            reports.append(report)
            print("=" * 72)
            _print_compare(report, verbose=args.verbose)
            score = (
                int(report["file_identical"]) * 1_000_000
                + int(report["param_identical"]) * 100_000
                + report["n_identical"]
            )
            if best is None or score > best[0]:
                best = (score, cand, report)
        if best is not None:
            print("=" * 72)
            print(
                f"closest to vendored: {best[1].name}  "
                f"file_identical={best[2]['file_identical']} "
                f"param_identical={best[2]['param_identical']} "
                f"identical_inits={best[2]['n_identical']}/{best[2]['n_shared']}"
            )
        if args.json:
            Path(args.json).write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
        return

    if not args.a or not args.b:
        parser.error("need a.onnx b.onnx, or --vendored + --candidates")
    report = compare(Path(args.a), Path(args.b))
    _print_compare(report, verbose=args.verbose)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
