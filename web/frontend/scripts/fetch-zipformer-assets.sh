#!/usr/bin/env bash
# Copy interp-gentle-a0.5 int8 ONNX + I/O manifest + zipformer phoneme corpus
# into the Vite public/ tree. The ONNX and lexicon are gitignored (NPL-derived).
# io.json is committed in-repo; this script only copies it when missing.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
FRONTEND="$(cd "$HERE/.." && pwd)"
DEST_MODELS="$FRONTEND/public/models"
ONNX_NAME="zipformer_interp_gentle_a05.int8.onnx"
IO_NAME="zipformer_interp_gentle_a05.io.json"
CORPUS_NAME="zipformer_quran.json"
RELEASE_BASE="https://github.com/yazinsai/tilawa/releases/download/v0.3.0"

SRC_DIR="${ZIPFORMER_EXPORT:-/tmp/zipformer-interp-gentle-a0.5/interp-gentle-a0.5}"
MAIN_PUBLIC="/Users/rock/ai/projects/offline-tarteel/web/frontend/public"
MAIN_CORPUS="/Users/rock/ai/projects/offline-tarteel/data/zipformer/quran.json"
REPO_ROOT="$(git -C "$FRONTEND" rev-parse --show-toplevel 2>/dev/null || echo "")"
WORKTREE_CORPUS="${REPO_ROOT:+$REPO_ROOT/../../data/zipformer/quran.json}"

mkdir -p "$DEST_MODELS"

download() {
  local url="$1" dest="$2"
  echo "Downloading $url"
  curl -fL --retry 3 --retry-delay 2 -o "$dest" "$url"
}

resolve_onnx() {
  if [[ -f "$SRC_DIR/model.int8.onnx" ]]; then echo "$SRC_DIR/model.int8.onnx"; return; fi
  if [[ -f "$MAIN_PUBLIC/models/$ONNX_NAME" ]]; then echo "$MAIN_PUBLIC/models/$ONNX_NAME"; return; fi
  echo ""
}

resolve_io() {
  if [[ -f "$DEST_MODELS/$IO_NAME" ]]; then echo "$DEST_MODELS/$IO_NAME"; return; fi
  if [[ -f "$SRC_DIR/model.io.json" ]]; then echo "$SRC_DIR/model.io.json"; return; fi
  if [[ -f "$MAIN_PUBLIC/models/$IO_NAME" ]]; then echo "$MAIN_PUBLIC/models/$IO_NAME"; return; fi
  echo ""
}

resolve_corpus() {
  for candidate in "$MAIN_CORPUS" ${WORKTREE_CORPUS:+"$WORKTREE_CORPUS"} "${ZIPFORMER_CORPUS:-}" "$MAIN_PUBLIC/$CORPUS_NAME"; do
    if [[ -n "$candidate" && -f "$candidate" ]]; then
      echo "$candidate"
      return
    fi
  done
  echo ""
}

ONNX_SRC="$(resolve_onnx)"
IO_SRC="$(resolve_io)"
CORPUS_SRC="$(resolve_corpus)"

if [[ -n "$ONNX_SRC" ]]; then
  cp "$ONNX_SRC" "$DEST_MODELS/$ONNX_NAME"
else
  download "$RELEASE_BASE/$ONNX_NAME" "$DEST_MODELS/$ONNX_NAME"
fi

if [[ -n "$IO_SRC" && "$IO_SRC" != "$DEST_MODELS/$IO_NAME" ]]; then
  cp "$IO_SRC" "$DEST_MODELS/$IO_NAME"
fi

if [[ -n "$CORPUS_SRC" ]]; then
  cp "$CORPUS_SRC" "$FRONTEND/public/$CORPUS_NAME"
else
  download "$RELEASE_BASE/$CORPUS_NAME" "$FRONTEND/public/$CORPUS_NAME"
fi

if [[ ! -f "$DEST_MODELS/$ONNX_NAME" || ! -f "$FRONTEND/public/$CORPUS_NAME" ]]; then
  echo "Failed to materialize Zipformer browser assets."
  echo "Tried local export ($SRC_DIR), main checkout public/, and $RELEASE_BASE"
  echo "Manual fallback:"
  echo "  modal volume get zipformer-ctc-training /exports/interp-gentle-a0.5 /tmp/zipformer-interp-gentle-a0.5"
  exit 1
fi

echo "Ready:"
echo "  $DEST_MODELS/$ONNX_NAME"
echo "  $DEST_MODELS/$IO_NAME"
echo "  $FRONTEND/public/$CORPUS_NAME"
echo "sha256 int8: $(shasum -a 256 "$DEST_MODELS/$ONNX_NAME" | awk '{print $1}')"
