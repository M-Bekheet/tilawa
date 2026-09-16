#!/usr/bin/env bash
# Copy interp-gentle-a0.5 int8 ONNX + I/O manifest + prompter phoneme corpus
# into the Vite public/ tree. The ONNX and lexicon are gitignored (NPL-derived).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
FRONTEND="$(cd "$HERE/.." && pwd)"
DEST_MODELS="$FRONTEND/public/models"
ONNX_NAME="zipformer_interp_gentle_a05.int8.onnx"
IO_NAME="zipformer_interp_gentle_a05.io.json"

SRC_DIR="${ZIPFORMER_EXPORT:-/tmp/zipformer-interp-gentle-a0.5/interp-gentle-a0.5}"
MAIN_CORPUS="/Users/rock/ai/projects/offline-tarteel/data/prompter/quran.json"
REPO_ROOT="$(git -C "$FRONTEND" rev-parse --show-toplevel 2>/dev/null || echo "")"
WORKTREE_CORPUS="${REPO_ROOT:+$REPO_ROOT/../../data/prompter/quran.json}"

need_modal() {
  echo "Zipformer browser assets not found locally."
  echo "Download the export, then re-run this script:"
  echo "  modal volume get zipformer-ctc-training /exports/interp-gentle-a0.5 /tmp/zipformer-interp-gentle-a0.5"
  echo "Expected files:"
  echo "  $SRC_DIR/model.int8.onnx"
  echo "  $SRC_DIR/model.io.json"
  echo "  $MAIN_CORPUS   (or data/prompter/quran.json in the main checkout)"
  exit 1
}

ONNX_SRC="$SRC_DIR/model.int8.onnx"
IO_SRC="$SRC_DIR/model.io.json"
CORPUS_SRC=""
for candidate in "$MAIN_CORPUS" ${WORKTREE_CORPUS:+"$WORKTREE_CORPUS"} "${PROMPTER_CORPUS:-}"; do
  if [[ -n "$candidate" && -f "$candidate" ]]; then
    CORPUS_SRC="$candidate"
    break
  fi
done

if [[ ! -f "$ONNX_SRC" || ! -f "$IO_SRC" || -z "$CORPUS_SRC" ]]; then
  need_modal
fi

mkdir -p "$DEST_MODELS"
cp "$ONNX_SRC" "$DEST_MODELS/$ONNX_NAME"
cp "$IO_SRC" "$DEST_MODELS/$IO_NAME"
cp "$CORPUS_SRC" "$FRONTEND/public/prompter_quran.json"

echo "Copied:"
echo "  $DEST_MODELS/$ONNX_NAME"
echo "  $DEST_MODELS/$IO_NAME"
echo "  $FRONTEND/public/prompter_quran.json"
echo "sha256 int8: $(shasum -a 256 "$DEST_MODELS/$ONNX_NAME" | awk '{print $1}')"
