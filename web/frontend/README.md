# Tilawa web frontend

Vanilla TypeScript + Vite 7. Default engine is the shipped FastConformer CTC model.

## Zipformer demo (`interp-gentle-a0.5`)

Streaming Zipformer2-CTC behind `?engine=zipformer` (or `localStorage.tilawaEngine=zipformer`). Default URL stays FastConformer.

Assets are gitignored (ONNX + NPL-derived lexicon). Fetch them once:

```bash
bash web/frontend/scripts/fetch-zipformer-assets.sh
# if the local export is missing:
#   modal volume get zipformer-ctc-training /exports/interp-gentle-a0.5 /tmp/zipformer-interp-gentle-a0.5
```

Then from `web/frontend` (symlink `node_modules` from the main checkout if you are in a worktree):

```bash
npm run dev
# open http://localhost:5173/?engine=zipformer
```

Node smoke (onnxruntime-node, no browser):

```bash
cd web/frontend
npx tsx test/zipformer-node-smoke.ts
```

int8 ONNX sha256 `eaf099af…` (66 MB). Threads stay off (`numThreads=1`, EP `wasm`). First load is ~66 MB into IndexedDB under `zipformer-interp-gentle-a05-int8`.
