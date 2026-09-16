# Tilawa web frontend

Vanilla TypeScript + Vite 7. Default engine is Zipformer2-CTC (`interp-gentle-a0.5` int8). FastConformer stays behind `?engine=fastconformer` (or `localStorage.tilawaEngine=fastconformer`).

## Zipformer (default, `interp-gentle-a0.5`)

Streaming Zipformer2-CTC. The status pill shows the active engine. Model artefacts are NPL-1.2; the tracker is the vendored alketab engine pending a native port.

Assets are gitignored (ONNX + NPL-derived lexicon). `zipformer_interp_gentle_a05.io.json` is committed. Fetch the rest once:

```bash
bash web/frontend/scripts/fetch-zipformer-assets.sh
# copies from a local export / the main checkout public/ tree, else downloads
# zipformer_interp_gentle_a05.int8.onnx and prompter_quran.json from
# GitHub release yazinsai/tilawa v0.3.0
```

Then from `web/frontend` (symlink `node_modules` from the main checkout if you are in a worktree):

```bash
npm run dev
# open http://localhost:5173/                      # Zipformer (default)
# open http://localhost:5173/?engine=fastconformer # previous engine
```

Node smoke (onnxruntime-node, no browser):

```bash
cd web/frontend
npx tsx test/zipformer-node-smoke.ts
```

Streaming stability (same JSON shape as the FastConformer report):

```bash
npx tsx test/stability-report.ts --engine=zipformer --repeats=3 --json=test/zipformer-default-stability.json
npx tsx test/stability-report.ts --engine=zipformer --repeats=3 --corpus=test_corpus_v2 --json=test/zipformer-default-v2-stability.json
```

int8 ONNX sha256 `eaf099af…` (66 MB). Threads stay off (`numThreads=1`, EP `wasm`). First load is ~66 MB into IndexedDB under `zipformer-interp-gentle-a05-int8`.
