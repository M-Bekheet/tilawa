# Changelog

## 0.2.1

Browser hang fix, packaging, and README that an outside consumer can follow.

- **EP default.** `{ ort, model }` now picks `["wasm"]` under onnxruntime-web and `["cpu"]` under onnxruntime-node (`listSupportedBackends` when present, else `ort.env.wasm`). Override with `executionProviders`. On web, `ort.env.wasm.numThreads` defaults to `1` unless the caller set it — pthread init hangs in workers without COOP/COEP.
- **FastConformer validation.** `createTilawaSession` / `createRecognitionSession({ engine: "fastconformer" })` throw `Error("fastconformer engine requires assets: vocab, ctcTokens, quran ...")` listing the missing keys before touching them.
- **Packaging.** `files` is `dist/`, `README.md`, `CHANGELOG.md`, `LICENSE`, `NOTICE.md`. Source maps and `src/` are no longer packed.
- **README.** Browser quick start uses `onnxruntime-web`, documents EP / `numThreads` / `wasmPaths` (Vite copies wasm by default), GitHub release asset URLs, licence split, and a Node snippet that matches the 300 ms / `stop()` / events path.

## 0.2.0

### Zipformer is now the default engine

The streaming Zipformer phoneme engine, previously only in the web demo, is now
part of the SDK and is what you get by default.

- **`createRecognitionSession(options)`** — new top-level factory with an
  `engine: "zipformer" | "fastconformer"` selector. Defaults to `"zipformer"`
  (`DEFAULT_ENGINE`). Returns a `RecognitionSession`: `feed()`, `stop()` /
  `flush()`, `reset()`, plus the underlying engine session.
- **`createZipformerSession(options)` / `ZipformerSession`** — the default engine
  directly. 16 kHz PCM → Kaldi fbank → streaming Zipformer2-CTC over 251
  tajweed-phoneme tokens → whole-Quran phoneme n-gram search → per-surah online DP
  tracker → per-word verdicts → the same `WorkerOutbound` verse events the
  FastConformer path emits. Exposes `transcript`, `tallies`, `verses`,
  `engineState`, and `config`.
- **ONNX injection, two shapes.** `{ ort, model }` for web and node (model bytes or
  a loader); `{ session, Tensor }` for React Native, where
  `InferenceSession.create()` takes a file path. No runtime is imported by the
  package.
- **`DEFAULT_ZIPFORMER_IO`** — the shipped model's I/O manifest is bundled, so only
  custom exports need an `io` override.
- The whole engine is exported from the package root (`src/recitation/`): engine,
  fbank, CTC decoder, corpus, search, alignment, phoneme cost table, and the
  event-emission helpers.

### Unchanged

`createTilawaSession(runner, assets, options?)`, `SessionRunner`, `TilawaAssets`,
`QuranDB`, `TextCTCDecoder`, `RecitationTracker`, and the `StreamingConfig`
presets keep their names and behaviour. Existing code runs untouched.

### Verified

- Zipformer streaming, median of 3 repeats: v1 53/53, v2 43/43 — 100% recall,
  precision, and sequence accuracy on both.
- Python lab harness on v1: 100% / 100% / 100%.
- 110 TypeScript test cases across `packages/core` (60) and `web/frontend` (50).

### React Native

No DOM, `Worker`, `fetch`, `TextDecoder`, or top-level `await` in the package;
typed arrays throughout. `BigInt64Array` (for the model's `int64` cache states) is
the only runtime requirement beyond ES2020, and a missing one now fails with an
explicit error. Setup walkthrough: `examples/react-native.md`.

## 0.1.0

Initial release. FastConformer text-CTC pipeline behind `createTilawaSession()`,
with the `SessionRunner` injection seam, `QuranDB` matching, and the streaming
`RecitationTracker`.
