# Zipformer web backend (`interp-gentle-a0.5`)

Worktree `zipformer-web` @ `9fcde0e`. Flag: `?engine=zipformer` / `localStorage.tilawaEngine`. Default FastConformer worker (`src/worker/inference.ts`, `session.ts`) is untouched.

## Architecture

```
mic worklet (16 kHz, 150 ms)
        │
        ▼
main.ts  --engine=zipformer-->  worker/zipformer-backend.ts
                                 │
                                 ├─ IndexedDB cache key `zipformer-interp-gentle-a05-int8`
                                 ├─ onnxruntime-web/wasm, EP ["wasm"], numThreads=1
                                 └─ ZipformerHost (worker/zipformer-session.ts)
                                      KaldiFbank → ZipformerRunner.accept (T=61 / hop 48)
                                      GreedyCtcDecoder → RecitationEngine.feed
                                      completed/idle → tally + startSearch() + reset fbank/decoder/runner
                                      stop → 2 s silence + inputFinished + CTC flush + fallbackSearch
                                      zipformer-emission.ts maps tallies → verse_match / final_sequence
```

Vendored engine: `web/frontend/src/vendor/alketab-engine/` (copy of `experiments/prompter-zipformer/engine/`, JS unchanged). Types via `src/vendor/alketab-engine.d.ts`; JS excluded from `tsconfig.app.json`. The only upstream edit (already in the experiment tree) is `ZipformerRunner.create` taking an EP list.

Assets (gitignored except io.json):

| file | source | sha256 |
|---|---|---|
| `public/models/zipformer_interp_gentle_a05.int8.onnx` (66 MB) | `/tmp/zipformer-interp-gentle-a0.5/interp-gentle-a0.5/model.int8.onnx` | `eaf099afefbe5cc8c9aee74df864ce7cc69744271e4f3bb46ef6c17612fbe335` |
| fp32 sibling (not shipped) | `model.onnx` | `ae298211f0588e9ce4673b69b29246f2c849edc1c5eb30b7266edd4343d741b4` |
| `public/models/zipformer_interp_gentle_a05.io.json` | export `model.io.json` (inputs match vendored `zipformer-io.json`; extra `outputs[]` block) | committed |
| `public/prompter_quran.json` | `data/prompter/quran.json` | gitignored (NPL lexicon) |

`model-cache.ts` gained an optional 3rd `cacheKey` argument; FastConformer still uses the default key.

`vite.config.ts` `server.fs.allow` includes the realpath of `node_modules` so a worktree symlink into the main checkout can serve `ort-wasm-simd-threaded.wasm` (without this, ORT instantiates a 403 body as WASM: magic `0a 20 20 20`).

## Event mapping

Host loop mirrors `experiments/prompter-zipformer/harness.mjs` (Recognize mode). Pure decisions live in `src/lib/zipformer-emission.ts` (vitest, no ONNX):

- Snapshot `tracer.verdicts(true)` per tracker; dump into accumulated tallies on `completed` / `idle` / `relocated`.
- `verse_match` the first time an ayah has `ok+unsure ≥ max(1, 0.5·words)` and `wrong ≤ ok+unsure`. `confidence = (ok+unsure)/words`. Text/name/surrounding from display `QuranDB` (`/quran.json`).
- `word_progress` on `cursor` / `verdicts`; `matched_indices` = ok/unsure words of the cursor ayah.
- `raw_transcript` from CTC token `sym`s.
- `verse_candidate` on `located`.
- `stop`: 2 s silence drain + `fallbackSearch` (whole-ayah graded distance, basmala → 1:1) only if nothing was emitted; then `final_sequence`.

## Verified

**Unit:** `cd web/frontend && npx vitest run` → 5 files, 67 tests (9 new in `test/zipformer-emission.test.ts`).

**tsc:** `npx tsc --noEmit` (solution tsconfig) green. `inference.ts` / `session.ts` diff empty.

**Node smoke** (`npx tsx test/zipformer-node-smoke.ts` on `benchmark/test_corpus/001002.mp3`, onnxruntime-node `cpu`, ~2.3 s):

```json
{
  "eventTypes": { "raw_transcript": 10, "verse_match": 1, "final_sequence": 1 },
  "verse_match": [{ "surah": 1, "ayah": 2, "confidence": 1 }],
  "final_sequence": [{ "surah": 1, "ayah": 2, "confidence": 1 }],
  "refs": ["1:2"]
}
```

Transcript: `ءَلحَمدُلِللَااهِرَببِلعَاالَمِۦۦۦۦن`. Short clip: no live `located`/`word_progress` (fallback-at-stop is allowed).

**Browser:** `npm run dev` → `http://localhost:5173/?engine=zipformer`. After “Try with my voice”, status line **Zipformer ready**, Start button shown. Console had no worker errors after `server.fs.allow`. No virtual-mic recitation pass (optional in the brief).

## Owner demo

From the worktree (Node 22):

```bash
ln -sfn /Users/rock/ai/projects/offline-tarteel/web/frontend/node_modules web/frontend/node_modules
bash web/frontend/scripts/fetch-zipformer-assets.sh
# if the export is missing:
#   modal volume get zipformer-ctc-training /exports/interp-gentle-a0.5 /tmp/zipformer-interp-gentle-a0.5
cd web/frontend && npm run dev
# open http://localhost:5173/?engine=zipformer
```

Default URL is still FastConformer. `localStorage.tilawaEngine` sticks after a `?engine=` visit.

## Known gaps

- Threads off (`numThreads=1`); first load ~66 MB into IndexedDB.
- Engine JS licence unstated (reference-only); ONNX + `prompter_quran.json` are NPL-1.2 Derivatives — see `NOTICE.md`.
- Worktree needs the `node_modules` symlink **and** the Vite `fs.allow` realpath (or ORT wasm 403s).
- Short one-ayah clips often emit only at `stop` via fallback, same as the Node harness.
- Zipformer ignores FastConformer `set_config` knobs (alketab `DEFAULT_CONFIG` + `PROMPTER_*` semantics).
