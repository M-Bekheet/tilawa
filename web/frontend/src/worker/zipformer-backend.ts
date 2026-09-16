import * as ort from "onnxruntime-web/wasm";
import type { WorkerInbound, WorkerOutbound } from "../lib/types";
import { loadModel } from "./model-cache";
import {
  DISPLAY_QURAN_URL,
  PROMPTER_QURAN_URL,
  ZIPFORMER_CACHE_KEY,
  ZIPFORMER_IO_URL,
  ZIPFORMER_MODEL_URL,
  ZipformerHost,
  displayQuranFromRaw,
} from "./zipformer-session";
import type { ZipformerIo } from "../vendor/alketab-engine/browser/zipformerRunner.js";

let host: ZipformerHost | null = null;
let debugEnabled = false;

function post(msg: WorkerOutbound): void {
  self.postMessage(msg);
}

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} fetch failed: ${res.status}`);
  return res.json() as Promise<T>;
}

async function init(): Promise<void> {
  try {
    post({ type: "loading_status", message: "Loading Zipformer I/O manifest..." });
    const io = await fetchJson<ZipformerIo>(ZIPFORMER_IO_URL);

    post({ type: "loading_status", message: "Loading Quran text..." });
    const quranRaw = await fetchJson<unknown>(DISPLAY_QURAN_URL);
    const quranDb = displayQuranFromRaw(quranRaw);

    post({ type: "loading_status", message: "Loading phoneme corpus..." });
    const corpusJson = await fetchJson<unknown>(PROMPTER_QURAN_URL);

    post({ type: "loading_status", message: "Downloading Zipformer model..." });
    const modelBuffer = await loadModel(
      ZIPFORMER_MODEL_URL,
      (loaded, total) => {
        post({
          type: "loading",
          percent: total ? Math.round((loaded / total) * 100) : 0,
        });
      },
      ZIPFORMER_CACHE_KEY,
    );

    post({ type: "loading_status", message: "Creating Zipformer session..." });
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;
    host = await ZipformerHost.create({
      ort,
      modelBytes: modelBuffer,
      io,
      corpusJson,
      quranDb,
      executionProviders: ["wasm"],
    });
    host.debugEnabled = debugEnabled;
    post({ type: "ready" });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    console.error("Zipformer worker init failed:", message);
    post({ type: "error", message });
  }
}

self.onmessage = async (e: MessageEvent<WorkerInbound>) => {
  const msg = e.data;
  if (msg.type === "init") {
    await init();
  } else if (msg.type === "reset") {
    host?.reset();
  } else if (msg.type === "set_debug") {
    debugEnabled = msg.enabled;
    if (host) host.debugEnabled = msg.enabled;
  } else if (msg.type === "set_config") {
    // Zipformer host uses the alketab engine config, not FastConformer streaming knobs.
  } else if (msg.type === "stop") {
    if (!host) return;
    for (const m of await host.stop()) post(m);
  } else if (msg.type === "audio") {
    if (!host) return;
    for (const m of await host.feed(msg.samples)) post(m);
  }
};
