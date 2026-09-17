import * as ort from "onnxruntime-web/wasm";
import { ZipformerSession, type ZipformerIo } from "@tilawa/core";
import type { WorkerInbound, WorkerOutbound } from "../lib/types";
import { loadModel } from "./model-cache";
import {
  DISPLAY_QURAN_URL,
  ZIPFORMER_QURAN_URL,
  ZIPFORMER_CACHE_KEY,
  ZIPFORMER_IO_URL,
  ZIPFORMER_MODEL_URL,
} from "./zipformer-session";

let session: ZipformerSession | null = null;
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
    const quran = await fetchJson<unknown[]>(DISPLAY_QURAN_URL);

    post({ type: "loading_status", message: "Loading phoneme corpus..." });
    const corpus = await fetchJson<unknown>(ZIPFORMER_QURAN_URL);

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
    session = await ZipformerSession.create({
      ort,
      model: modelBuffer,
      io,
      corpus,
      quran,
      executionProviders: ["wasm"],
      debug: debugEnabled,
    });
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
    session?.reset();
  } else if (msg.type === "set_debug") {
    debugEnabled = msg.enabled;
    if (session) session.debugEnabled = msg.enabled;
  } else if (msg.type === "set_config") {
    // Zipformer runs the recitation-engine config, not FastConformer streaming knobs.
  } else if (msg.type === "stop") {
    if (!session) return;
    for (const m of await session.stop()) post(m);
  } else if (msg.type === "audio") {
    if (!session) return;
    for (const m of await session.feed(msg.samples)) post(m);
  }
};
