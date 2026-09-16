export type EngineName = "zipformer" | "fastconformer";

export const DEFAULT_ENGINE: EngineName = "zipformer";
export const ENGINE_STORAGE_KEY = "tilawaEngine";

export interface EngineStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

function asSearchParams(search: string | URLSearchParams): URLSearchParams {
  if (search instanceof URLSearchParams) return search;
  const raw = search.startsWith("?") ? search.slice(1) : search;
  return new URLSearchParams(raw);
}

export function resolveEngine(
  search: string | URLSearchParams = "",
  storage: EngineStorage | null = null,
): EngineName {
  const fromUrl = asSearchParams(search).get("engine");
  if (fromUrl === "zipformer" || fromUrl === "fastconformer") {
    try {
      storage?.setItem(ENGINE_STORAGE_KEY, fromUrl);
    } catch {
      /* ignore quota / private mode */
    }
    return fromUrl;
  }
  try {
    const stored = storage?.getItem(ENGINE_STORAGE_KEY);
    if (stored === "zipformer" || stored === "fastconformer") return stored;
  } catch {
    /* ignore */
  }
  return DEFAULT_ENGINE;
}

export function engineLabel(engine: EngineName): string {
  return engine === "zipformer" ? "Zipformer" : "FastConformer";
}
