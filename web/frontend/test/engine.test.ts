import { describe, expect, it } from "vitest";
import {
  DEFAULT_ENGINE,
  ENGINE_STORAGE_KEY,
  engineLabel,
  resolveEngine,
} from "../src/lib/engine";

function memoryStorage(initial: Record<string, string> = {}) {
  const store = { ...initial };
  return {
    getItem(key: string): string | null {
      return Object.prototype.hasOwnProperty.call(store, key) ? store[key] : null;
    },
    setItem(key: string, value: string): void {
      store[key] = value;
    },
  };
}

describe("resolveEngine", () => {
  it("defaults to zipformer when no flag or localStorage is set", () => {
    expect(DEFAULT_ENGINE).toBe("zipformer");
    expect(resolveEngine("", null)).toBe("zipformer");
    expect(resolveEngine("", memoryStorage())).toBe("zipformer");
    expect(resolveEngine("?foo=bar", memoryStorage())).toBe("zipformer");
  });

  it("honours ?engine=fastconformer and persists it", () => {
    const storage = memoryStorage();
    expect(resolveEngine("?engine=fastconformer", storage)).toBe("fastconformer");
    expect(storage.getItem(ENGINE_STORAGE_KEY)).toBe("fastconformer");
  });

  it("reads a stored fastconformer preference when the URL has no engine", () => {
    const storage = memoryStorage({ [ENGINE_STORAGE_KEY]: "fastconformer" });
    expect(resolveEngine("", storage)).toBe("fastconformer");
  });

  it("lets the URL override localStorage", () => {
    const storage = memoryStorage({ [ENGINE_STORAGE_KEY]: "fastconformer" });
    expect(resolveEngine("engine=zipformer", storage)).toBe("zipformer");
    expect(storage.getItem(ENGINE_STORAGE_KEY)).toBe("zipformer");
  });

  it("labels engines for the status pill", () => {
    expect(engineLabel("zipformer")).toBe("Zipformer");
    expect(engineLabel("fastconformer")).toBe("FastConformer");
  });
});
