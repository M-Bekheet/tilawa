import { realpathSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const root = path.dirname(fileURLToPath(import.meta.url));
let nodeModulesReal: string | undefined;
try {
  nodeModulesReal = realpathSync(path.join(root, "node_modules"));
} catch {
  nodeModulesReal = undefined;
}

export default defineConfig({
  resolve: {
    alias: {
      "@tilawa/core": fileURLToPath(
        new URL("../../packages/core/src/index.ts", import.meta.url),
      ),
    },
  },
  worker: {
    format: "es",
  },
  optimizeDeps: {
    exclude: ["onnxruntime-web"],
  },
  server: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
    // Worktree `node_modules` is a symlink into the main checkout; Vite's
    // default fs.allow is the worktree root and 403s the ORT wasm fetch.
    fs: {
      allow: [root, ...(nodeModulesReal ? [nodeModulesReal] : [])],
    },
  },
});
