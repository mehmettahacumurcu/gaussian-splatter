import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import * as fs from "node:fs";
import * as path from "node:path";

// @ts-expect-error process is a nodejs global
const host = process.env.TAURI_DEV_HOST;

// https://vite.dev/config/
export default defineConfig(async () => ({
  plugins: [
    react(),
    // Serve /worlds/* from the sibling worlds/ directory (../worlds relative to frontend/).
    // This avoids copying large .ply files into public/ while still making them
    // available at the expected URL during dev.
    {
      name: "serve-worlds-dir",
      configureServer(server) {
        // Resolve once at config load; URL-decode pathname so directories with
        // spaces (e.g. "Gaussian Splatter") survive the file:// round-trip.
        const configDir = path.dirname(
          decodeURIComponent(
            new URL(import.meta.url).pathname.replace(/^\/([A-Z]:)/, "$1"),
          ),
        );
        const worldsRoot = path.resolve(configDir, "..", "worlds");
        console.log(`[serve-worlds-dir] mounted /worlds -> ${worldsRoot}`);

        server.middlewares.use("/worlds", (req, res, next) => {
          // req.url already starts with "/" relative to the mount path.
          const reqPath = decodeURIComponent((req.url ?? "/").split("?")[0]);
          const filepath = path.join(worldsRoot, reqPath);
          fs.stat(filepath, (err, stat) => {
            if (err || !stat.isFile()) {
              console.warn(
                `[serve-worlds-dir] miss ${req.url} -> ${filepath} (${err?.code ?? "not a file"})`,
              );
              // Real 404, NOT next(): falling through hits Vite's SPA fallback,
              // which answers 200 + index.html — that poisons the world
              // availability probe and collider fetches for missing worlds.
              res.statusCode = 404;
              res.end("world file not found");
              return;
            }
            const ext = path.extname(filepath).toLowerCase();
            const contentType =
              ext === ".json"
                ? "application/json"
                : ext === ".ply"
                  ? "application/octet-stream"
                  : "application/octet-stream";
            res.setHeader("Content-Type", contentType);
            res.setHeader("Cache-Control", "no-cache");
            fs.createReadStream(filepath).pipe(res);
          });
        });
      },
    },
  ],

  // Vite options tailored for Tauri development and only applied in `tauri dev` or `tauri build`
  //
  // 1. prevent Vite from obscuring rust errors
  clearScreen: false,
  // 2. tauri expects a fixed port, fail if that port is not available
  server: {
    port: 1420,
    strictPort: true,
    host: host || false,
    hmr: host
      ? {
          protocol: "ws",
          host,
          port: 1421,
        }
      : undefined,
    watch: {
      // 3. tell Vite to ignore watching `src-tauri`
      ignored: ["**/src-tauri/**"],
    },
    // Allow Vite's fs middleware to reach files one level above the frontend root.
    fs: {
      allow: [".."],
    },
  },
}));
