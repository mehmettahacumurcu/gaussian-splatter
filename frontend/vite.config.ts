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
        server.middlewares.use("/worlds", (req, res, next) => {
          const filepath = path.join(
            path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Z]:)/, "$1")),
            "..",
            "worlds",
            req.url ?? "",
          );
          fs.stat(filepath, (err, stat) => {
            if (err || !stat.isFile()) return next();
            const ext = path.extname(filepath).toLowerCase();
            const contentType =
              ext === ".json" ? "application/json" : "application/octet-stream";
            res.setHeader("Content-Type", contentType);
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
