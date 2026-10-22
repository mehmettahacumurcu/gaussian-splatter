import type { TrainingPipelineNotebookSpec } from "../notebook/PipelineNotebookPanel";
import type { InputMode, SpirulaSettings } from "./spirulaApi";

export function formatGb(mib: number): string {
  const gb = mib / 1024;
  return gb >= 10 ? `${gb.toFixed(0)} GB` : `${gb.toFixed(1)} GB`;
}

export function formatDuration(seconds: number): string {
  if (seconds < 90) return `~${Math.max(1, Math.round(seconds / 60))} dk`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `~${minutes} dk`;
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;
  return m ? `~${h} sa ${m} dk` : `~${h} sa`;
}

/** 3_000_000 → "3M", 450_000 → "450K". */
export function formatCount(n: number): string {
  if (n >= 1e6) return `${+(n / 1e6).toFixed(n % 1e6 === 0 ? 0 : 1)}M`;
  if (n >= 1e3) return `${Math.round(n / 1e3)}K`;
  return String(n);
}

/** "3m", "3.5M", "500k", "2500000" → number (null if unparseable). */
export function parseCount(raw: string): number | null {
  const m = /^\s*([\d.,]+)\s*([kKmM]?)\s*$/.exec(raw);
  if (!m) return null;
  const value = Number(m[1].replace(",", "."));
  if (!Number.isFinite(value)) return null;
  const mult = m[2].toLowerCase() === "m" ? 1e6 : m[2].toLowerCase() === "k" ? 1e3 : 1;
  return Math.round(value * mult);
}

const APOSTROPHES = /[\u2019\u2018`]/g;

/**
 * Local path inside the Google Drive for desktop folder → MyDrive-relative path for the notebook.
 * `G:\Drive'ım\GaussianTests\a.MOV` with root `G:\Drive'ım` → `GaussianTests/a.MOV`.
 */
export function driveRelative(localPath: string, driveRoot: string): string | null {
  const norm = (p: string) => p.trim().replace(/^"|"$/g, "").replace(APOSTROPHES, "'").replace(/\//g, "\\").replace(/\\+$/, "");
  const path = norm(localPath);
  const root = norm(driveRoot);
  if (!root || path.length <= root.length + 1) return null;
  if (path.slice(0, root.length).toLowerCase() !== root.toLowerCase() || path[root.length] !== "\\") return null;
  const rel = path.slice(root.length + 1).split("\\").filter(Boolean);
  return rel.length ? rel.join("/") : null;
}

export function notebookSpec(mode: InputMode, drivePath: string, s: SpirulaSettings): TrainingPipelineNotebookSpec {
  const input_mode = mode === "video" ? "video" : drivePath.toLowerCase().endsWith(".zip") ? "dataset_zip" : "dataset_folder";
  return {
    pipeline: "spirula",
    input_mode,
    input_path: drivePath,
    preset: s.preset,
    iterations: s.iterations,
    max_gaussians: s.cap_max,
    fps: Math.max(1, Math.round(s.fps)),
    model_name: "0",
    allow_partial: s.allow_partial,
    generate_depth: s.generate_depth,
    geometry_model: s.geometry_model,
    sfm_quality: s.sfm_quality,
    appearance: s.appearance,
    floater_suppression: s.floater_suppression,
    resolution_divisor: s.resolution_divisor,
  };
}
