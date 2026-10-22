import { fetchJson } from "../api";
import { getApiBase, withTokenParam } from "../connection";

export type Preset = "baseline" | "quality" | "ultra";
export type InputMode = "video" | "images" | "dataset";
export type GpuName = "local" | "T4" | "L4" | "A100" | "H100" | "G4";

/** Mirrors backend `SpirulaSettings`; null = take the preset's value. */
export interface SpirulaSettings {
  preset: Preset;
  iterations: number | null;
  cap_max: number | null;
  fps: number;
  sfm_quality: "high" | "extreme" | null;
  geometry_model: "moge2-vits" | "moge2-vitb" | "moge2-vitl";
  generate_depth: boolean;
  appearance: boolean;
  floater_suppression: "off" | "mild" | "strong";
  resolution_divisor: 1 | 2 | 4;
  allow_partial: boolean;
}

export interface PresetInfo {
  label: string;
  quality: string;
  iterations: number;
  cap_max: number;
  sfm_quality: "high" | "extreme";
}

export interface SpirulaInfo {
  available: boolean;
  binary: string | null;
  version: string | null;
  error: string | null;
  presets: Record<Preset, PresetInfo>;
  defaults: SpirulaSettings;
  gpu_profiles: Record<GpuName, { label: string; vram_gib: number | null; speed: number }>;
  viewer_port: number;
}

export interface GpuReading {
  name: string;
  total_mib: number;
  used_mib: number;
  free_mib: number;
  util_pct: number | null;
}

export interface GpuStatus {
  available: boolean;
  gpus: GpuReading[];
  error: string | null;
}

export interface Estimate {
  gpu: GpuName;
  gpu_label: string;
  vram_mib: number;
  vram_splats_mib: number;
  vram_base_mib: number;
  available_mib: number | null;
  fits: boolean | null;
  max_splats: number | null;
  train_seconds: number;
  train_megapixels: number;
}

export interface ResolvedRun {
  quality: string;
  iterations: number;
  cap_max: number;
  fps: number;
  sfm_quality: string;
  geometry_model: string;
  generate_depth: boolean;
  appearance: boolean;
  floater_suppression: string;
  resolution_divisor: number;
  allow_partial: boolean;
}

export interface EstimateResponse {
  resolved: ResolvedRun;
  estimates: Estimate[];
  gpu: GpuStatus;
}

export interface ProbeResult {
  kind: InputMode;
  frames: number | null;
  width: number | null;
  height: number | null;
  duration?: number;
  has_sfm?: boolean;
  images_subdir?: boolean;
  warning?: string;
}

export interface SpirulaRun {
  run_id: string;
  name: string;
  status: "running" | "completed" | "failed" | "cancelled";
  input_mode: InputMode;
  input_path: string;
  settings: ResolvedRun;
  started_at: number;
  finished_at: number | null;
  splat: string | null;
  error: string | null;
}

export interface RunLog {
  label: string | null;
  labels?: string[];
  lines: string[];
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const getSpirulaInfo = () => fetchJson<SpirulaInfo>("/spirula/info");
export const getGpuStatus = () => fetchJson<GpuStatus>("/system/gpu");

export function estimateRun(settings: SpirulaSettings, width?: number | null, height?: number | null,
  init?: RequestInit): Promise<EstimateResponse> {
  return fetchJson<EstimateResponse>("/spirula/estimate", {
    ...json({ settings, ...(width && height ? { image_width: width, image_height: height } : {}) }),
    ...init,
  });
}

export function probeInput(input_mode: InputMode, input_path: string, fps: number,
  init?: RequestInit): Promise<ProbeResult> {
  return fetchJson<ProbeResult>("/spirula/probe", { ...json({ input_mode, input_path, fps }), ...init });
}

export function startLocalRun(req: { input_mode: InputMode; input_path: string; settings: SpirulaSettings; name?: string }) {
  return fetchJson<{ job_id: string; run_id: string }>("/spirula/local", json(req));
}

export const listSpirulaRuns = () => fetchJson<SpirulaRun[]>("/spirula/runs");
export const getRunLog = (runId: string, tail = 200) =>
  fetchJson<RunLog>(`/spirula/runs/${encodeURIComponent(runId)}/log?tail=${tail}`);
export const runSplatUrl = (runId: string) =>
  withTokenParam(`${getApiBase()}/spirula/runs/${encodeURIComponent(runId)}/splat`);
