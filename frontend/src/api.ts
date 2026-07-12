/**
 * FastAPI backend ile HTTP iletişimi.
 *
 * Backend address + auth token are managed in `connection.ts` and persisted
 * to localStorage. Default: http://127.0.0.1:8000 with no auth (local dev).
 *
 * The legacy `API_BASE` export is kept as a *snapshot at module load time*
 * for the topbar's display string only — runtime callers must use the
 * connection helpers so settings changes apply without a reload.
 */

import { getApiBase, getAuthToken, withTokenParam } from "./connection";
import type {
  GeneratedNotebook,
  StaticNotebookPresetsResponse,
  StaticNotebookRunSpec,
} from "./notebook/types";

/** @deprecated Display-only snapshot; runtime fetches use getApiBase(). */
export const API_BASE = getApiBase();

export interface HealthResponse {
  service: string;
  version: string;
  gpu_available: boolean;
  gpu_name: string | null;
  active_jobs: number;
}

export interface SplatInfo {
  job_id: string;
  scene: string;
  num_frames: number;
  frame_urls: string[];
  total_size_bytes: number;
  status: string;
  source?: "registry" | "disk";
}

export type JobStatus = "queued" | "running" | "completed" | "failed";

export interface Job {
  id: string;
  scene: string;
  status: JobStatus;
  smoke_test: boolean;
  phase: {
    name: string;
    progress: number;
    message: string;
    details: Record<string, unknown>;
  };
  overall_progress: number;
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
  error: string | null;
  ply_dir: string | null;
  download_url: string | null;
}

export interface SceneListItem {
  name: string;
  num_frames: number;
  total_size_bytes: number;
  modified_ts: number;
}

export interface SceneListResponse {
  scenes: SceneListItem[];
  total: number;
}

/**
 * Tüm opsiyonel hyperparam override'ları. Hepsi null olabilir
 * (null = backend default'u kullanır).
 */
export interface HyperParams {
  // Temel
  iters?: number | null;
  resolution?: string | null; // "640x360"
  num_timestamps?: number | null;
  fps?: number | null;
  // Loss weights
  lambda_ssim?: number | null;
  lambda_deform_reg?: number | null;
  lambda_smoothness?: number | null;
  lambda_rigidity?: number | null;
  lambda_depth?: number | null;
  lambda_mask_motion?: number | null;
  lambda_track?: number | null;
  lambda_scale?: number | null;           // v3 — scale regularizer (outlier blow-up önleme)
  opacity_reset_interval?: number | null; // v3 — INRIA-style opacity reset
  track_sample_k?: number | null;
  warmup_iters?: number | null;
  // Learning rates
  lr_deform?: number | null;
  lr_means?: number | null;
  // Density control
  density_start_iter?: number | null;
  density_end_iter?: number | null;
  density_interval?: number | null;
  densify_grad_threshold?: number | null;
  prune_min_opacity?: number | null;
  prune_max_scale?: number | null;
  max_gaussians?: number | null;
  // Model
  sh_degree?: number | null;
  hexplane_resolution?: number | null;
  hexplane_feat_dim?: number | null;
  mlp_width?: number | null;
  mlp_depth?: number | null;
  num_time_freqs?: number | null;
  // v3.6 / Yol C — Per-gaussian Fourier trajectory
  deform_pos_mode?: string | null;        // "mlp" | "fourier" | "hybrid"
  fourier_K?: number | null;               // Frekans sayısı
  lr_fourier?: number | null;
  lambda_fourier_reg?: number | null;
  // Foundation models
  metric3d_model?: string | null;
  cotracker_num_points?: number | null;
  cotracker_grid_size?: number | null;
  sam2_threshold?: number | null;
  // v3.8 — Anti-streak
  lambda_aniso?: number | null;
  aniso_threshold?: number | null;
  dpos_total_cap_frac?: number | null;
  // v3.9 — Preprocessing
  resize_long_edge?: number | null;
  colmap_matching?: string | null;        // "sequential" | "exhaustive"
  init_subsample_mode?: string | null;    // "random" | "confidence"
  // v6.1 — 4D Quality knobs
  sh_progressive_schedule?: number | null;  // 0|1 — backend bool olarak parse eder
  lambda_accel?: number | null;
  cam_grad_clip_norm?: number | null;
  mip_scale_floor_frac?: number | null;
  dynamic_densify_scale?: number | null;
  // v6.1 — Sparse-view init method
  init_method?: string | null;            // "colmap" | "dust3r" | "auto"
}

/**
 * Pipeline mode — Static 3DGS (foto/sparse-view) vs 4D Dynamic (video).
 * v6.0 frontend cleanup: tek alan, eski boolean preset flag'leri yerine
 * mode + preset string ile gönderim.
 */
export type JobMode = "static" | "dynamic";

/** Static 3DGS preset'leri — scripts/static_3dgs.py PRESETS ile birebir. */
export type StaticPreset = "fast" | "balanced" | "high" | "premium" | "sota" | "ultra";

export interface SubmitJobOptions {
  scene: string;
  /** v6.0: zorunlu — backend bu alana göre static_mode set eder. */
  mode: JobMode;
  /** v6.0: mode'a uygun preset string. Backend tanımıyorsa default'a düşer. */
  preset: string;
  skip_foundation?: boolean;
  /** v6.1 — NVS evaluation enabled (held-out cam metrics + orbit mp4). */
  nvs_eval?: boolean;
  hyperparams?: HyperParams;
}

export interface NvsEvalReport {
  available: boolean;
  scene?: string;
  type?: "mv" | "sv";
  held_out_cam?: string;
  held_out_metrics?: {
    psnr: number;
    ssim: number;
    lpips: number;
    n_frames: number;
  };
  temporal_holdout?: {
    psnr: number;
    ssim: number;
    lpips: number;
    n_frames: number;
  };
  orbit?: {
    success: boolean;
    n_frames: number;
    path?: string;
    error?: string;
  };
  orbit_url?: string | null;
  message?: string;
}

export interface ProcessResponse {
  job_id: string;
  status: JobStatus;
  status_url: string;
  message: string;
}

/**
 * Build a full URL from a path-relative endpoint, adding the configured
 * base + auth header. Always read getApiBase()/getAuthToken() *per call*
 * so a settings change takes effect immediately.
 */
function authHeaders(extra?: HeadersInit): Headers {
  const headers = new Headers(extra);
  const tok = getAuthToken();
  if (tok && !headers.has("Authorization")) {
    headers.set("Authorization", `Bearer ${tok}`);
  }
  return headers;
}

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const url = path.startsWith("http") ? path : `${getApiBase()}${path}`;
  const res = await fetch(url, { ...init, headers: authHeaders(init?.headers) });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`HTTP ${res.status} ${res.statusText}: ${text || url}`);
  }
  return res.json() as Promise<T>;
}

export async function getHealth(): Promise<HealthResponse> {
  return fetchJson<HealthResponse>(`/`);
}

export async function listJobs(): Promise<{ jobs: Job[]; total: number }> {
  return fetchJson(`/jobs`);
}

export async function getJobStatus(jobId: string): Promise<Job> {
  return fetchJson<Job>(`/status/${jobId}`);
}

export async function cancelJob(jobId: string): Promise<{ ok: boolean; message: string }> {
  const res = await fetch(`${getApiBase()}/cancel/${jobId}`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
  });
  if (!res.ok) {
    throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  }
  return res.json();
}

export async function listDiskScenes(): Promise<SceneListResponse> {
  return fetchJson<SceneListResponse>(`/scenes`);
}

export async function getSplatInfo(identifier: string): Promise<SplatInfo> {
  return fetchJson<SplatInfo>(`/splat/${identifier}/info`);
}

// ---------------------------------------------------------------------------
// Analytics — training metrics + summary
// ---------------------------------------------------------------------------
export interface TrainMetric {
  t: number;              // seconds since run start
  iter: number;
  n_iters: number;
  loss: number;
  psnr: number;
  n_points: number;
  dpos_mean: number;
  dpos_max: number;
  warmup: number;
  it_per_sec: number;
  recon: number;
  depth: number;
  track: number;
  deform_reg: number;
  smooth: number;
  rigid: number;
  scale: number;
}

export interface MetricsResponse {
  metrics: TrainMetric[];
  count: number;
  note?: string;
}

export interface EventsResponse {
  events: string[];
  count: number;
}

export async function getJobMetrics(identifier: string): Promise<MetricsResponse> {
  return fetchJson<MetricsResponse>(`/jobs/${identifier}/metrics`);
}

export async function getJobSummary(identifier: string): Promise<Record<string, unknown>> {
  return fetchJson(`/jobs/${identifier}/summary`);
}

export async function getJobEvents(identifier: string, tail?: number): Promise<EventsResponse> {
  const q = tail ? `?tail=${tail}` : "";
  return fetchJson<EventsResponse>(`/jobs/${identifier}/events${q}`);
}

// v6.1 — NVS Evaluation
export async function getJobEval(identifier: string): Promise<NvsEvalReport> {
  return fetchJson<NvsEvalReport>(`/jobs/${identifier}/eval`);
}

// URL helpers — these are loaded by <video src> / splat library's loader,
// neither of which can attach our Authorization header. When a token is set
// (cloud), withTokenParam appends ?token= as the backend's accepted fallback.
export function orbitVideoUrl(identifier: string): string {
  return withTokenParam(`${getApiBase()}/jobs/${identifier}/orbit.mp4`);
}

export function frameUrl(identifier: string, idx: number): string {
  return withTokenParam(`${getApiBase()}/splat/${identifier}/frame/${idx}`);
}

export function downloadUrl(jobId: string): string {
  return withTokenParam(`${getApiBase()}/download/${jobId}`);
}

/**
 * Video dosyası + params -> job submit.
 * Backend FormData bekler (multipart upload).
 */
export async function submitJob(
  videoFile: File | null,
  options: SubmitJobOptions,
): Promise<ProcessResponse> {
  const fd = new FormData();
  // Static modda video opsiyonel (photo set scenario). Backend kendi tarafinda
  // sahne klasorunde images/ varsa video.mp4 yoksa hata firlatmiyor.
  if (videoFile) fd.append("video", videoFile);
  fd.append("scene", options.scene);
  // v6.0: mode + preset (single source of truth)
  fd.append("mode", options.mode);
  fd.append("preset", options.preset);
  if (options.skip_foundation !== undefined)
    fd.append("skip_foundation", String(options.skip_foundation));
  if (options.nvs_eval !== undefined)
    fd.append("nvs_eval", String(options.nvs_eval));

  // Hyperparams — sadece null/undefined olmayanları gönder
  const overrideKeys: string[] = [];
  if (options.hyperparams) {
    for (const [key, value] of Object.entries(options.hyperparams)) {
      if (value !== null && value !== undefined && value !== "") {
        fd.append(key, String(value));
        overrideKeys.push(`${key}=${value}`);
      }
    }
  }

  // Diagnostic — without this, intermittent submit failures are unsolvable.
  // Open browser devtools (Tauri: right-click → Inspect Element) to see these.
  console.log("[submitJob] target:", `${getApiBase()}/process`);
  console.log("[submitJob] scene:", options.scene, "mode:", options.mode, "preset:", options.preset);
  console.log("[submitJob] video file:", videoFile ? `${videoFile.name} (${videoFile.size} bytes)` : "<none>");
  console.log("[submitJob] hyperparam overrides:", overrideKeys.length === 0 ? "<none>" : overrideKeys);

  const res = await fetch(`${getApiBase()}/process`, {
    method: "POST",
    body: fd,
    headers: authHeaders(),
  }).catch((err) => {
    console.error("[submitJob] fetch threw:", err);
    throw new Error(`Network error: ${err}. Check connection settings + backend log.`);
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    console.error("[submitJob] HTTP error:", res.status, res.statusText, text);
    throw new Error(`HTTP ${res.status}: ${text || res.statusText}`);
  }
  const json = (await res.json()) as ProcessResponse;
  console.log("[submitJob] success:", json);
  return json;
}

export async function getStaticNotebookPresets(): Promise<StaticNotebookPresetsResponse> {
  return fetchJson<StaticNotebookPresetsResponse>("/notebooks/static/presets");
}

function sanitizeAttachmentFilename(value: string): string | null {
  const cleaned = value
    .replace(/[\\/]+/g, "_")
    .replace(/[\u0000-\u001f\u007f]/g, "")
    .trim();
  return cleaned && cleaned !== "." && cleaned !== ".." ? cleaned : null;
}

export function parseAttachmentFilename(header: string | null): string | null {
  if (!header) return null;
  const extended = /filename\*\s*=\s*([^;]+)/i.exec(header);
  if (extended) {
    try {
      const encoded = extended[1].trim().replace(/^"|"$/g, "");
      const rfc5987 = /^[^']*'[^']*'(.*)$/.exec(encoded);
      const value = decodeURIComponent(rfc5987?.[1] ?? encoded);
      const safe = sanitizeAttachmentFilename(value);
      if (safe) return safe;
    } catch {
      // Fall through to the plain filename parameter.
    }
  }
  const plain = /(?:^|;)\s*filename\s*=\s*("(?:[^"\\]|\\.)*"|[^;]+)/i.exec(
    header,
  );
  if (!plain) return null;
  let value = plain[1].trim();
  if (value.startsWith('"') && value.endsWith('"')) {
    value = value.slice(1, -1).replace(/\\(["\\])/g, "$1");
  }
  return sanitizeAttachmentFilename(value);
}

export async function generateStaticNotebook(
  spec: StaticNotebookRunSpec,
): Promise<GeneratedNotebook> {
  const response = await fetch(`${getApiBase()}/notebooks/static`, {
    method: "POST",
    headers: authHeaders({
      "Content-Type": "application/json",
      Accept: "application/x-ipynb+json",
    }),
    body: JSON.stringify(spec),
  });
  if (!response.ok) {
    const body = await response.text().catch(() => "");
    throw new Error(
      `HTTP ${response.status} ${response.statusText}: ${body || response.url}`,
    );
  }
  return {
    blob: await response.blob(),
    filename:
      parseAttachmentFilename(response.headers.get("Content-Disposition")) ??
      "static_splat.ipynb",
  };
}
