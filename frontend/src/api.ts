/**
 * FastAPI backend ile HTTP iletişimi.
 * Backend default olarak http://127.0.0.1:8000 üzerinde çalışır.
 */

export const API_BASE = "http://127.0.0.1:8000";

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
}

export interface SubmitJobOptions {
  scene: string;
  smoke_test?: boolean;
  micro_test?: boolean;
  cloud?: boolean;
  high_test?: boolean;
  ultra_test?: boolean;
  skip_foundation?: boolean;
  hyperparams?: HyperParams;
}

export interface ProcessResponse {
  job_id: string;
  status: JobStatus;
  status_url: string;
  message: string;
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`HTTP ${res.status} ${res.statusText}: ${text || url}`);
  }
  return res.json() as Promise<T>;
}

export async function getHealth(): Promise<HealthResponse> {
  return fetchJson<HealthResponse>(`${API_BASE}/`);
}

export async function listJobs(): Promise<{ jobs: Job[]; total: number }> {
  return fetchJson(`${API_BASE}/jobs`);
}

export async function getJobStatus(jobId: string): Promise<Job> {
  return fetchJson<Job>(`${API_BASE}/status/${jobId}`);
}

export async function listDiskScenes(): Promise<SceneListResponse> {
  return fetchJson<SceneListResponse>(`${API_BASE}/scenes`);
}

export async function getSplatInfo(identifier: string): Promise<SplatInfo> {
  return fetchJson<SplatInfo>(`${API_BASE}/splat/${identifier}/info`);
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
  return fetchJson<MetricsResponse>(`${API_BASE}/jobs/${identifier}/metrics`);
}

export async function getJobSummary(identifier: string): Promise<Record<string, unknown>> {
  return fetchJson(`${API_BASE}/jobs/${identifier}/summary`);
}

export async function getJobEvents(identifier: string, tail?: number): Promise<EventsResponse> {
  const q = tail ? `?tail=${tail}` : "";
  return fetchJson<EventsResponse>(`${API_BASE}/jobs/${identifier}/events${q}`);
}

export function frameUrl(identifier: string, idx: number): string {
  return `${API_BASE}/splat/${identifier}/frame/${idx}`;
}

/**
 * Video dosyası + params -> job submit.
 * Backend FormData bekler (multipart upload).
 */
export async function submitJob(
  videoFile: File,
  options: SubmitJobOptions,
): Promise<ProcessResponse> {
  const fd = new FormData();
  fd.append("video", videoFile);
  fd.append("scene", options.scene);
  if (options.smoke_test !== undefined)
    fd.append("smoke_test", String(options.smoke_test));
  if (options.micro_test !== undefined)
    fd.append("micro_test", String(options.micro_test));
  if (options.cloud !== undefined)
    fd.append("cloud", String(options.cloud));
  if (options.high_test !== undefined)
    fd.append("high_test", String(options.high_test));
  if (options.ultra_test !== undefined)
    fd.append("ultra_test", String(options.ultra_test));
  if (options.skip_foundation !== undefined)
    fd.append("skip_foundation", String(options.skip_foundation));

  // Hyperparams — sadece null/undefined olmayanları gönder
  if (options.hyperparams) {
    for (const [key, value] of Object.entries(options.hyperparams)) {
      if (value !== null && value !== undefined && value !== "") {
        fd.append(key, String(value));
      }
    }
  }

    const res = await fetch(`${API_BASE}/process`, {
    method: "POST",
    body: fd,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`HTTP ${res.status}: ${text || res.statusText}`);
  }
  return res.json() as Promise<ProcessResponse>;
}
