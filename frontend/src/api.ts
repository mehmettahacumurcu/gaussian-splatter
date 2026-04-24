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
  // Foundation models
  metric3d_model?: string | null;
  cotracker_num_points?: number | null;
  cotracker_grid_size?: number | null;
  sam2_threshold?: number | null;
}

export interface SubmitJobOptions {
  scene: string;
  smoke_test?: boolean;
  cloud?: boolean;
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
  if (options.cloud !== undefined)
    fd.append("cloud", String(options.cloud));
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
