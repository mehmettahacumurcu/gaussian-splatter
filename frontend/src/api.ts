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

export interface Job {
  id: string;
  scene: string;
  status: "queued" | "running" | "completed" | "failed";
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

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
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

export async function listDiskScenes(): Promise<SceneListResponse> {
  return fetchJson<SceneListResponse>(`${API_BASE}/scenes`);
}

export async function getSplatInfo(identifier: string): Promise<SplatInfo> {
  return fetchJson<SplatInfo>(`${API_BASE}/splat/${identifier}/info`);
}

export function frameUrl(identifier: string, idx: number): string {
  return `${API_BASE}/splat/${identifier}/frame/${idx}`;
}
