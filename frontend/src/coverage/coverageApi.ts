import { fetchJson } from "../api";
import { getApiBase, withTokenParam } from "../connection";

const enc = encodeURIComponent;

export interface CoverageSummary {
  splats: number;
  opaque_splats: number;
  unseen_share: number;
  weak_share: number;
  count_percentiles: Record<string, number>;
  good_count: number;
  weak_score: number;
}

export interface CoverageMeta {
  id: string;
  name: string;
  created_at: number;
  ply_path: string;
  sparse_dir: string;
  images_dir: string | null;
  cameras: number;
  test_views: number;
  up: [number, number, number];
  seconds: number;
  summary: CoverageSummary;
}

export interface CoverageView {
  index: number;
  name: string;
  test: boolean;
  /** Camera-to-world, row-major 4x4 (OpenCV axes: x right, y down, z forward). */
  c2w: number[];
  fov_y: number;
  aspect: number;
  /** Mean coverage score (0..1) of the splats this camera sees. */
  score: number;
  /** Share of the splats this camera sees that are weakly covered (0..1). */
  weak: number;
}

export interface CoverageDetail extends CoverageMeta {
  views: CoverageView[];
}

export interface CoverageStarted {
  job_id: string;
  coverage_id: string;
  ply_path: string;
  sparse_dir: string;
}

export function listCoverage(): Promise<CoverageMeta[]> {
  return fetchJson<CoverageMeta[]>("/coverage");
}

export function startCoverage(req: { path: string; ply_path?: string; sparse_dir?: string; name?: string }): Promise<CoverageStarted> {
  return fetchJson<CoverageStarted>("/coverage", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
}

export function getCoverage(id: string): Promise<CoverageDetail> {
  return fetchJson<CoverageDetail>(`/coverage/${enc(id)}`);
}

export function deleteCoverage(id: string): Promise<{ ok: boolean }> {
  return fetchJson<{ ok: boolean }>(`/coverage/${enc(id)}`, { method: "DELETE" });
}

export function coverageSplatUrl(id: string, which: "normal" | "heat"): string {
  return withTokenParam(`${getApiBase()}/coverage/${enc(id)}/splat/${which}`);
}

export function coveragePhotoUrl(id: string, index: number): string {
  return withTokenParam(`${getApiBase()}/coverage/${enc(id)}/photo/${index}`);
}
