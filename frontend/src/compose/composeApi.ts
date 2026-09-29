import { fetchJson } from "../api";
import { getApiBase, withTokenParam } from "../connection";
import type { Asset, SceneDoc, SceneSummary } from "./types";

const JSON_HEADERS = { "Content-Type": "application/json" };
const enc = encodeURIComponent;

export function listAssets(): Promise<Asset[]> {
  return fetchJson<Asset[]>("/compose/assets");
}

export function uploadAsset(file: File): Promise<Asset> {
  const form = new FormData();
  form.append("file", file);
  return fetchJson<Asset>("/compose/assets", { method: "POST", body: form });
}

export function assetFileUrl(assetId: string): string {
  return withTokenParam(`${getApiBase()}/compose/assets/${enc(assetId)}/file`);
}

export function listScenes(): Promise<SceneSummary[]> {
  return fetchJson<SceneSummary[]>("/compose/scenes");
}

export function createScene(name: string, baseAsset: string): Promise<SceneDoc> {
  return fetchJson<SceneDoc>("/compose/scenes", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({ name, base_asset: baseAsset }),
  });
}

export function getScene(id: string): Promise<SceneDoc> {
  return fetchJson<SceneDoc>(`/compose/scenes/${enc(id)}`);
}

export function saveScene(doc: SceneDoc): Promise<SceneDoc> {
  return fetchJson<SceneDoc>(`/compose/scenes/${enc(doc.id)}`, {
    method: "PUT",
    headers: JSON_HEADERS,
    body: JSON.stringify(doc),
  });
}

export function exportScene(id: string): Promise<{ job_id: string }> {
  return fetchJson<{ job_id: string }>(`/compose/scenes/${enc(id)}/export`, { method: "POST" });
}

export function backendUrl(path: string): string {
  return withTokenParam(`${getApiBase()}${path}`);
}
