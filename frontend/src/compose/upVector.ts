/** Scene "up" helpers (pure, no three.js). Up is view-only: camera, snap, mesh insert. */
import type { SceneDoc, Vec3 } from "./types";

const MIN_NORM = 1e-9;

/** Unit copy of `v`, or null when it is non-finite or (near) zero. */
export function normalizeUp(v: readonly number[] | null | undefined): Vec3 | null {
  if (!v || v.length !== 3 || !v.every(Number.isFinite)) return null;
  const n = Math.hypot(v[0], v[1], v[2]);
  if (!Number.isFinite(n) || n < MIN_NORM) return null;
  return [v[0] / n, v[1] / n, v[2] / n];
}

/** `doc.up` when set (and usable), else ±Y from `viewUp`. */
export function effectiveUp(doc: Pick<SceneDoc, "up" | "viewUp">): Vec3 {
  const up = doc.up ? normalizeUp(doc.up) : null;
  if (up) return up;
  return doc.viewUp === "-y" ? [0, -1, 0] : [0, 1, 0];
}

/** Stable string for keys / comparisons (6 decimals). */
export function upKey(up: Vec3): string {
  return up.map((c) => (Object.is(c, -0) ? 0 : c).toFixed(6)).join(",");
}
