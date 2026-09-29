import { Matrix4, Object3D, Quaternion, Vector3 } from "three";
import type { CropBox, Transform, Vec3 } from "./types";

/** Same order as backend bake: scale → rotate → translate. */
export function transformMatrix(t: Transform): Matrix4 {
  return new Matrix4().compose(
    new Vector3(...t.position),
    new Quaternion(...t.quaternion).normalize(),
    new Vector3(t.scale, t.scale, t.scale),
  );
}

export function applyTransform(t: Transform, p: Vec3): Vec3 {
  const v = new Vector3(...p).applyMatrix4(transformMatrix(t));
  return [v.x, v.y, v.z];
}

/** Crop test in the object's local frame: |R_cᵀ (p − c)| ≤ halfSize (matches bake.crop_mask). */
export function isInsideCrop(crop: CropBox, localPoint: Vec3): boolean {
  const inv = new Quaternion(...crop.quaternion).normalize().invert();
  const v = new Vector3(...localPoint).sub(new Vector3(...crop.center)).applyQuaternion(inv);
  return (
    Math.abs(v.x) <= crop.halfSize[0] && Math.abs(v.y) <= crop.halfSize[1] && Math.abs(v.z) <= crop.halfSize[2]
  );
}

/**
 * Half-extent of the crop helper box. The helper is a box of half-extent 1
 * (e.g. BoxGeometry(2,2,2) / a SplatEditSdf BOX whose scale is the half size),
 * parented under the splat mesh so its position/quaternion are object-local;
 * hence `scale × CROP_BOX_HALF_EXTENT = halfSize`.
 */
export const CROP_BOX_HALF_EXTENT = 1;

/** TransformControls scales one axis at a time; follow the axis that changed most. */
export function uniformScaleFrom(prev: number, s: { x: number; y: number; z: number }): number {
  let best: number | null = null;
  for (const c of [s.x, s.y, s.z]) {
    if (!Number.isFinite(c)) continue;
    const a = Math.abs(c);
    if (best === null || Math.abs(a - prev) > Math.abs(best - prev)) best = a;
  }
  return Math.max(best ?? prev, 1e-4);
}

function allFinite(values: number[]): boolean {
  return values.every(Number.isFinite);
}

export function readTransform(obj: Object3D, prevScale: number): Transform {
  const p = obj.position;
  const q = obj.quaternion.clone().normalize();
  if (!allFinite([p.x, p.y, p.z, q.x, q.y, q.z, q.w])) throw new Error("Non-finite transform");
  return {
    position: [p.x, p.y, p.z],
    quaternion: [q.x, q.y, q.z, q.w],
    scale: uniformScaleFrom(prevScale, obj.scale),
  };
}

export function readCrop(obj: Object3D): CropBox {
  const p = obj.position;
  const s = obj.scale;
  const q = obj.quaternion.clone().normalize();
  if (!allFinite([p.x, p.y, p.z, s.x, s.y, s.z, q.x, q.y, q.z, q.w])) throw new Error("Non-finite crop");
  const h = (v: number) => Math.max(Math.abs(v) * CROP_BOX_HALF_EXTENT, 1e-3);
  return {
    center: [p.x, p.y, p.z],
    halfSize: [h(s.x), h(s.y), h(s.z)],
    quaternion: [q.x, q.y, q.z, q.w],
  };
}

/** True when every component is finite, scale > 0 and the quaternion is (nearly) unit length. */
export function isValidTransform(t: Transform): boolean {
  if (!allFinite([...t.position, ...t.quaternion, t.scale])) return false;
  if (t.scale <= 0) return false;
  const norm = Math.hypot(...t.quaternion);
  return Math.abs(norm - 1) <= 1e-3;
}
