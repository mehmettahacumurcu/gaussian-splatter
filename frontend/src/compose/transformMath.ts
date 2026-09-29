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

/** TransformControls scales one axis at a time; follow the axis that changed most. */
export function uniformScaleFrom(prev: number, s: { x: number; y: number; z: number }): number {
  let best = s.x;
  for (const c of [s.y, s.z]) if (Math.abs(c - prev) > Math.abs(best - prev)) best = c;
  return Math.max(Math.abs(best), 1e-4);
}

export function readTransform(obj: Object3D): Transform {
  const q = obj.quaternion;
  return {
    position: [obj.position.x, obj.position.y, obj.position.z],
    quaternion: [q.x, q.y, q.z, q.w],
    scale: obj.scale.x,
  };
}

export function readCrop(obj: Object3D): CropBox {
  const q = obj.quaternion;
  const h = (v: number) => Math.max(Math.abs(v), 1e-3);
  return {
    center: [obj.position.x, obj.position.y, obj.position.z],
    halfSize: [h(obj.scale.x), h(obj.scale.y), h(obj.scale.z)],
    quaternion: [q.x, q.y, q.z, q.w],
  };
}
