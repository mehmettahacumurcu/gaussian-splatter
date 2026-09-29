import * as THREE from "three";
import type { ObjectRegistry, RegistryEntry } from "./registry";
import type { SceneDoc, SceneObject, Vec3 } from "./types";
import { effectiveUp } from "./upVector";

function localBoundsWithCrop(obj: SceneObject, entry: RegistryEntry): THREE.Box3 | null {
  if (!entry.localBounds) return null;
  const box = entry.localBounds.clone();
  if (obj.crop) {
    const c = obj.crop;
    // Unit box (half-extent 1) scaled by halfSize = the crop in the local frame.
    const m = new THREE.Matrix4().compose(
      new THREE.Vector3(...c.center),
      new THREE.Quaternion(...c.quaternion).normalize(),
      new THREE.Vector3(...c.halfSize),
    );
    box.intersect(new THREE.Box3(new THREE.Vector3(-1, -1, -1), new THREE.Vector3(1, 1, 1)).applyMatrix4(m));
  }
  return box.isEmpty() ? null : box;
}

/** Eight corners of `box` (in its own frame), optionally transformed by `m`. */
function boxCorners(box: THREE.Box3, m?: THREE.Matrix4): THREE.Vector3[] {
  const out: THREE.Vector3[] = [];
  for (const x of [box.min.x, box.max.x])
    for (const y of [box.min.y, box.max.y])
      for (const z of [box.min.z, box.max.z]) {
        const v = new THREE.Vector3(x, y, z);
        out.push(m ? v.applyMatrix4(m) : v);
      }
  return out;
}

/**
 * World-space corners of an object's bounds (cropped for splats). Splats use
 * their local box through the world matrix (a tight oriented box); meshes
 * use their world AABB.
 */
export function worldCorners(obj: SceneObject, entry: RegistryEntry): THREE.Vector3[] | null {
  entry.group.updateWorldMatrix(true, true);
  if (obj.kind === "mesh") {
    const box = new THREE.Box3().setFromObject(entry.group);
    return box.isEmpty() ? null : boxCorners(box);
  }
  const local = localBoundsWithCrop(obj, entry);
  return local ? boxCorners(local, entry.group.matrixWorld) : null;
}

/** World-space bounds of an object (cropped for splats). */
export function worldBounds(obj: SceneObject, entry: RegistryEntry): THREE.Box3 | null {
  const corners = worldCorners(obj, entry);
  return corners ? new THREE.Box3().setFromPoints(corners) : null;
}

/**
 * New position that drops ``id`` onto the first visible surface below it
 * (along −up, `effectiveUp(doc)`, any direction), or null when nothing is
 * hit. Hits on cropped-away splats never reach us: SplatObject's raycast
 * override already drops them.
 */
export function snapToGround(doc: SceneDoc, registry: ObjectRegistry, id: string): Vec3 | null {
  const obj = doc.objects.find((o) => o.id === id);
  const entry = registry.get(id);
  if (!obj || !entry || obj.role === "base") return null;
  const corners = worldCorners(obj, entry);
  if (!corners) return null;

  const up = new THREE.Vector3(...effectiveUp(doc));
  // Centre of the (symmetric) corner set; extent of the corners along up.
  const center = corners.reduce((acc, c) => acc.add(c), new THREE.Vector3()).divideScalar(corners.length);
  let lo = Infinity;
  let hi = -Infinity;
  for (const c of corners) {
    const d = c.dot(up);
    lo = Math.min(lo, d);
    hi = Math.max(hi, d);
  }
  const half = (hi - lo) / 2;
  const bottom = center.clone().addScaledVector(up, lo - center.dot(up));
  const origin = bottom.clone().addScaledVector(up, half);

  const targets: THREE.Object3D[] = [];
  for (const [otherId, other] of registry.all()) {
    if (otherId === id) continue;
    const otherObj = doc.objects.find((o) => o.id === otherId);
    if (!otherObj || !otherObj.visible) continue;
    other.group.updateWorldMatrix(true, true);
    targets.push(other.group);
  }

  const raycaster = new THREE.Raycaster(origin, up.clone().negate());
  const hit = raycaster.intersectObjects(targets, true)[0];
  if (!hit) return null;
  const delta = hit.point.clone().sub(bottom).dot(up);
  const p = obj.transform.position;
  return [p[0] + up.x * delta, p[1] + up.y * delta, p[2] + up.z * delta];
}
