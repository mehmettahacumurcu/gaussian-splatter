import * as THREE from "three";
import type { ObjectRegistry, RegistryEntry } from "./registry";
import type { SceneDoc, SceneObject, Vec3 } from "./types";

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

/** World-space bounds of an object (cropped for splats). */
export function worldBounds(obj: SceneObject, entry: RegistryEntry): THREE.Box3 | null {
  entry.group.updateWorldMatrix(true, true);
  if (obj.kind === "mesh") {
    const box = new THREE.Box3().setFromObject(entry.group);
    return box.isEmpty() ? null : box;
  }
  const local = localBoundsWithCrop(obj, entry);
  return local ? local.applyMatrix4(entry.group.matrixWorld) : null;
}

/**
 * New position that drops ``id`` onto the first visible surface below it
 * (along −up), or null when nothing is hit. Hits on cropped-away splats never
 * reach us: SplatObject's raycast override already drops them.
 */
export function snapToGround(doc: SceneDoc, registry: ObjectRegistry, id: string): Vec3 | null {
  const obj = doc.objects.find((o) => o.id === id);
  const entry = registry.get(id);
  if (!obj || !entry || obj.role === "base") return null;
  const box = worldBounds(obj, entry);
  if (!box) return null;

  const up = new THREE.Vector3(0, doc.viewUp === "-y" ? -1 : 1, 0);
  const center = box.getCenter(new THREE.Vector3());
  const bottomY = up.y > 0 ? box.min.y : box.max.y;
  const bottom = new THREE.Vector3(center.x, bottomY, center.z);
  const height = box.max.y - box.min.y;
  const origin = bottom.clone().addScaledVector(up, height / 2);

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
