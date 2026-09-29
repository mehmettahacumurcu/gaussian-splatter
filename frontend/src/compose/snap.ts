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

/**
 * Centre of the visible content in the object's local frame (splat bounds ∩
 * crop box), or null before the splat has loaded. The pivot for "Dikleştir".
 */
export function contentPivot(obj: SceneObject, entry: RegistryEntry): Vec3 | null {
  const box = localBoundsWithCrop(obj, entry);
  if (!box) return null;
  const c = box.getCenter(new THREE.Vector3());
  return [c.x, c.y, c.z];
}

/** Calls `visit` with every world-space content point (reused vector: copy to keep). */
type PointSource = (visit: (p: THREE.Vector3) => void) => void;

/**
 * Meshes: every vertex through its mesh's world matrix. Exact (a rotated
 * mesh's AABB would leave it floating) and O(vertices) once per click, which
 * is cheap next to a raycast. Skinning / morph targets are ignored.
 */
function meshPoints(entry: RegistryEntry): PointSource {
  return (visit) => {
    const v = new THREE.Vector3();
    entry.group.traverse((node) => {
      const mesh = node as THREE.Mesh;
      const pos = mesh.isMesh ? mesh.geometry?.getAttribute("position") : undefined;
      if (!pos) return;
      for (let i = 0; i < pos.count; i++) visit(v.fromBufferAttribute(pos, i).applyMatrix4(mesh.matrixWorld));
    });
  };
}

/** Splats: the centres kept by the crop (tested in the local/raw frame, like the bake), in world space. */
function splatPoints(obj: SceneObject, entry: RegistryEntry): PointSource | null {
  const splat = entry.splat;
  if (!splat) return null;
  const crop = obj.crop ?? null;
  // Same test as isInsideCrop, without per-splat allocations (runs per centre).
  const inv = crop ? new THREE.Quaternion(...crop.quaternion).normalize().invert() : null;
  const cropCenter = crop ? new THREE.Vector3(...crop.center) : null;
  return (visit) => {
    const m = splat.matrixWorld;
    const v = new THREE.Vector3();
    const l = new THREE.Vector3();
    splat.forEachSplat((_i, center) => {
      if (crop && inv && cropCenter) {
        l.copy(center).sub(cropCenter).applyQuaternion(inv);
        const h = crop.halfSize;
        if (Math.abs(l.x) > h[0] || Math.abs(l.y) > h[1] || Math.abs(l.z) > h[2]) return;
      }
      visit(v.copy(center).applyMatrix4(m));
    });
  };
}

interface UpExtent {
  /** Min / max projection of the content on up. */
  lo: number;
  hi: number;
  /** Centre of the content's world AABB (horizontal position of the drop ray). */
  center: THREE.Vector3;
}

/** Extent of the object's actual content along `up`, or null when it has none (yet). */
export function contentExtentAlongUp(obj: SceneObject, entry: RegistryEntry, up: THREE.Vector3): UpExtent | null {
  entry.group.updateWorldMatrix(true, true);
  const source = obj.kind === "mesh" ? meshPoints(entry) : splatPoints(obj, entry);
  if (!source) return null;
  let lo = Infinity;
  let hi = -Infinity;
  const box = new THREE.Box3();
  try {
    source((p) => {
      const d = p.dot(up);
      if (d < lo) lo = d;
      if (d > hi) hi = d;
      box.expandByPoint(p);
    });
  } catch {
    return null; // splat data not loaded
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return null;
  return { lo, hi, center: box.getCenter(new THREE.Vector3()) };
}

/**
 * New position that drops ``id`` onto the first visible surface below it
 * (along −up, `effectiveUp(doc)`, any direction), or null when nothing is
 * hit. The object's lowest content point along up lands on the hit. Hits on
 * cropped-away splats never reach us: SplatObject's raycast override already
 * drops them.
 */
export function snapToGround(doc: SceneDoc, registry: ObjectRegistry, id: string): Vec3 | null {
  const obj = doc.objects.find((o) => o.id === id);
  const entry = registry.get(id);
  if (!obj || !entry || obj.role === "base") return null;

  const up = new THREE.Vector3(...effectiveUp(doc));
  const extent = contentExtentAlongUp(obj, entry, up);
  if (!extent) return null;
  const { lo, hi, center } = extent;
  const bottom = center.clone().addScaledVector(up, lo - center.dot(up));
  const origin = bottom.clone().addScaledVector(up, (hi - lo) / 2);

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
