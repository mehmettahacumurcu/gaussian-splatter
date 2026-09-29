import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { ObjectRegistry } from "../registry";
import type { SplatMesh } from "@sparkjsdev/spark";
import { contentPivot, snapToGround } from "../snap";
import { isInsideCrop, meshInsertQuaternion } from "../transformMath";
import type { CropBox, SceneDoc, SceneObject, Transform, Vec3 } from "../types";

function unit(v: Vec3): Vec3 {
  const n = Math.hypot(...v);
  return [v[0] / n, v[1] / n, v[2] / n];
}

function meshObject(id: string, position: Vec3, visible = true): SceneObject {
  return {
    id,
    kind: "mesh",
    asset: `${id}.glb`,
    name: id,
    role: "object",
    visible,
    transform: { position, quaternion: [0, 0, 0, 1], scale: 1 },
  };
}

/** A big floor plane through `point` whose normal is `up`, plus a 1 m box `height` above it. */
function setup(up: Vec3, point: Vec3, opts: { floorVisible?: boolean; viewUp?: "y" | "-y"; useUp?: boolean } = {}) {
  const upV = new THREE.Vector3(...up);
  const registry = new ObjectRegistry();

  const floorGroup = new THREE.Group();
  const plane = new THREE.Mesh(new THREE.PlaneGeometry(50, 50), new THREE.MeshBasicMaterial({ side: THREE.DoubleSide }));
  plane.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), upV);
  floorGroup.add(plane);
  floorGroup.position.set(...point);
  registry.set("floor", { group: floorGroup });

  // Offset sideways (perpendicular to up) so the box isn't centred over the plane origin.
  const side = new THREE.Vector3(1, 0, 0).cross(upV).normalize();
  const boxPos = new THREE.Vector3(...point).addScaledVector(upV, 3).addScaledVector(side, 0.7);
  const boxGroup = new THREE.Group();
  boxGroup.add(new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial()));
  boxGroup.position.copy(boxPos);
  registry.set("box", { group: boxGroup });

  const doc: SceneDoc = {
    version: 1,
    id: "s",
    name: "s",
    viewUp: opts.viewUp ?? "y",
    up: opts.useUp === false ? null : up,
    objects: [
      meshObject("floor", point, opts.floorVisible ?? true),
      meshObject("box", [boxPos.x, boxPos.y, boxPos.z]),
    ],
  };
  return { doc, registry, boxGroup, upV };
}

/** Signed distance of the box's lowest world corner above the plane. */
function bottomGap(boxGroup: THREE.Group, upV: THREE.Vector3, point: Vec3): number {
  boxGroup.updateWorldMatrix(true, true);
  const box = new THREE.Box3().setFromObject(boxGroup);
  let min = Infinity;
  for (const x of [box.min.x, box.max.x])
    for (const y of [box.min.y, box.max.y])
      for (const z of [box.min.z, box.max.z]) {
        min = Math.min(min, new THREE.Vector3(x, y, z).sub(new THREE.Vector3(...point)).dot(upV));
      }
  return min;
}

describe("snapToGround", () => {
  it("drops an object onto a floor perpendicular to a tilted up", () => {
    const up = unit([-0.94, -0.11, -0.32]);
    const point: Vec3 = [0.5, -1, 2];
    const { doc, registry, boxGroup, upV } = setup(up, point);
    expect(bottomGap(boxGroup, upV, point)).toBeGreaterThan(1);

    const p = snapToGround(doc, registry, "box");
    expect(p).not.toBeNull();
    boxGroup.position.set(...p!);
    expect(bottomGap(boxGroup, upV, point)).toBeCloseTo(0, 5);
    // Moved only along up.
    const moved = new THREE.Vector3(...p!).sub(new THREE.Vector3(...doc.objects[1].transform.position));
    expect(moved.clone().cross(upV).length()).toBeLessThan(1e-6);
  });

  it("still works for the ±Y fallback (no explicit up)", () => {
    const { doc, registry, boxGroup, upV } = setup([0, -1, 0], [0, 2, 0], { viewUp: "-y", useUp: false });
    const p = snapToGround(doc, registry, "box");
    expect(p).not.toBeNull();
    boxGroup.position.set(...p!);
    expect(bottomGap(boxGroup, upV, [0, 2, 0])).toBeCloseTo(0, 5);
  });

  it("ignores hidden objects", () => {
    const { doc, registry } = setup(unit([0.2, 0.9, -0.3]), [0, 0, 0], { floorVisible: false });
    expect(snapToGround(doc, registry, "box")).toBeNull();
  });
  it("uses the true mesh extent along up, not its world AABB", () => {
    const up = unit([-0.94, -0.11, -0.32]);
    const point: Vec3 = [0.5, -1, 2];
    const { doc, registry, upV } = setup(up, point);
    // Unit cube inserted like a glTF mesh: rotated so its +Y matches up.
    const q = meshInsertQuaternion(up);
    const cubeMesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial());
    const start = new THREE.Vector3(...point).addScaledVector(upV, 4);
    const t: Transform = { position: [start.x, start.y, start.z], quaternion: q, scale: 1 };
    const group = placed(cubeMesh, t);
    registry.set("cube", { group });
    const withCube: SceneDoc = { ...doc, objects: [...doc.objects, { ...meshObject("cube", t.position), transform: t }] };

    const p = snapToGround(withCube, registry, "cube");
    expect(p).not.toBeNull();
    group.position.set(...p!);
    expect(minVertexGap(group, upV, point)).toBeCloseTo(0, 3);
  });

  it("uses the splat centres inside the crop (world frame) for splats", () => {
    const up = unit([0.3, 0.85, -0.4]);
    const point: Vec3 = [0, -2, 1];
    const { doc, registry, upV } = setup(up, point);
    // Local splat centres: a 1 m cluster plus a far outlier that the crop removes.
    const centres: Vec3[] = [
      [0, 0, 0],
      [0.5, 0.2, -0.3],
      [-0.4, -0.5, 0.1],
      [0.2, 0.5, 0.4],
      [0, -20, 0],
    ];
    const crop: CropBox = { center: [0, 0, 0], halfSize: [1, 1, 1], quaternion: [0, 0, 0, 1] };
    const stub = new THREE.Object3D() as THREE.Object3D & { forEachSplat: SplatMesh["forEachSplat"] };
    stub.forEachSplat = (cb) => {
      const v = new THREE.Vector3();
      centres.forEach((c, i) => cb(i, v.set(...c), new THREE.Vector3(), new THREE.Quaternion(), 1, new THREE.Color()));
    };
    const rot = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 2, 0.5).normalize(), 0.8);
    const start = new THREE.Vector3(...point).addScaledVector(upV, 5);
    const t: Transform = {
      position: [start.x, start.y, start.z],
      quaternion: [rot.x, rot.y, rot.z, rot.w],
      scale: 1.5,
    };
    const group = placed(stub, t);
    registry.set("splat", { group, splat: stub as unknown as SplatMesh });
    const obj: SceneObject = { ...meshObject("splat", t.position), kind: "splat", transform: t, crop };
    const withSplat: SceneDoc = { ...doc, objects: [...doc.objects, obj] };

    const p = snapToGround(withSplat, registry, "splat");
    expect(p).not.toBeNull();
    group.position.set(...p!);
    group.updateWorldMatrix(true, true);
    const gaps = centres
      .filter((c) => isInsideCrop(crop, c))
      .map((c) => new THREE.Vector3(...c).applyMatrix4(stub.matrixWorld).sub(new THREE.Vector3(...point)).dot(upV));
    expect(Math.min(...gaps)).toBeCloseTo(0, 5);
  });
});

describe("contentPivot", () => {
  it("is the centre of the splat's local bounds intersected with the crop", () => {
    const group = new THREE.Group();
    const bounds = new THREE.Box3(new THREE.Vector3(-1, -1, -1), new THREE.Vector3(1, 1, 1));
    const obj: SceneObject = {
      ...meshObject("s", [0, 0, 0]),
      kind: "splat",
      crop: { center: [0.5, 0, 0], halfSize: [1, 1, 1], quaternion: [0, 0, 0, 1] },
    };
    expect(contentPivot(obj, { group, localBounds: bounds })).toEqual([0.25, 0, 0]);
    expect(contentPivot({ ...obj, crop: null }, { group, localBounds: bounds })).toEqual([0, 0, 0]);
    expect(contentPivot(obj, { group })).toBeNull();
  });
});

function placed(child: THREE.Object3D, t: Transform): THREE.Group {
  const group = new THREE.Group();
  group.add(child);
  group.position.set(...t.position);
  group.quaternion.set(...t.quaternion);
  group.scale.setScalar(t.scale);
  return group;
}

/** Signed distance of the lowest mesh vertex above the plane through `point`. */
function minVertexGap(group: THREE.Group, upV: THREE.Vector3, point: Vec3): number {
  group.updateWorldMatrix(true, true);
  let min = Infinity;
  group.traverse((node) => {
    const mesh = node as THREE.Mesh;
    if (!mesh.isMesh) return;
    const pos = mesh.geometry.getAttribute("position");
    const v = new THREE.Vector3();
    for (let i = 0; i < pos.count; i++) {
      v.fromBufferAttribute(pos, i).applyMatrix4(mesh.matrixWorld);
      min = Math.min(min, v.sub(new THREE.Vector3(...point)).dot(upV));
    }
  });
  return min;
}
