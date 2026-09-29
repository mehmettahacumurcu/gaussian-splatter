import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { ObjectRegistry } from "../registry";
import { snapToGround } from "../snap";
import type { SceneDoc, SceneObject, Vec3 } from "../types";

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
});
