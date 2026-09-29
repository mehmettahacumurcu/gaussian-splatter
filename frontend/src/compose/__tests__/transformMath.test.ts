import { Object3D, Quaternion, Vector3 } from "three";
import { describe, expect, it } from "vitest";
import {
  CROP_BOX_HALF_EXTENT,
  isValidTransform,
  meshInsertQuaternion,
  orbitPositionForUp,
  readCrop,
  readTransform,
  straightenQuaternion,
  uniformScaleFrom,
} from "../transformMath";

describe("uniformScaleFrom guards", () => {
  it("ignores non-finite components", () => {
    expect(uniformScaleFrom(1, { x: Infinity, y: 1, z: 1 })).toBe(1);
    expect(uniformScaleFrom(1, { x: NaN, y: 2, z: 1 })).toBe(2);
  });
  it("returns prev when nothing is finite, and clamps", () => {
    expect(uniformScaleFrom(3, { x: NaN, y: Infinity, z: -Infinity })).toBe(3);
    expect(uniformScaleFrom(1, { x: 0, y: 0, z: 0 })).toBeCloseTo(1e-4);
  });
  it("uses absolute values", () => {
    expect(uniformScaleFrom(1, { x: 1, y: -2.5, z: 1 })).toBe(2.5);
  });
});

describe("readTransform", () => {
  it("follows the changed scale axis", () => {
    const o = new Object3D();
    o.scale.set(1, 2.5, 1);
    expect(readTransform(o, 1).scale).toBe(2.5);
  });
  it("keeps the scale positive", () => {
    const o = new Object3D();
    o.scale.set(-2, 1, 1);
    expect(readTransform(o, 1).scale).toBe(2);
  });
  it("normalises a copy of the quaternion without mutating the object", () => {
    const o = new Object3D();
    o.quaternion.set(0, 0, 0, 2);
    const t = readTransform(o, 1);
    expect(t.quaternion).toEqual([0, 0, 0, 1]);
    expect(o.quaternion.w).toBe(2);
  });
  it("throws on non-finite positions", () => {
    const o = new Object3D();
    o.position.set(NaN, 0, 0);
    expect(() => readTransform(o, 1)).toThrow("Non-finite transform");
  });
});

describe("readCrop", () => {
  it("maps scale to halfSize via the helper extent", () => {
    const o = new Object3D();
    o.position.set(1, 2, 3);
    o.scale.set(2, -3, 0);
    o.quaternion.set(0, 0, 0, 2);
    const c = readCrop(o);
    expect(c.center).toEqual([1, 2, 3]);
    expect(c.halfSize[0]).toBeCloseTo(2 * CROP_BOX_HALF_EXTENT);
    expect(c.halfSize[1]).toBeCloseTo(3 * CROP_BOX_HALF_EXTENT);
    expect(c.halfSize[2]).toBeCloseTo(1e-3);
    expect(c.quaternion).toEqual([0, 0, 0, 1]);
    expect(o.quaternion.w).toBe(2);
  });
  it("throws on non-finite values", () => {
    const o = new Object3D();
    o.scale.set(Infinity, 1, 1);
    expect(() => readCrop(o)).toThrow("Non-finite crop");
    const p = new Object3D();
    p.position.set(0, NaN, 0);
    expect(() => readCrop(p)).toThrow("Non-finite crop");
  });
});

describe("isValidTransform", () => {
  it("accepts a sane transform", () => {
    expect(isValidTransform({ position: [0, 1, 2], quaternion: [0, 0, 0, 1], scale: 1 })).toBe(true);
  });
  it("rejects NaN, zero/negative scale and non-unit quaternions", () => {
    expect(isValidTransform({ position: [NaN, 0, 0], quaternion: [0, 0, 0, 1], scale: 1 })).toBe(false);
    expect(isValidTransform({ position: [0, 0, 0], quaternion: [0, 0, 0, 1], scale: 0 })).toBe(false);
    expect(isValidTransform({ position: [0, 0, 0], quaternion: [0, 0, 0, 1], scale: -1 })).toBe(false);
    expect(isValidTransform({ position: [0, 0, 0], quaternion: [0, 0, 0, 2], scale: 1 })).toBe(false);
  });
});


function rotate(q: [number, number, number, number], v: [number, number, number]): Vector3 {
  return new Vector3(...v).applyQuaternion(new Quaternion(...q));
}

function unit(v: [number, number, number]): [number, number, number] {
  const n = Math.hypot(...v);
  return [v[0] / n, v[1] / n, v[2] / n];
}

describe("meshInsertQuaternion", () => {
  it("maps glTF +Y onto the scene up", () => {
    const up = unit([-0.94, -0.11, -0.32]);
    const r = rotate(meshInsertQuaternion(up), [0, 1, 0]);
    expect(r.x).toBeCloseTo(up[0], 6);
    expect(r.y).toBeCloseTo(up[1], 6);
    expect(r.z).toBeCloseTo(up[2], 6);
    expect(meshInsertQuaternion([0, 1, 0])).toEqual([0, 0, 0, 1]);
    expect(rotate(meshInsertQuaternion([0, -1, 0]), [0, 1, 0]).y).toBeCloseTo(-1, 6);
  });
});

describe("straightenQuaternion", () => {
  it("from identity equals the shortest rotation objUp → sceneUp", () => {
    const objUp = unit([0.2, 0.1, 0.97]);
    const sceneUp = unit([-0.94, -0.11, -0.32]);
    const q = straightenQuaternion([0, 0, 0, 1], objUp, sceneUp);
    const expected = new Quaternion().setFromUnitVectors(new Vector3(...objUp), new Vector3(...sceneUp));
    expect(q[0]).toBeCloseTo(expected.x, 9);
    expect(q[1]).toBeCloseTo(expected.y, 9);
    expect(q[2]).toBeCloseTo(expected.z, 9);
    expect(q[3]).toBeCloseTo(expected.w, 9);
  });

  it("maps the object's own up onto the scene up from any current rotation", () => {
    const objUp = unit([0, 0, 1]);
    const sceneUp = unit([0.3, -0.9, 0.1]);
    const current = new Quaternion().setFromAxisAngle(new Vector3(1, 2, 3).normalize(), 1.1);
    const q = straightenQuaternion([current.x, current.y, current.z, current.w], objUp, sceneUp);
    expect(Math.hypot(...q)).toBeCloseTo(1, 9);
    const r = rotate(q, objUp);
    expect(r.x).toBeCloseTo(sceneUp[0], 6);
    expect(r.y).toBeCloseTo(sceneUp[1], 6);
    expect(r.z).toBeCloseTo(sceneUp[2], 6);
  });

  it("handles opposite vectors (upside-down objects)", () => {
    const q = straightenQuaternion([0, 0, 0, 1], [0, -1, 0], [0, 1, 0]);
    expect(rotate(q, [0, -1, 0]).y).toBeCloseTo(1, 6);
  });
});

describe("orbitPositionForUp", () => {
  it("mirrors the camera for a +Y → −Y flip (same distance and elevation)", () => {
    const p = orbitPositionForUp([1, 3, 4], [1, 1, 0], [0, 1, 0], [0, -1, 0]);
    expect(p[0]).toBeCloseTo(1, 9);
    expect(p[1]).toBeCloseTo(-1, 9);
    expect(p[2]).toBeCloseTo(4, 9);
  });

  it("keeps distance and elevation for an arbitrary up", () => {
    const target: [number, number, number] = [0.5, -1, 2];
    const up = unit([-0.94, -0.11, -0.32]);
    const p = new Vector3(...orbitPositionForUp([0, 1.5, 4], target, [0, 1, 0], up));
    const d = p.sub(new Vector3(...target));
    const before = new Vector3(0, 1.5, 4).sub(new Vector3(...target));
    expect(d.length()).toBeCloseTo(before.length(), 9);
    expect(Math.asin(d.clone().normalize().dot(new Vector3(...up)))).toBeCloseTo(
      Math.asin(before.clone().normalize().y),
      9,
    );
  });

  it("uses the given elevation and survives a camera on the target or along up", () => {
    const p = orbitPositionForUp([0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1], Math.PI / 6);
    expect(Math.hypot(...p)).toBeCloseTo(4, 9);
    expect(p[2]).toBeCloseTo(2, 9);
    const q = orbitPositionForUp([0, 5, 0], [0, 0, 0], [0, 1, 0], [0, 1, 0]);
    expect(q.every(Number.isFinite)).toBe(true);
    expect(Math.hypot(...q)).toBeCloseTo(5, 9);
  });
});
