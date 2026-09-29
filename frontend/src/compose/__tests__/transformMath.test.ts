import { Object3D } from "three";
import { describe, expect, it } from "vitest";
import {
  CROP_BOX_HALF_EXTENT,
  isValidTransform,
  readCrop,
  readTransform,
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
