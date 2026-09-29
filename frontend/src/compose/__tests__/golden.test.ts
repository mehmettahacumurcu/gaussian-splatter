import { describe, expect, it } from "vitest";
import golden from "../../../../tests/fixtures/compose_golden.json";
import { applyColor } from "../colorMath";
import { applyTransform, isInsideCrop, uniformScaleFrom } from "../transformMath";
import type { ColorAdjust, CropBox, Transform, Vec3 } from "../types";

function close(a: number[], b: number[], eps = 1e-5) {
  expect(a.length).toBe(b.length);
  a.forEach((v, i) => expect(Math.abs(v - b[i])).toBeLessThan(eps));
}

describe("preview math matches the backend golden fixture", () => {
  it("transforms points like the bake", () => {
    for (const c of golden.transform_cases) {
      c.points.forEach((p, i) => close(applyTransform(c.transform as Transform, p as Vec3), c.expected[i]));
    }
  });

  it("crops like the bake", () => {
    for (const c of golden.crop_cases) {
      const inside = c.points.map((p) => isInsideCrop(c.crop as CropBox, p as Vec3));
      expect(inside).toEqual(c.inside);
      expect(inside).toContain(true);
      expect(inside).toContain(false);
    }
  });

  it("colour-matches like the bake", () => {
    for (const c of golden.color_cases) {
      c.rgb.forEach((rgb, i) => close(applyColor(c.color as ColorAdjust, rgb as Vec3), c.expected[i]));
    }
  });
});

describe("uniformScaleFrom", () => {
  it("follows the axis the gizmo changed", () => {
    expect(uniformScaleFrom(1, { x: 1, y: 2.5, z: 1 })).toBe(2.5);
    expect(uniformScaleFrom(2, { x: 0.5, y: 2, z: 2 })).toBe(0.5);
    expect(uniformScaleFrom(1, { x: -3, y: 1, z: 1 })).toBeGreaterThan(0);
  });
});
