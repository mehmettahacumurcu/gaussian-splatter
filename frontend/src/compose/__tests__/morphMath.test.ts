import { describe, expect, it } from "vitest";
import { MORPH_STRIDE, prepareMorph } from "../morphData";
import { interpolateMorphParticle } from "../morphMath";

const a = new Float32Array([1, 2, 3, 0.9, 0.01, 0.2, 2, 0.317, 0, 0, 0, 1, 0.2, 0.4, 0.6, 0]);
const b = new Float32Array([8, 9, 10, 0.6, 0.3, 0.02, 3, 0.317, 0, 1, 0, 0, 0.7, 0.3, 0.1, 0]);

describe("morph interpolation", () => {
  it("returns exactly A and B at the full timeline endpoints, including zero scales", () => {
    const zeroScale = a.slice();
    zeroScale[4] = 0;
    for (const dissolve of [0, 0.3, 1]) {
      expect(interpolateMorphParticle(zeroScale, b, 0, 0, dissolve, 1, 10)).toEqual(zeroScale);
      expect(interpolateMorphParticle(zeroScale, b, 0, 1, dissolve, 1, 10)).toEqual(b);
    }
  });

  it("preserves padded endpoint opacities exactly", () => {
    const pair = prepareMorph(a, new Float32Array([...b, ...a, ...b]), 7);
    for (let index = 0; index < pair.count; index++) {
      expect(interpolateMorphParticle(pair.a, pair.b, index, 0, 1, 1, pair.sceneRadius)).toEqual(pair.a.slice(index * MORPH_STRIDE, (index + 1) * MORPH_STRIDE));
      expect(interpolateMorphParticle(pair.a, pair.b, index, 1, 1, 1, pair.sceneRadius)).toEqual(pair.b.slice(index * MORPH_STRIDE, (index + 1) * MORPH_STRIDE));
    }
  });

  it("uses targetBlend as the reachable fraction of the transition and clamps UI ranges", () => {
    expect(interpolateMorphParticle(a, b, 0, 1, 1, 0, 10)).toEqual(a);
    expect(interpolateMorphParticle(a, b, 0, 1, 1, 0.5, 10)).toEqual(interpolateMorphParticle(a, b, 0, 0.5, 1, 1, 10));
    expect(interpolateMorphParticle(a, b, 0, -1, 1, 1, 10)).toEqual(a);
    expect(interpolateMorphParticle(a, b, 0, 2, 1, 2, 10)).toEqual(b);
  });

  it("makes full-dissolve midpoint scales small and isotropic", () => {
    const result = interpolateMorphParticle(a, b, 0, 0.5, 1, 1, 10);
    for (let axis = 4; axis < 7; axis++) expect(result[axis]).toBeCloseTo(0.015, 7);
    expect(result[3]).toBeCloseTo((a[3] + b[3]) * 0.5 * 0.75);
  });

  it("uses log-scale interpolation and shortest quaternion slerp", () => {
    const result = interpolateMorphParticle(a, b, 0, 0.5, 0, 1, 10);
    for (let axis = 0; axis < 3; axis++) {
      expect(result[axis]).toBeCloseTo((a[axis] + b[axis]) * 0.5);
      expect(result[4 + axis]).toBeCloseTo(Math.sqrt(a[4 + axis] * b[4 + axis]));
      expect(result[12 + axis]).toBeCloseTo((a[12 + axis] + b[12 + axis]) * 0.5);
    }
    expect(result[9]).toBeCloseTo(Math.SQRT1_2);
    expect(result[11]).toBeCloseTo(Math.SQRT1_2);
    const antipodal = a.slice();
    antipodal[11] = -1;
    expect(Array.from(interpolateMorphParticle(a, antipodal, 0, 0.5, 0, 1, 10).slice(8, 12))).toEqual([0, 0, 0, 1]);
  });

  it("keeps scales finite and bounded, rotations normalized, and seeded drift inside its radius", () => {
    const tiny = a.slice();
    tiny[4] = 0;
    tiny[5] = 1e-9;
    tiny[6] = 200;
    const radius = 8;
    for (const dissolve of [0, 0.5, 1]) {
      for (let tick = 1; tick < 100; tick++) {
        const t = tick / 100;
        const result = interpolateMorphParticle(tiny, b, 0, t, dissolve, 1, radius);
        const repeat = interpolateMorphParticle(tiny, b, 0, t, dissolve, 1, radius);
        expect(result).toEqual(repeat);
        for (let axis = 4; axis < 7; axis++) {
          expect(Number.isFinite(result[axis])).toBe(true);
          expect(result[axis]).toBeGreaterThanOrEqual(1e-6 - 1e-12);
          expect(result[axis]).toBeLessThanOrEqual(200);
        }
        expect(Math.hypot(...result.slice(8, 12))).toBeCloseTo(1, 6);
        const u = t * t * (3 - 2 * t);
        const driftLength = Math.hypot(...[0, 1, 2].map((axis) => result[axis] - (tiny[axis] * (1 - u) + b[axis] * u)));
        expect(driftLength).toBeLessThanOrEqual(radius * 1.5 * dissolve * Math.sin(Math.PI * t) ** 2 + 1e-5);
      }
    }
  });

  it("handles degenerate quaternion input and rejects invalid reference arguments", () => {
    const zeroQuaternion = a.slice();
    zeroQuaternion.fill(0, 8, 12);
    const result = interpolateMorphParticle(zeroQuaternion, b, 0, 0.5, 1, 1, 0);
    expect(Array.from(result).every(Number.isFinite)).toBe(true);
    expect(Math.hypot(...result.slice(8, 12))).toBeCloseTo(1, 6);
    expect(() => interpolateMorphParticle(a, b, -1, 0.5, 1, 1, 10)).toThrow("index");
    expect(() => interpolateMorphParticle(a, b, 1, 0.5, 1, 1, 10)).toThrow("index");
    expect(() => interpolateMorphParticle(a, b, 0, NaN, 1, 1, 10)).toThrow("finite");
  });
});
