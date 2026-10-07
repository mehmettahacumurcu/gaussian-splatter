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

describe("shape-preserving interpolation", () => {
  const sample = (source: Float32Array, target: Float32Array, t: number, wave = 0, arc = 0, blend = 1) => (
    interpolateMorphParticle(source, target, 0, t, 1, blend, 10, "shape", wave, arc)
  );

  it("preserves exact endpoints and zero-opacity clones across all shape controls", () => {
    const source = a.slice();
    source[3] = source[4] = 0;
    const target = b.slice();
    target[3] = target[5] = 0;
    for (const wave of [0, 0.4, 1]) {
      for (const arc of [0, 0.4, 1]) {
        for (const projection of [0, 0.4, 1]) {
          source[15] = projection;
          expect(sample(source, target, 0, wave, arc)).toEqual(source);
          expect(sample(source, target, 1, wave, arc)).toEqual(target);
        }
      }
    }
  });

  it("follows a smootherstep straight path with log scales, slerp, and linear appearance", () => {
    for (const t of [0.1, 0.25, 0.5, 0.8]) {
      const u = 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3;
      const result = sample(a, b, t);
      for (let axis = 0; axis < 3; axis++) {
        expect(result[axis]).toBeCloseTo(a[axis] + (b[axis] - a[axis]) * u, 5);
        expect(result[4 + axis]).toBeCloseTo(a[4 + axis] ** (1 - u) * b[4 + axis] ** u, 5);
        expect(result[12 + axis]).toBeCloseTo(a[12 + axis] + (b[12 + axis] - a[12 + axis]) * u, 6);
      }
      expect(result[3]).toBeCloseTo(a[3] + (b[3] - a[3]) * u, 6);
      expect(result[9]).toBeCloseTo(Math.sin(u * Math.PI / 2), 6);
      expect(result[11]).toBeCloseTo(Math.cos(u * Math.PI / 2), 6);
    }
  });

  it("ignores cloud controls in shape mode and shape controls in cloud mode", () => {
    const plain = interpolateMorphParticle(a, b, 0, 0.3, 0, 1, 0, "shape");
    expect(interpolateMorphParticle(a, b, 0, 0.3, 1, 1, 10000, "shape")).toEqual(plain);
    expect(interpolateMorphParticle(a, b, 0, 0.3, 0.5, 1, 10, "cloud", 1, 1))
      .toEqual(interpolateMorphParticle(a, b, 0, 0.3, 0.5, 1, 10));
  });

  it("sweeps from source X=0 to X=1 with the same total duration", () => {
    const early = a.slice();
    const late = a.slice();
    early[15] = 0;
    late[15] = 1;
    expect(sample(late, b, 0.49, 1)).toEqual(late);
    expect(sample(early, b, 0.5, 1)).toEqual(b);
    expect(sample(late, b, 0.5, 1)).toEqual(late);
    const earlyMid = sample(early, b, 0.25, 1);
    const lateMid = sample(late, b, 0.75, 1);
    expect(Array.from(earlyMid.slice(0, 15))).toEqual(Array.from(lateMid.slice(0, 15)));
    for (let axis = 0; axis < 3; axis++) expect(lateMid[axis]).toBeCloseTo((a[axis] + b[axis]) / 2);
    expect(sample(late, b, 1, 1)).toEqual(b);
  });

  it("uses the requested wave delay fraction before smootherstep", () => {
    const source = a.slice();
    source[15] = 0.6;
    const wave = 0.4;
    const delay = 0.5 * wave * source[15];
    expect(sample(source, b, delay, wave)).toEqual(source);
    const midpoint = sample(source, b, delay + (1 - wave * 0.5) * 0.5, wave);
    for (let axis = 0; axis < 3; axis++) expect(midpoint[axis]).toBeCloseTo((a[axis] + b[axis]) / 2);
  });

  it("keeps seeded arcs perpendicular and within five percent of pair distance", () => {
    for (const delta of [[7, 7, 7], [3, 0, 0], [0, -4, 0], [0, 0, 5], [0.00001, -0.00002, 0.00003]]) {
      const source = a.slice();
      source.fill(0, 0, 3);
      const target = b.slice();
      target.set(delta);
      const distance = Math.hypot(...target.slice(0, 3));
      for (const arc of [0.25, 1]) {
        for (const t of [0.1, 0.3, 0.5, 0.9]) {
          const straight = sample(source, target, t);
          const curved = sample(source, target, t, 0, arc);
          const displacement = [0, 1, 2].map((axis) => curved[axis] - straight[axis]);
          expect(sample(source, target, t, 0, arc)).toEqual(curved);
          expect(Math.abs(displacement.reduce((dot, component, axis) => dot + component * target[axis], 0)))
            .toBeLessThanOrEqual(distance ** 2 * 1e-6);
          expect(Math.hypot(...displacement)).toBeCloseTo(distance * 0.05 * arc * Math.sin(Math.PI * t) ** 2, 6);
          expect(Math.hypot(...displacement)).toBeLessThanOrEqual(distance * 0.05 * arc + 1e-6);
        }
      }
    }
    const differentSeed = a.slice();
    differentSeed[7] += 0.25;
    expect(sample(differentSeed, b, 0.5, 0, 1).slice(0, 3)).not.toEqual(sample(a, b, 0.5, 0, 1).slice(0, 3));
  });

  it("keeps identical centers stationary and degenerate attributes finite", () => {
    const source = a.slice();
    source.fill(0, 4, 7);
    source.fill(0, 8, 12);
    const target = b.slice();
    target.set(source.slice(0, 3));
    target.fill(0, 8, 12);
    for (const t of [0.01, 0.3, 0.5, 0.79]) {
      const result = sample(source, target, t, 0.4, 1);
      expect(result.slice(0, 3)).toEqual(source.slice(0, 3));
      expect(Array.from(result).every(Number.isFinite)).toBe(true);
      expect(Math.hypot(...result.slice(8, 12))).toBeCloseTo(1, 6);
      for (let axis = 4; axis < 7; axis++) expect(result[axis]).toBeGreaterThanOrEqual(1e-6 - 1e-12);
    }
  });

  it("preserves targetBlend semantics with wave and clamps shape controls", () => {
    const source = a.slice();
    source[15] = 0.7;
    expect(sample(source, b, 1, 0.7, 0.5, 0)).toEqual(source);
    expect(sample(source, b, 1, 0.7, 0.5, 0.6)).toEqual(sample(source, b, 0.6, 0.7, 0.5));
    expect(sample(source, b, -1, 1, 1)).toEqual(source);
    expect(sample(source, b, 2, 1, 1, 2)).toEqual(b);
    expect(sample(source, b, 0.7, -1, -1)).toEqual(sample(source, b, 0.7));
    expect(sample(source, b, 0.7, 2, 2)).toEqual(sample(source, b, 0.7, 1, 1));
    expect(() => sample(source, b, 0.5, NaN)).toThrow("finite");
    expect(() => sample(source, b, 0.5, 0, Infinity)).toThrow("finite");
  });
});
