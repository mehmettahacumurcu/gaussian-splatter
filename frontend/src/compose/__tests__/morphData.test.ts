import { describe, expect, it } from "vitest";
import { MORPH_STRIDE, mortonCode, mortonOrder, prepareMorph, robustMorphBounds } from "../morphData";

function splats(positions: number[][]): Float32Array {
  const attributes = new Float32Array(positions.length * MORPH_STRIDE);
  positions.forEach((position, i) => {
    attributes.set([
      ...position, 0.25 + i / (positions.length + 1) * 0.5,
      0.1, 0.2, 0.3, 0,
      0, 0, 0, 1,
      0.2, 0.4, 0.8, i,
    ], i * MORPH_STRIDE);
  });
  return attributes;
}

function assertEndpointOriginals(original: Float32Array, paired: Float32Array): void {
  const visible = [];
  for (let offset = 0; offset < paired.length; offset += MORPH_STRIDE) {
    if (paired[offset + 3] === 0) continue;
    const index = paired[offset + 15];
    visible.push(index);
    for (let component = 0; component < MORPH_STRIDE; component++) {
      if (component !== 7) expect(paired[offset + component]).toBe(original[index * MORPH_STRIDE + component]);
    }
  }
  expect(visible.sort((a, b) => a - b)).toEqual(Array.from({ length: original.length / MORPH_STRIDE }, (_, i) => i));
}

describe("Morton correspondence", () => {
  it("interleaves all three axes and clamps to the 10-bit cube", () => {
    expect(mortonCode(0, 0, 0)).toBe(0);
    expect(mortonCode(1, 0, 0)).toBe(1);
    expect(mortonCode(0, 1, 0)).toBe(2);
    expect(mortonCode(0, 0, 1)).toBe(4);
    expect(mortonCode(2, 0, 0)).toBe(8);
    expect(mortonCode(1023, 1023, 1023)).toBe(0x3fffffff);
    expect(mortonCode(-1e20, 1e20, 1024)).toBe(mortonCode(0, 1023, 1023));
  });

  it("orders normalized positions, with stable input-order ties", () => {
    const attributes = splats([[1, 1, 1], [0, 0, 1], [1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 1, 0]]);
    expect(Array.from(mortonOrder(attributes))).toEqual([3, 4, 2, 5, 1, 0]);
  });

  it("matches translated, uniformly scaled targets without changing world endpoints", () => {
    const a = splats([[1, 1, 1], [0, 0, 1], [1, 0, 0], [0, 0, 0], [0, 1, 0]]);
    const b = a.slice();
    for (let offset = 0; offset < b.length; offset += MORPH_STRIDE) {
      for (let axis = 0; axis < 3; axis++) b[offset + axis] = b[offset + axis] * 4 + 20;
    }
    const prepared = prepareMorph(a, b, 42);
    for (let offset = 0; offset < prepared.a.length; offset += MORPH_STRIDE) {
      expect(prepared.a[offset + 15]).toBe(prepared.b[offset + 15]);
      for (let axis = 0; axis < 3; axis++) expect(prepared.b[offset + axis]).toBe(prepared.a[offset + axis] * 4 + 20);
    }
    assertEndpointOriginals(a, prepared.a);
    assertEndpointOriginals(b, prepared.b);
  });

  it("uses robust bounds so isolated floaters do not set cloud size", () => {
    const positions = Array.from({ length: 98 }, (_, i) => [i, 0, 0]);
    positions.push([-10000, 0, 0], [10000, 0, 0]);
    const bounds = robustMorphBounds(splats(positions));
    expect(bounds.min).toEqual([0, 0, 0]);
    expect(bounds.max).toEqual([97, 0, 0]);
    expect(bounds.radius).toBe(48.5);
  });
});

describe("morph pairing and padding", () => {
  it.each([[2, 5], [5, 2], [1, 10001], [10001, 1], [1, 1], [7, 7]])(
    "preserves all original endpoint attributes for %i and %i splats",
    (countA, countB) => {
      const a = splats(Array.from({ length: countA }, (_, i) => [i, i % 3, 0]));
      const b = splats(Array.from({ length: countB }, (_, i) => [i * 2 + 10, i % 5, 1]));
      const prepared = prepareMorph(a, b, 123);
      expect(prepared.count).toBe(Math.max(countA, countB));
      expect(prepared.a.length).toBe(prepared.count * MORPH_STRIDE);
      expect(prepared.b.length).toBe(prepared.count * MORPH_STRIDE);
      expect(prepared.sceneRadius).toBeGreaterThan(0);
      assertEndpointOriginals(a, prepared.a);
      assertEndpointOriginals(b, prepared.b);
    },
  );

  it("pads with adjacent rank clones and zero opacity", () => {
    const prepared = prepareMorph(splats([[0, 0, 0], [1, 0, 0]]), splats([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]]), 1);
    const ids = Array.from({ length: prepared.count }, (_, i) => prepared.a[i * MORPH_STRIDE + 15]);
    const opacity = Array.from({ length: prepared.count }, (_, i) => prepared.a[i * MORPH_STRIDE + 3]);
    expect(ids).toEqual([0, 0, 1, 1, 1]);
    expect(opacity[0]).toBeGreaterThan(0);
    expect(opacity[2]).toBeGreaterThan(0);
    expect([opacity[1], opacity[3], opacity[4]]).toEqual([0, 0, 0]);
  });

  it("is deterministic, shares particle seeds between endpoints, and leaves inputs untouched", () => {
    const a = splats([[0, 0, 0], [1, 2, 3]]);
    const b = splats([[1, 0, 0], [4, 2, 3], [1, 2, 1]]);
    const originalA = a.slice();
    const originalB = b.slice();
    const first = prepareMorph(a, b, 42);
    const second = prepareMorph(a, b, 42);
    const changed = prepareMorph(a, b, 43);
    expect(first).toEqual(second);
    expect(a).toEqual(originalA);
    expect(b).toEqual(originalB);
    for (let i = 0; i < first.count; i++) {
      const offset = i * MORPH_STRIDE + 7;
      expect(first.a[offset]).toBe(first.b[offset]);
      expect(first.a[offset]).toBeGreaterThanOrEqual(0);
      expect(first.a[offset]).toBeLessThan(1);
      expect(first.a[offset]).not.toBe(changed.a[offset]);
    }
  });

  it("reports empty, incomplete, non-finite, and negative-scale inputs", () => {
    const good = splats([[0, 0, 0]]);
    expect(() => prepareMorph(new Float32Array(), good, 1)).toThrow("complete splat");
    expect(() => prepareMorph(good, new Float32Array(17), 1)).toThrow("complete splat");
    const invalid = good.slice();
    invalid[0] = Infinity;
    expect(() => prepareMorph(invalid, good, 1)).toThrow("non-finite");
    invalid[0] = NaN;
    expect(() => prepareMorph(good, invalid, 1)).toThrow("non-finite");
    invalid[0] = 0;
    invalid[5] = -1;
    expect(() => prepareMorph(good, invalid, 1)).toThrow("negative");
    expect(() => prepareMorph(good, good, NaN)).toThrow("seed");
  });
});
