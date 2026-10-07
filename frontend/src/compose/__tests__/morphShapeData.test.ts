import { describe, expect, it } from "vitest";
import { createHash } from "node:crypto";
import { MORPH_STRIDE, prepareMorph, robustMorphBounds } from "../morphData";
import { meanNormalizedTravel, nearestNeighborLowerBound, noisySphere } from "./fixtures/morphSpheres";
import { asymmetricCloud, rotatedCloud } from "./fixtures/morphAlignment";
import { alignMorphCoordinates } from "../morphAlignment";

function splats(positions: number[][]): Float32Array {
  const attributes = new Float32Array(positions.length * MORPH_STRIDE);
  positions.forEach((position, index) => {
    attributes.set([...position, 0.75, 0.1, 0.2, 0.3, 0, 0, 0, 0, 1, index, 0.4, 0.8, 0], index * MORPH_STRIDE);
  });
  return attributes;
}

function assertOriginals(original: Float32Array, paired: Float32Array): void {
  const visible: number[] = [];
  for (let offset = 0; offset < paired.length; offset += MORPH_STRIDE) {
    if (paired[offset + 3] === 0) continue;
    const index = paired[offset + 12];
    visible.push(index);
    for (let component = 0; component < MORPH_STRIDE; component++) {
      // Seed and sweep coordinate are correspondence metadata, not attributes.
      if (component !== 7 && component !== 15) expect(paired[offset + component]).toBe(original[index * MORPH_STRIDE + component]);
    }
  }
  expect(visible.sort((a, b) => a - b)).toEqual(Array.from({ length: original.length / MORPH_STRIDE }, (_, index) => index));
}

describe("shape-preserving spatial correspondence", () => {
  it("keeps unequal noisy spheres local and close to the nearest-neighbor relaxation", () => {
    const a = noisySphere(1500, 12345);
    const b = noisySphere(2000, 67890);
    const cloud = meanNormalizedTravel(prepareMorph(a, b, 42, "cloud"), a, b);
    const shape = meanNormalizedTravel(prepareMorph(a, b, 42, "shape"), a, b);
    const lowerBound = nearestNeighborLowerBound(a, b);
    expect(shape).toBeLessThan(cloud * 0.5);
    // Capacity-free nearest neighbors are unattainable in general when every
    // original must survive. Here balanced matching is within 3x that bound and
    // within 4.5% of the robust diameter in absolute excess travel.
    expect(shape).toBeGreaterThanOrEqual(lowerBound - 1e-7);
    expect(shape).toBeLessThan(lowerBound * 3);
    expect(shape - lowerBound).toBeLessThan(0.045);
  });

  it("recovers exact positional identity for shuffled inputs, including coordinate ties", () => {
    const a = noisySphere(4093, 678);
    const b = new Float32Array(a.length);
    // 37 and 4093 are coprime, so this is a complete permutation.
    for (let i = 0; i < 4093; i++) b.set(a.subarray(i * MORPH_STRIDE, (i + 1) * MORPH_STRIDE), ((i * 37) % 4093) * MORPH_STRIDE);
    expect(meanNormalizedTravel(prepareMorph(a, b, 42, "shape"), a, b)).toBe(0);
    const tiedA = splats(Array.from({ length: 127 }, (_, i) => [i % 3, Math.floor(i / 3) % 3, i % 2]));
    const tiedB = new Float32Array(tiedA.length);
    for (let i = 0; i < 127; i++) tiedB.set(tiedA.subarray(i * MORPH_STRIDE, (i + 1) * MORPH_STRIDE), ((i * 37) % 127) * MORPH_STRIDE);
    expect(meanNormalizedTravel(prepareMorph(tiedA, tiedB, 42, "shape"), tiedA, tiedB)).toBe(0);
  });

  it.each([[2, 5], [5, 2], [1, 101], [101, 1], [1, 1], [37, 37], [31, 57], [57, 31]])(
    "preserves every original exactly once and pads locally for %i vs %i",
    (countA, countB) => {
      const a = splats(Array.from({ length: countA }, (_, i) => [i, i % 3, i % 5]));
      const b = splats(Array.from({ length: countB }, (_, i) => [i * 2 + 10, i % 7, 1]));
      const originalA = a.slice();
      const originalB = b.slice();
      const prepared = prepareMorph(a, b, 42, "shape");
      expect(prepared.count).toBe(Math.max(countA, countB));
      expect(prepared.a.length).toBe(prepared.count * MORPH_STRIDE);
      expect(prepared.b.length).toBe(prepared.count * MORPH_STRIDE);
      assertOriginals(a, prepared.a);
      assertOriginals(b, prepared.b);
      expect(a).toEqual(originalA);
      expect(b).toEqual(originalB);
      const bounds = robustMorphBounds(a);
      for (let offset = 0; offset < prepared.a.length; offset += MORPH_STRIDE) {
        const width = bounds.max[0] - bounds.min[0];
        const projection = width <= 1e-6 ? 1 : Math.max(0, Math.min(1, (prepared.a[offset] - bounds.min[0]) / width));
        expect(prepared.a[offset + 15]).toBeCloseTo(projection, 6);
        expect(prepared.b[offset + 15]).toBe(prepared.a[offset + 15]);
      }
    },
  );

  it("uses the whole sweep duration for narrow and flat source bounds", () => {
    const a = splats([[0, 0, 0], [0.01, 100, 0]]);
    const b = splats([[1, 0, 0], [1.01, 100, 0]]);
    const prepared = prepareMorph(a, b, 42, "shape");
    expect(Array.from({ length: prepared.count }, (_, i) => prepared.a[i * MORPH_STRIDE + 15]).sort()).toEqual([0, 1]);
    const flat = prepareMorph(splats([[0, 0, 0], [0, 100, 0]]), b, 42, "shape");
    expect(flat.a[15]).toBe(1);
    expect(flat.a[MORPH_STRIDE + 15]).toBe(1);
  });

  it("keeps correspondence seed-independent and reproducible while varying animation seeds", () => {
    const a = noisySphere(101, 53);
    const b = noisySphere(127, 91);
    const first = prepareMorph(a, b, 42, "shape");
    expect(prepareMorph(a, b, 42, "shape")).toEqual(first);
    const changed = prepareMorph(a, b, 43, "shape");
    for (let i = 0; i < first.a.length; i++) {
      if (i % MORPH_STRIDE === 7) {
        expect(changed.a[i]).not.toBe(first.a[i]);
        expect(first.a[i]).toBe(first.b[i]);
      } else {
        expect(changed.a[i]).toBe(first.a[i]);
        expect(changed.b[i]).toBe(first.b[i]);
      }
    }
  });

  it("normalizes for matching without moving either endpoint", () => {
    const a = splats(Array.from({ length: 129 }, (_, i) => [i % 11, i % 7, i % 3]));
    const b = a.slice();
    for (let offset = 0; offset < b.length; offset += MORPH_STRIDE) {
      for (let axis = 0; axis < 3; axis++) b[offset + axis] = b[offset + axis] * 4 + 20;
    }
    const prepared = prepareMorph(a, b, 42, "shape");
    expect(meanNormalizedTravel(prepared, a, b)).toBe(0);
    assertOriginals(a, prepared.a);
    assertOriginals(b, prepared.b);
  });

  it("handles collapsed and ordered inputs without losing originals", () => {
    for (const positions of [
      Array.from({ length: 513 }, () => [0, 0, 0]),
      Array.from({ length: 513 }, (_, i) => [i, i, i]),
      Array.from({ length: 513 }, (_, i) => [Math.abs(i - 256), 0, 0]),
    ]) {
      const a = splats(positions);
      const b = splats([...positions].reverse());
      const prepared = prepareMorph(a, b, 42, "shape");
      expect(meanNormalizedTravel(prepared, a, b)).toBe(0);
      assertOriginals(a, prepared.a);
      assertOriginals(b, prepared.b);
    }
  });
});

describe("robust percentile selection", () => {
  it("matches sorted percentiles for duplicate, ascending, descending, organ-pipe and random coordinates", () => {
    const random = noisySphere(1001, 847);
    const arrays = [
      splats(Array.from({ length: 1001 }, () => [2, 2, 2])),
      splats(Array.from({ length: 1001 }, (_, i) => [i, 1000 - i, Math.abs(i - 500)])),
      random,
    ];
    for (const attributes of arrays) {
      const actual = robustMorphBounds(attributes);
      const count = attributes.length / MORPH_STRIDE;
      for (let axis = 0; axis < 3; axis++) {
        const sorted = Array.from({ length: count }, (_, i) => attributes[i * MORPH_STRIDE + axis]).sort((a, b) => a - b);
        expect(actual.min[axis]).toBe(sorted[Math.floor((count - 1) * 0.02)]);
        expect(actual.max[axis]).toBe(sorted[Math.ceil((count - 1) * 0.98)]);
      }
    }
  });
});

describe("optional automatic shape alignment", () => {
  it.each([53, 217, 1234, 93789])("recovers a shuffled arbitrarily rotated copy (seed %i)", (seed) => {
    const a = asymmetricCloud(4093);
    const b = rotatedCloud(a, seed, 3, 7, true);
    const off = prepareMorph(a, b, 42, "shape");
    const aligned = prepareMorph(a, b, 42, "shape", true);
    expect(off.matchingTravel).toBeGreaterThan(0.02);
    expect(aligned.matchingTravel).toBeLessThan(1e-5);
    // Alignment changes correspondence only: the far-away, rotated endpoint
    // still has a large rendered path, and every attribute is preserved.
    expect(aligned.meanTravel).toBeGreaterThan(1);
    expect(aligned.alignmentMs).toBeGreaterThan(0);
    if (seed === 53) {
      assertOriginals(a, aligned.a);
      assertOriginals(b, aligned.b);
    }
  });

  it("keeps alignment off and cloud mode byte-identical to the default path", () => {
    const a = asymmetricCloud(137), b = rotatedCloud(a, 123);
    // Full attribute-buffer hashes captured from e348246 before this change.
    const legacyHashes = {
      shape: "43f29d39de8f679188e21329b5b1dbef98c0bf3379351ebc1607496fca31addf",
      cloud: "4cf6a6e4398b3c661739da3ca5a0ec6c293f788425e50c66d5f1a7b75a4d3e8f",
    };
    for (const mode of ["shape", "cloud"] as const) {
      const result = prepareMorph(a, b, 42, mode, false);
      expect(result).toEqual(prepareMorph(a, b, 42, mode));
      expect(createHash("sha256").update(new Uint8Array(result.a.buffer)).update(new Uint8Array(result.b.buffer)).digest("hex")).toBe(legacyHashes[mode]);
    }
    expect(prepareMorph(a, b, 42, "cloud", true)).toEqual(prepareMorph(a, b, 42, "cloud"));
  });

  it("is deterministic and seed-independent, including unequal counts and refinement", () => {
    const a = asymmetricCloud(777, 35), b = rotatedCloud(asymmetricCloud(1021, 72), 167);
    const first = prepareMorph(a, b, 42, "shape", true);
    const second = prepareMorph(a, b, 42, "shape", true);
    expect({ ...first, alignmentMs: 0 }).toEqual({ ...second, alignmentMs: 0 });
    const otherSeed = prepareMorph(a, b, 72, "shape", true);
    expect(otherSeed.matchingTravel).toBe(first.matchingTravel);
    expect(otherSeed.meanTravel).toBe(first.meanTravel);
    assertOriginals(a, first.a);
    assertOriginals(b, first.b);
  });

  it("never reflects an asymmetric mirrored shape", () => {
    const a = asymmetricCloud(521), b = a.slice();
    for (let offset = 0; offset < b.length; offset += MORPH_STRIDE) b[offset] *= -1;
    const { rotation: r } = alignMorphCoordinates(a, b);
    const det = r[0] * (r[4] * r[8] - r[5] * r[7]) - r[1] * (r[3] * r[8] - r[5] * r[6]) + r[2] * (r[3] * r[7] - r[4] * r[6]);
    expect(det).toBeCloseTo(1, 10);
    expect(prepareMorph(a, b, 42, "shape", true).matchingTravel).toBeGreaterThan(0.01);
  });

  it("handles coincident points and lines without invalid coordinates", () => {
    for (const points of [Array.from({ length: 63 }, () => [1, 2, 3]), Array.from({ length: 63 }, (_, i) => [i, 0, 0])]) {
      const a = splats(points), b = rotatedCloud(a, 173);
      const result = prepareMorph(a, b, 42, "shape", true);
      expect(Number.isFinite(result.matchingTravel)).toBe(true);
      expect(result.matchingTravel).toBeLessThan(1e-5);
      assertOriginals(a, result.a);
      assertOriginals(b, result.b);
    }
  });

  it("reports real world endpoint travel as a fraction of the robust scene diagonal", () => {
    const a = splats([[0, 0, 0], [1, 0, 0]]), b = splats([[10, 0, 0], [11, 0, 0]]);
    for (const mode of ["shape", "cloud"] as const) {
      const result = prepareMorph(a, b, 42, mode);
      expect(result.meanTravel).toBe(10);
      expect(result.matchingTravel).toBe(0);
      expect(result.alignmentMs).toBe(0);
    }
  });
});
