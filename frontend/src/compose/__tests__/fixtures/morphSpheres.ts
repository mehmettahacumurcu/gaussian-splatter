import { MORPH_STRIDE, robustMorphBounds } from "../../morphData.ts";
import type { MorphBounds, PreparedMorph } from "../../morphData.ts";

/** Independent angular samples of a sphere with ±1% radial noise. */
export function noisySphere(count: number, seed: number): Float32Array {
  let state = seed >>> 0;
  function random(): number {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  }
  const attributes = new Float32Array(count * MORPH_STRIDE);
  for (let i = 0; i < count; i++) {
    const z = 2 * random() - 1;
    const angle = 2 * Math.PI * random();
    const radius = 1 + (random() - 0.5) * 0.02;
    const xy = Math.sqrt(1 - z * z);
    const offset = i * MORPH_STRIDE;
    attributes[offset] = radius * xy * Math.cos(angle);
    attributes[offset + 1] = radius * xy * Math.sin(angle);
    attributes[offset + 2] = radius * z;
    attributes[offset + 3] = 0.8;
    attributes[offset + 4] = attributes[offset + 5] = attributes[offset + 6] = 0.005;
    attributes[offset + 11] = 1;
    attributes[offset + 12] = 0.9;
    attributes[offset + 13] = 0.4;
    attributes[offset + 14] = 0.1;
  }
  return attributes;
}

export function normalizedPosition(attributes: Float32Array, offset: number, bounds: MorphBounds): number[] {
  return [0, 1, 2].map((axis) => (attributes[offset + axis] - bounds.center[axis]) / bounds.size);
}

/** Includes invisible padding paths, in units of each scene's robust diameter. */
export function meanNormalizedTravel(prepared: PreparedMorph, originalA: Float32Array, originalB: Float32Array): number {
  const boundsA = robustMorphBounds(originalA);
  const boundsB = robustMorphBounds(originalB);
  let total = 0;
  for (let i = 0; i < prepared.count; i++) {
    const offset = i * MORPH_STRIDE;
    const a = normalizedPosition(prepared.a, offset, boundsA);
    const b = normalizedPosition(prepared.b, offset, boundsB);
    total += Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  }
  return total / prepared.count;
}

/** Unconstrained nearest-neighbor relaxation: a true lower bound on any pairing. */
export function nearestNeighborLowerBound(a: Float32Array, b: Float32Array): number {
  const boundsA = robustMorphBounds(a);
  const boundsB = robustMorphBounds(b);
  const small = a.length < b.length ? a : b;
  const large = a.length < b.length ? b : a;
  const smallBounds = a.length < b.length ? boundsA : boundsB;
  const largeBounds = a.length < b.length ? boundsB : boundsA;
  const positions = Array.from({ length: small.length / MORPH_STRIDE }, (_, i) => normalizedPosition(small, i * MORPH_STRIDE, smallBounds));
  let total = 0;
  for (let i = 0; i < large.length; i += MORPH_STRIDE) {
    const position = normalizedPosition(large, i, largeBounds);
    let minSquared = Infinity;
    for (const other of positions) {
      const dx = position[0] - other[0], dy = position[1] - other[1], dz = position[2] - other[2];
      minSquared = Math.min(minSquared, dx * dx + dy * dy + dz * dz);
    }
    total += Math.sqrt(minSquared);
  }
  return total / (large.length / MORPH_STRIDE);
}
