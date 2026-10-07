/** Four vec4s per particle: position/opacity, scale/seed, quaternion, RGB/reserved. */
export const MORPH_STRIDE = 16;

export interface PreparedMorph {
  a: Float32Array;
  b: Float32Array;
  count: number;
  sceneRadius: number;
}

export interface MorphBounds {
  min: [number, number, number];
  max: [number, number, number];
  center: [number, number, number];
  /** Longest robust dimension, used for uniform (not per-axis) alignment. */
  size: number;
  radius: number;
}

function validateAttributes(attributes: Float32Array, label: string): number {
  const count = attributes.length / MORPH_STRIDE;
  if (!Number.isInteger(count) || count === 0) {
    throw new Error(`${label} must contain at least one complete splat.`);
  }
  for (let i = 0; i < attributes.length; i++) {
    if (!Number.isFinite(attributes[i])) {
      throw new Error(`${label} contains a non-finite splat attribute.`);
    }
    const component = i % MORPH_STRIDE;
    if (component >= 4 && component <= 6 && attributes[i] < 0) {
      throw new Error(`${label} contains a negative splat scale.`);
    }
  }
  return count;
}

/** Percentiles prevent a few distant floaters from shrinking the entire cloud. */
export function robustMorphBounds(attributes: Float32Array): MorphBounds {
  const count = validateAttributes(attributes, "Morph input");
  const min: MorphBounds["min"] = [0, 0, 0];
  const max: MorphBounds["max"] = [0, 0, 0];
  const center: MorphBounds["center"] = [0, 0, 0];
  const coordinates = new Float32Array(count);
  const lowRank = Math.floor((count - 1) * 0.02);
  const highRank = Math.ceil((count - 1) * 0.98);
  for (let axis = 0; axis < 3; axis++) {
    for (let i = 0; i < count; i++) coordinates[i] = attributes[i * MORPH_STRIDE + axis];
    coordinates.sort();
    min[axis] = coordinates[lowRank];
    max[axis] = coordinates[highRank];
    center[axis] = (min[axis] + max[axis]) * 0.5;
  }
  const x = max[0] - min[0];
  const y = max[1] - min[1];
  const z = max[2] - min[2];
  return { min, max, center, size: Math.max(x, y, z, 1e-6), radius: Math.hypot(x, y, z) * 0.5 };
}

function spreadMortonBits(value: number): number {
  let bits = Math.max(0, Math.min(1023, value)) | 0;
  bits = (bits | (bits << 16)) & 0x030000ff;
  bits = (bits | (bits << 8)) & 0x0300f00f;
  bits = (bits | (bits << 4)) & 0x030c30c3;
  return (bits | (bits << 2)) & 0x09249249;
}

/** 10 bits per axis, x in the least significant bit of each interleaved triplet. */
export function mortonCode(x: number, y: number, z: number): number {
  return (spreadMortonBits(x) | (spreadMortonBits(y) << 1) | (spreadMortonBits(z) << 2)) >>> 0;
}

/**
 * Centering and dividing by the longest robust dimension is equivalent to
 * aligning B's center and uniform scale to A before using A's Morton cube.
 * Alignment only determines correspondence; output attributes stay world-space.
 */
export function mortonOrder(attributes: Float32Array, bounds = robustMorphBounds(attributes)): Uint32Array {
  const count = attributes.length / MORPH_STRIDE;
  const codes = new Uint32Array(count);
  const order = new Uint32Array(count);
  for (let i = 0; i < count; i++) {
    const offset = i * MORPH_STRIDE;
    const x = Math.floor(((attributes[offset] - bounds.center[0]) / bounds.size + 0.5) * 1023);
    const y = Math.floor(((attributes[offset + 1] - bounds.center[1]) / bounds.size + 0.5) * 1023);
    const z = Math.floor(((attributes[offset + 2] - bounds.center[2]) / bounds.size + 0.5) * 1023);
    codes[i] = mortonCode(x, y, z);
    order[i] = i;
  }
  // An explicit index tie-break keeps clamped outliers and coincident splats stable.
  order.sort((a, b) => codes[a] - codes[b] || a - b);
  return order;
}

function particleSeed(seed: number, index: number): number {
  let hash = (seed | 0) ^ Math.imul(index + 1, 0x9e3779b9);
  hash = Math.imul(hash ^ (hash >>> 16), 0x85ebca6b);
  hash = Math.imul(hash ^ (hash >>> 13), 0xc2b2ae35);
  hash ^= hash >>> 16;
  // Exactly representable Float32 values in [0,1), including after transfer/upload.
  return (hash >>> 8) / 16777216;
}

function rankAt(index: number, count: number, pairedCount: number): number {
  return pairedCount === 1 ? 0 : Math.round(index * (count - 1) / (pairedCount - 1));
}

export function prepareMorph(a: Float32Array, b: Float32Array, seed: number): PreparedMorph {
  if (!Number.isFinite(seed)) throw new Error("Morph seed must be finite.");
  const boundsA = robustMorphBounds(a);
  const boundsB = robustMorphBounds(b);
  const orderA = mortonOrder(a, boundsA);
  const orderB = mortonOrder(b, boundsB);
  const count = Math.max(orderA.length, orderB.length);
  const pairedA = new Float32Array(count * MORPH_STRIDE);
  const pairedB = new Float32Array(count * MORPH_STRIDE);
  let previousA = -1;
  let previousB = -1;
  for (let i = 0; i < count; i++) {
    const rankA = rankAt(i, orderA.length, count);
    const rankB = rankAt(i, orderB.length, count);
    const sourceA = orderA[rankA] * MORPH_STRIDE;
    const sourceB = orderB[rankB] * MORPH_STRIDE;
    const offset = i * MORPH_STRIDE;
    for (let component = 0; component < MORPH_STRIDE; component++) {
      pairedA[offset + component] = a[sourceA + component];
      pairedB[offset + component] = b[sourceB + component];
    }
    // Every original appears once. Extra rank-neighbor copies contribute no
    // endpoint opacity, so unequal counts never duplicate visible originals.
    if (rankA === previousA) pairedA[offset + 3] = 0;
    if (rankB === previousB) pairedB[offset + 3] = 0;
    pairedA[offset + 7] = pairedB[offset + 7] = particleSeed(seed, i);
    previousA = rankA;
    previousB = rankB;
  }
  return {
    a: pairedA,
    b: pairedB,
    count,
    sceneRadius: Math.max(boundsA.radius, boundsB.radius, 1e-3),
  };
}
