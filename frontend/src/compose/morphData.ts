import { matchCoordinates, type Coordinates } from "./morphMatching.ts";
import { alignMorphCoordinates } from "./morphAlignment.ts";
import type { MorphMode } from "./morphTypes";

/** Four vec4s per particle: position/opacity, scale/seed, quaternion, RGB/sweep. */
export const MORPH_STRIDE = 16;

export interface PreparedMorph {
  a: Float32Array;
  b: Float32Array;
  count: number;
  sceneRadius: number;
  /** Mean world-space endpoint distance / larger robust bounding-box diagonal. */
  meanTravel: number;
  /** Mean residual in matching coordinates (after optional alignment). */
  matchingTravel: number;
  /** Worker-side PCA, sample search/refinement and coordinate transform time. */
  alignmentMs: number;
}

export interface MorphBounds {
  min: [number, number, number];
  max: [number, number, number];
  center: [number, number, number];
  /** Longest robust dimension, used for uniform (not per-axis) alignment. */
  size: number;
  radius: number;
}

/** In-place order statistic: linear expected work, without sorting the tail. */
function selectCoordinate(values: Float32Array, rank: number, start = 0): number {
  let left = start;
  let right = values.length - 1;
  while (left < right) {
    const middle = (left + right) >>> 1;
    const x = values[left], y = values[middle], z = values[right];
    const pivot = x < y ? (y < z ? y : Math.max(x, z)) : (x < z ? x : Math.max(y, z));
    let lo = left;
    let hi = right;
    while (lo <= hi) {
      while (values[lo] < pivot) lo++;
      while (values[hi] > pivot) hi--;
      if (lo <= hi) {
        const value = values[lo]; values[lo++] = values[hi]; values[hi--] = value;
      }
    }
    if (rank <= hi) right = hi;
    else if (rank >= lo) left = lo;
    else break;
  }
  return values[rank];
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
    min[axis] = selectCoordinate(coordinates, lowRank);
    max[axis] = selectCoordinate(coordinates, highRank, lowRank);
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

function normalizedCoordinates(attributes: Float32Array, bounds: MorphBounds): Coordinates {
  const count = attributes.length / MORPH_STRIDE;
  const result: Coordinates = [new Float32Array(count), new Float32Array(count), new Float32Array(count)];
  const inverseSize = 1 / bounds.size;
  const [x, y, z] = result;
  const [cx, cy, cz] = bounds.center;
  for (let i = 0; i < count; i++) {
    const offset = i * MORPH_STRIDE;
    x[i] = (attributes[offset] - cx) * inverseSize;
    y[i] = (attributes[offset + 1] - cy) * inverseSize;
    z[i] = (attributes[offset + 2] - cz) * inverseSize;
  }
  return result;
}

/**
 * Lockstep spatial bisection matches equal mass at every subdivision. Uniform
 * robust normalization affects matching only; all rendered endpoints retain
 * the input world transforms. A singleton leaf is equivalent to subdividing
 * its other side while borrowing invisible copies of its sole member.
 */
function prepareShapeMorph(a: Float32Array, b: Float32Array, seed: number, boundsA: MorphBounds, boundsB: MorphBounds, autoAlign: boolean): PreparedMorph {
  const countA = a.length / MORPH_STRIDE;
  const countB = b.length / MORPH_STRIDE;
  const count = Math.max(countA, countB);
  const pairedA = new Float32Array(count * MORPH_STRIDE);
  const pairedB = new Float32Array(count * MORPH_STRIDE);
  const alignmentStart = autoAlign ? performance.now() : 0;
  const aligned = autoAlign ? alignMorphCoordinates(a, b) : undefined;
  const alignmentMs = autoAlign ? performance.now() - alignmentStart : 0;
  const coordinatesA = aligned?.a ?? normalizedCoordinates(a, boundsA);
  const coordinatesB = aligned?.b ?? normalizedCoordinates(b, boundsB);
  const sweepWidth = boundsA.max[0] - boundsA.min[0];
  const inverseSweepWidth = sweepWidth > 1e-6 ? 1 / sweepWidth : 0;
  let outputIndex = 0;
  let totalTravel = 0;
  let totalMatchingTravel = 0;

  function emit(indexA: number, indexB: number, visibleA: boolean, visibleB: boolean): void {
    const sourceA = indexA * MORPH_STRIDE;
    const sourceB = indexB * MORPH_STRIDE;
    const offset = outputIndex * MORPH_STRIDE;
    totalTravel += Math.hypot(a[sourceA] - b[sourceB], a[sourceA + 1] - b[sourceB + 1], a[sourceA + 2] - b[sourceB + 2]);
    totalMatchingTravel += Math.hypot(coordinatesA[0][indexA] - coordinatesB[0][indexB], coordinatesA[1][indexA] - coordinatesB[1][indexB], coordinatesA[2][indexA] - coordinatesB[2][indexB]);
    for (let component = 0; component < MORPH_STRIDE; component++) {
      pairedA[offset + component] = a[sourceA + component];
      pairedB[offset + component] = b[sourceB + component];
    }
    if (!visibleA) pairedA[offset + 3] = 0;
    if (!visibleB) pairedB[offset + 3] = 0;
    pairedA[offset + 7] = pairedB[offset + 7] = particleSeed(seed, outputIndex);
    // Normalize the source's robust +X extent independently for sweep timing,
    // so even a narrow object reaches the end of the wave at t=1. A zero-width
    // source has a common final delay. This does not change matching geometry.
    const sweep = inverseSweepWidth === 0 ? 1 : Math.max(0, Math.min(1, (a[sourceA] - boundsA.min[0]) * inverseSweepWidth));
    pairedA[offset + 15] = pairedB[offset + 15] = sweep;
    outputIndex++;
  }

  matchCoordinates(coordinatesA, coordinatesB, emit);
  const sceneRadius = Math.max(boundsA.radius, boundsB.radius, 1e-3);
  return {
    a: pairedA, b: pairedB, count, sceneRadius,
    meanTravel: totalTravel / (count * 2 * sceneRadius),
    matchingTravel: totalMatchingTravel / count,
    alignmentMs,
  };
}

export function prepareMorph(a: Float32Array, b: Float32Array, seed: number, mode: MorphMode = "cloud", autoAlign = false): PreparedMorph {
  if (!Number.isFinite(seed)) throw new Error("Morph seed must be finite.");
  const boundsA = robustMorphBounds(a);
  const boundsB = robustMorphBounds(b);
  if (mode === "shape") return prepareShapeMorph(a, b, seed, boundsA, boundsB, autoAlign);
  const orderA = mortonOrder(a, boundsA);
  const orderB = mortonOrder(b, boundsB);
  const count = Math.max(orderA.length, orderB.length);
  const pairedA = new Float32Array(count * MORPH_STRIDE);
  const pairedB = new Float32Array(count * MORPH_STRIDE);
  let previousA = -1;
  let previousB = -1;
  let totalTravel = 0;
  let totalMatchingTravel = 0;
  for (let i = 0; i < count; i++) {
    const rankA = rankAt(i, orderA.length, count);
    const rankB = rankAt(i, orderB.length, count);
    const sourceA = orderA[rankA] * MORPH_STRIDE;
    const sourceB = orderB[rankB] * MORPH_STRIDE;
    const offset = i * MORPH_STRIDE;
    totalTravel += Math.hypot(a[sourceA] - b[sourceB], a[sourceA + 1] - b[sourceB + 1], a[sourceA + 2] - b[sourceB + 2]);
    totalMatchingTravel += Math.hypot(
      (a[sourceA] - boundsA.center[0]) / boundsA.size - (b[sourceB] - boundsB.center[0]) / boundsB.size,
      (a[sourceA + 1] - boundsA.center[1]) / boundsA.size - (b[sourceB + 1] - boundsB.center[1]) / boundsB.size,
      (a[sourceA + 2] - boundsA.center[2]) / boundsA.size - (b[sourceB + 2] - boundsB.center[2]) / boundsB.size,
    );
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
    meanTravel: totalTravel / (count * 2 * Math.max(boundsA.radius, boundsB.radius, 1e-3)),
    matchingTravel: totalMatchingTravel / count,
    alignmentMs: 0,
  };
}
