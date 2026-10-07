import { matchCoordinates, type Coordinates } from "./morphMatching.ts";

/** A fixed cap keeps the 24-candidate search independent of asset size. */
export const ALIGNMENT_SAMPLE_LIMIT = 1024;
const STRIDE = 16;
type Matrix = Float64Array;

interface PrincipalCloud {
  coordinates: Coordinates;
  axes: Matrix;
  spectrum: number[];
}

/** Small, deterministic Jacobi eigensolver; eigenvectors occupy matrix columns. */
function eigenSymmetric(input: Matrix, n: number): { values: number[]; vectors: Matrix } {
  const a = input.slice();
  const vectors = new Float64Array(n * n);
  for (let i = 0; i < n; i++) vectors[i * n + i] = 1;
  const tolerance = Math.max(1e-30, Array.from({ length: n }, (_, i) => Math.abs(a[i * n + i])).reduce((x, y) => x + y, 0) * 1e-14);
  for (let iteration = 0; iteration < 64; iteration++) {
    let p = 0, q = 1, largest = 0;
    for (let row = 0; row < n; row++) {
      for (let col = row + 1; col < n; col++) {
        const value = Math.abs(a[row * n + col]);
        if (value > largest) { largest = value; p = row; q = col; }
      }
    }
    if (largest <= tolerance) break;
    const app = a[p * n + p], aqq = a[q * n + q], apq = a[p * n + q];
    const angle = 0.5 * Math.atan2(2 * apq, aqq - app);
    const c = Math.cos(angle), s = Math.sin(angle);
    for (let row = 0; row < n; row++) {
      if (row !== p && row !== q) {
        const arp = a[row * n + p], arq = a[row * n + q];
        a[row * n + p] = a[p * n + row] = c * arp - s * arq;
        a[row * n + q] = a[q * n + row] = s * arp + c * arq;
      }
      const vrp = vectors[row * n + p], vrq = vectors[row * n + q];
      vectors[row * n + p] = c * vrp - s * vrq;
      vectors[row * n + q] = s * vrp + c * vrq;
    }
    a[p * n + p] = c * c * app - 2 * c * s * apq + s * s * aqq;
    a[q * n + q] = s * s * app + 2 * c * s * apq + c * c * aqq;
    a[p * n + q] = a[q * n + p] = 0;
  }
  return { values: Array.from({ length: n }, (_, i) => a[i * n + i]), vectors };
}

function determinant(m: Matrix): number {
  return m[0] * (m[4] * m[8] - m[5] * m[7])
    - m[1] * (m[3] * m[8] - m[5] * m[6])
    + m[2] * (m[3] * m[7] - m[4] * m[6]);
}

/**
 * With alignment enabled, mean/2*RMS-radius normalization is rotation invariant.
 * Axis-aligned percentile bounds are not: using those before PCA would change
 * both center and scale when an otherwise identical asset rotates.
 */
function principalCloud(attributes: Float32Array): PrincipalCloud {
  const count = attributes.length / STRIDE;
  let cx = 0, cy = 0, cz = 0;
  for (let offset = 0; offset < attributes.length; offset += STRIDE) {
    cx += attributes[offset]; cy += attributes[offset + 1]; cz += attributes[offset + 2];
  }
  cx /= count; cy /= count; cz /= count;
  const covariance = new Float64Array(9);
  for (let offset = 0; offset < attributes.length; offset += STRIDE) {
    const x = attributes[offset] - cx, y = attributes[offset + 1] - cy, z = attributes[offset + 2] - cz;
    covariance[0] += x * x; covariance[1] += x * y; covariance[2] += x * z;
    covariance[4] += y * y; covariance[5] += y * z; covariance[8] += z * z;
  }
  covariance[3] = covariance[1]; covariance[6] = covariance[2]; covariance[7] = covariance[5];
  const trace = covariance[0] + covariance[4] + covariance[8];
  const inverseSize = 1 / Math.max(2 * Math.sqrt(trace / count), 1e-6);
  // Normalize covariance too, keeping eigensolver tolerances scale independent.
  for (let i = 0; i < 9; i++) covariance[i] /= Math.max(trace, 1e-30);
  const eigen = eigenSymmetric(covariance, 3);
  const order = [0, 1, 2].sort((a, b) => eigen.values[b] - eigen.values[a] || a - b);
  const axes = new Float64Array(9);
  for (let row = 0; row < 3; row++) for (let col = 0; col < 3; col++) axes[row * 3 + col] = eigen.vectors[row * 3 + order[col]];
  // Both bases are right handed; proper signed permutations remain rotations.
  if (determinant(axes) < 0) for (let row = 0; row < 3; row++) axes[row * 3 + 2] *= -1;
  const coordinates: Coordinates = [new Float32Array(count), new Float32Array(count), new Float32Array(count)];
  for (let i = 0; i < count; i++) {
    coordinates[0][i] = (attributes[i * STRIDE] - cx) * inverseSize;
    coordinates[1][i] = (attributes[i * STRIDE + 1] - cy) * inverseSize;
    coordinates[2][i] = (attributes[i * STRIDE + 2] - cz) * inverseSize;
  }
  return { coordinates, axes, spectrum: order.map((index) => eigen.values[index]) };
}

function sample(coordinates: Coordinates): Coordinates {
  const count = Math.min(ALIGNMENT_SAMPLE_LIMIT, coordinates[0].length);
  if (count === coordinates[0].length) return coordinates;
  const result: Coordinates = [new Float32Array(count), new Float32Array(count), new Float32Array(count)];
  for (let i = 0; i < count; i++) {
    const index = Math.floor(i * coordinates[0].length / count);
    for (let axis = 0; axis < 3; axis++) result[axis][i] = coordinates[axis][index];
  }
  return result;
}

function rotate(coordinates: Coordinates, rotation: Matrix): Coordinates {
  const count = coordinates[0].length;
  const result: Coordinates = [new Float32Array(count), new Float32Array(count), new Float32Array(count)];
  for (let i = 0; i < count; i++) {
    const x = coordinates[0][i], y = coordinates[1][i], z = coordinates[2][i];
    result[0][i] = rotation[0] * x + rotation[1] * y + rotation[2] * z;
    result[1][i] = rotation[3] * x + rotation[4] * y + rotation[5] * z;
    result[2][i] = rotation[6] * x + rotation[7] * y + rotation[8] * z;
  }
  return result;
}

function travel(a: Coordinates, b: Coordinates): number {
  let distance = 0;
  matchCoordinates(a, b, (ia, ib) => {
    distance += Math.hypot(a[0][ia] - b[0][ib], a[1][ia] - b[1][ib], a[2][ia] - b[2][ib]);
  });
  return distance / Math.max(a[0].length, b[0].length);
}

/** Quaternion form of proper orthogonal Procrustes (Kabsch/Horn), never a reflection. */
function fittedRotation(a: Coordinates, originalB: Coordinates, rotatedB: Coordinates): Matrix {
  const s = new Float64Array(9);
  matchCoordinates(a, rotatedB, (ia, ib) => {
    for (let row = 0; row < 3; row++) for (let col = 0; col < 3; col++) s[row * 3 + col] += originalB[row][ib] * a[col][ia];
  });
  const [xx, xy, xz, yx, yy, yz, zx, zy, zz] = s;
  const horn = new Float64Array([
    xx + yy + zz, yz - zy, zx - xz, xy - yx,
    yz - zy, xx - yy - zz, xy + yx, zx + xz,
    zx - xz, xy + yx, -xx + yy - zz, yz + zy,
    xy - yx, zx + xz, yz + zy, -xx - yy + zz,
  ]);
  const eigen = eigenSymmetric(horn, 4);
  let best = 0;
  for (let i = 1; i < 4; i++) if (eigen.values[i] > eigen.values[best]) best = i;
  const w = eigen.vectors[best], x = eigen.vectors[4 + best], y = eigen.vectors[8 + best], z = eigen.vectors[12 + best];
  return new Float64Array([
    1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
  ]);
}

export interface AlignedMorphCoordinates {
  a: Coordinates;
  b: Coordinates;
  /** Row-major proper rotation from normalized B to normalized A; matching only. */
  rotation: Matrix;
}

/**
 * Full-set PCA followed by 24 proper principal-axis candidates. All candidates
 * use the same bounded sample and full pairing's bisection cost. Up to three
 * rigid refinements are retained only if they lower mean sample travel.
 */
export function alignMorphCoordinates(a: Float32Array, b: Float32Array): AlignedMorphCoordinates {
  const principalA = principalCloud(a), principalB = principalCloud(b);
  const sampleA = sample(principalA.coordinates), sampleB = sample(principalB.coordinates);
  let bestRotation = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  let bestTravel = Infinity;
  const permutations = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
  for (const permutation of permutations) {
    let inversions = 0;
    for (let i = 0; i < 3; i++) for (let j = i + 1; j < 3; j++) if (permutation[i] > permutation[j]) inversions++;
    const parity = inversions % 2 === 0 ? 1 : -1;
    for (const sx of [1, -1]) for (const sy of [1, -1]) {
      const signs = [sx, sy, parity * sx * sy];
      const rotation = new Float64Array(9);
      for (let row = 0; row < 3; row++) {
        for (let col = 0; col < 3; col++) {
          for (let axis = 0; axis < 3; axis++) rotation[row * 3 + col] += principalA.axes[row * 3 + permutation[axis]] * signs[axis] * principalB.axes[col * 3 + axis];
        }
      }
      const distance = travel(sampleA, rotate(sampleB, rotation));
      if (distance < bestTravel) { bestTravel = distance; bestRotation = rotation; }
    }
  }
  // Equal spectra describe a possible rigid copy. PCA is already exact for a
  // nondegenerate copy; fitting independently sampled correspondences would
  // introduce sampling drift. Keep that exact solution instead of refining it.
  const matchingSpectra = principalA.spectrum.every((value, i) => Math.abs(value - principalB.spectrum[i]) < 1e-6);
  if (!matchingSpectra) {
    for (let iteration = 0; iteration < 3; iteration++) {
      const rotatedB = rotate(sampleB, bestRotation);
      const rotation = fittedRotation(sampleA, sampleB, rotatedB);
      const distance = travel(sampleA, rotate(sampleB, rotation));
      if (distance >= bestTravel - 1e-8) break;
      bestTravel = distance; bestRotation = rotation;
    }
  }
  return { a: principalA.coordinates, b: rotate(principalB.coordinates, bestRotation), rotation: bestRotation };
}
