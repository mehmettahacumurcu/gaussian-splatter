import { MORPH_STRIDE } from "../../morphData.ts";
import { noisySphere } from "./morphSpheres.ts";

/** A lopsided, anisotropic shell with distinctive proper principal axes. */
export function asymmetricCloud(count: number, seed = 12345): Float32Array {
  const attributes = noisySphere(count, seed);
  for (let i = 0; i < count; i++) {
    const offset = i * MORPH_STRIDE;
    const x = attributes[offset], y = attributes[offset + 1], z = attributes[offset + 2];
    attributes[offset] = 2 * x + 0.8 * x * x + 0.2 * y * z;
    attributes[offset + 1] = y * (0.7 + 0.3 * x) + 0.1 * z * z;
    attributes[offset + 2] = z * (0.35 + 0.2 * y);
    attributes[offset + 12] = i;
  }
  return attributes;
}

/** Seeded arbitrary SO(3) rotation, optional uniform scale/translation/permutation. */
export function rotatedCloud(source: Float32Array, seed: number, scale = 1, translate = 0, shuffle = false): Float32Array {
  let state = seed >>> 0;
  const random = () => ((state = (Math.imul(state, 1664525) + 1013904223) >>> 0) / 4294967296);
  const u = random(), v = 2 * Math.PI * random(), w = 2 * Math.PI * random();
  const qx = Math.sqrt(1 - u) * Math.sin(v), qy = Math.sqrt(1 - u) * Math.cos(v);
  const qz = Math.sqrt(u) * Math.sin(w), qw = Math.sqrt(u) * Math.cos(w);
  const r = [
    1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw),
    2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw),
    2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy),
  ];
  const count = source.length / MORPH_STRIDE;
  const order = Uint32Array.from({ length: count }, (_, i) => i);
  if (shuffle) {
    for (let i = count - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
  }
  const output = new Float32Array(source.length);
  for (let i = 0; i < count; i++) {
    const from = order[i] * MORPH_STRIDE, offset = i * MORPH_STRIDE;
    output.set(source.subarray(from, from + MORPH_STRIDE), offset);
    const x = source[from], y = source[from + 1], z = source[from + 2];
    for (let axis = 0; axis < 3; axis++) output[offset + axis] = scale * (r[3 * axis] * x + r[3 * axis + 1] * y + r[3 * axis + 2] * z) + translate * (axis + 1);
    // Each endpoint's own id permits exact original-attribute assertions.
    output[offset + 12] = i;
  }
  return output;
}
