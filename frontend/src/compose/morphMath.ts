import { MORPH_STRIDE } from "./morphData";

const MIN_SCALE = 1e-6;
const TAU = Math.PI * 2;
const clamp01 = (value: number) => Math.max(0, Math.min(1, value));
const mix = (a: number, b: number, t: number) => a * (1 - t) + b * t;

function normalizedQuaternion(attributes: Float32Array, offset: number): number[] {
  const quaternion = Array.from(attributes.subarray(offset + 8, offset + 12));
  const length = Math.hypot(...quaternion);
  return length > 1e-6 ? quaternion.map((component) => component / length) : [0, 0, 0, 1];
}

/** CPU reference for the GPU path below. Exact endpoints bypass all interpolation. */
export function interpolateMorphParticle(
  a: Float32Array,
  b: Float32Array,
  index: number,
  t: number,
  dissolve: number,
  targetBlend: number,
  sceneRadius: number,
): Float32Array {
  const offset = index * MORPH_STRIDE;
  if (!Number.isInteger(index) || index < 0 || offset + MORPH_STRIDE > a.length || offset + MORPH_STRIDE > b.length) {
    throw new Error("Morph particle index is out of range.");
  }
  if (![t, dissolve, targetBlend, sceneRadius].every(Number.isFinite)) {
    throw new Error("Morph interpolation parameters must be finite.");
  }
  const p = clamp01(t) * clamp01(targetBlend);
  if (p <= 0) return a.slice(offset, offset + MORPH_STRIDE);
  if (p >= 1) return b.slice(offset, offset + MORPH_STRIDE);
  const result = a.slice(offset, offset + MORPH_STRIDE);
  const u = p * p * (3 - 2 * p);
  const envelope = Math.sin(Math.PI * p) ** 2;
  const cloudWeight = clamp01(dissolve) * envelope;
  const radius = Math.max(sceneRadius, 0);
  const angle = a[offset + 7] * TAU;
  // Distinct high seed frequencies spread offsets throughout the cloud, while
  // the low progress frequencies keep each particle's trajectory smooth.
  const drift = [
    Math.sin(angle * 131.37 + p * TAU),
    Math.cos(angle * 173.11 - p * TAU * 0.75),
    Math.sin(angle * 219.73 + p * TAU * 1.25),
  ];
  const particleLogScale = Math.log(Math.max(radius * 0.0015, MIN_SCALE));
  for (let axis = 0; axis < 3; axis++) {
    result[axis] = mix(a[offset + axis], b[offset + axis], u)
      + drift[axis] / Math.sqrt(3) * radius * 1.5 * cloudWeight;
    const baseLogScale = mix(
      Math.log(Math.max(a[offset + 4 + axis], MIN_SCALE)),
      Math.log(Math.max(b[offset + 4 + axis], MIN_SCALE)),
      u,
    );
    result[4 + axis] = Math.exp(mix(baseLogScale, particleLogScale, cloudWeight));
    result[12 + axis] = mix(a[offset + 12 + axis], b[offset + 12 + axis], u);
  }
  result[3] = mix(a[offset + 3], b[offset + 3], u) * (1 - 0.25 * cloudWeight);

  const qa = normalizedQuaternion(a, offset);
  let qb = normalizedQuaternion(b, offset);
  let dot = qa.reduce((sum, component, i) => sum + component * qb[i], 0);
  if (dot < 0) {
    qb = qb.map((component) => -component);
    dot = -dot;
  }
  let quaternion: number[];
  if (dot > 0.9995) {
    quaternion = qa.map((component, i) => mix(component, qb[i], u));
  } else {
    const theta = Math.acos(clamp01(dot));
    const denominator = Math.sin(theta);
    quaternion = qa.map((component, i) => (
      component * Math.sin((1 - u) * theta) + qb[i] * Math.sin(u * theta)
    ) / denominator);
  }
  const length = Math.hypot(...quaternion);
  for (let component = 0; component < 4; component++) result[8 + component] = quaternion[component] / length;
  return result;
}

/** Keep this GLSL and the CPU reference above together when changing the effect. */
export const MORPH_GLSL = /* glsl */ `
vec4 morphNormalizeQuat(vec4 q) {
  float magnitude = length(q);
  return magnitude > 0.000001 ? q / magnitude : vec4(0.0, 0.0, 0.0, 1.0);
}

vec4 morphSlerp(vec4 a, vec4 b, float t) {
  a = morphNormalizeQuat(a);
  b = morphNormalizeQuat(b);
  float cosine = dot(a, b);
  if (cosine < 0.0) { b = -b; cosine = -cosine; }
  if (cosine > 0.9995) return normalize(mix(a, b, t));
  float angle = acos(clamp(cosine, 0.0, 1.0));
  return normalize((a * sin((1.0 - t) * angle) + b * sin(t * angle)) / sin(angle));
}

void interpolateMorph(
  in vec4 a0, in vec4 a1, in vec4 aq, in vec4 ac,
  in vec4 b0, in vec4 b1, in vec4 bq, in vec4 bc,
  in float t, in float dissolve, in float targetBlend, in float sceneRadius,
  out vec3 center, out vec3 scales, out vec4 quaternion, out vec4 rgba
) {
  float p = clamp(t, 0.0, 1.0) * clamp(targetBlend, 0.0, 1.0);
  if (p <= 0.0) {
    center = a0.xyz; scales = a1.xyz; quaternion = aq; rgba = vec4(ac.xyz, a0.w);
    return;
  }
  if (p >= 1.0) {
    center = b0.xyz; scales = b1.xyz; quaternion = bq; rgba = vec4(bc.xyz, b0.w);
    return;
  }
  float u = p * p * (3.0 - 2.0 * p);
  float envelope = sin(3.141592653589793 * p);
  float cloudWeight = clamp(dissolve, 0.0, 1.0) * envelope * envelope;
  float radius = max(sceneRadius, 0.0);
  float angle = a1.w * 6.283185307179586;
  vec3 drift = vec3(
    sin(angle * 131.37 + p * 6.283185307179586),
    cos(angle * 173.11 - p * 6.283185307179586 * 0.75),
    sin(angle * 219.73 + p * 6.283185307179586 * 1.25)
  ) * 0.5773502691896258;
  center = mix(a0.xyz, b0.xyz, u) + drift * radius * 1.5 * cloudWeight;
  vec3 baseLogScale = mix(log(max(a1.xyz, vec3(0.000001))), log(max(b1.xyz, vec3(0.000001))), u);
  float particleLogScale = log(max(radius * 0.0015, 0.000001));
  scales = exp(mix(baseLogScale, vec3(particleLogScale), cloudWeight));
  quaternion = morphSlerp(aq, bq, u);
  rgba = vec4(mix(ac.xyz, bc.xyz, u), mix(a0.w, b0.w, u) * (1.0 - 0.25 * cloudWeight));
}
`;
