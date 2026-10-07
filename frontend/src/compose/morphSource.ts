import { Matrix3, Quaternion, Vector3 } from "three";
import type { PackedSplats } from "@sparkjsdev/spark";
import { colorMatrix } from "./colorMath";
import { MORPH_STRIDE } from "./morphData";
import { DEFAULT_COLOR, type SceneObject } from "./types";

/** Decode in short batches; Spark owns the decoded source, never transfer it. */
export async function extractMorphSource(packed: PackedSplats, object: SceneObject, signal: AbortSignal) {
  await packed.initialized;
  signal.throwIfAborted();
  const result = new Float32Array(packed.numSplats * MORPH_STRIDE);
  const rotation = new Quaternion(...object.transform.quaternion).normalize();
  const translation = new Vector3(...object.transform.position);
  const scale = object.transform.scale;
  const values = colorMatrix(object.color ?? DEFAULT_COLOR);
  const color = new Matrix3().set(...values as [number, number, number, number, number, number, number, number, number]);
  const rgb = new Vector3();
  const crop = object.crop;
  const cropRotation = crop && new Quaternion(...crop.quaternion).normalize().invert();
  const cropCenter = crop && new Vector3(...crop.center);
  const cropPoint = new Vector3();
  let count = 0;
  for (let start = 0; start < packed.numSplats; start += 4096) {
    signal.throwIfAborted();
    const end = Math.min(start + 4096, packed.numSplats);
    for (let i = start; i < end; i++) {
      const splat = packed.getSplat(i);
      if (crop && cropRotation && cropCenter) {
        cropPoint.copy(splat.center).sub(cropCenter).applyQuaternion(cropRotation);
        if (Math.abs(cropPoint.x) > crop.halfSize[0] || Math.abs(cropPoint.y) > crop.halfSize[1] ||
            Math.abs(cropPoint.z) > crop.halfSize[2]) continue;
      }
      splat.center.multiplyScalar(scale).applyQuaternion(rotation).add(translation);
      splat.scales.multiplyScalar(scale);
      splat.quaternion.premultiply(rotation).normalize();
      rgb.set(splat.color.r, splat.color.g, splat.color.b).applyMatrix3(color);
      const offset = count++ * MORPH_STRIDE;
      splat.center.toArray(result, offset);
      result[offset + 3] = object.visible ? splat.opacity : 0;
      splat.scales.toArray(result, offset + 4);
      splat.quaternion.toArray(result, offset + 8);
      rgb.toArray(result, offset + 12);
    }
    // Yield to input/paint and allow a pair change to cancel the extraction.
    if (end < packed.numSplats) await new Promise<void>((resolve) => setTimeout(resolve, 0));
  }
  signal.throwIfAborted();
  return result.subarray(0, count * MORPH_STRIDE);
}
