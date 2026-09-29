import * as THREE from "three";
import { LN_SCALE_MAX, LN_SCALE_MIN, PackedSplats, type PackedSplatsOptions, type SplatMesh } from "@sparkjsdev/spark";

type SplatEncoding = NonNullable<PackedSplatsOptions["splatEncoding"]>;

/**
 * Base-colour range for composer splats.
 *
 * Spark packs the DC colour (0.5 + C0·f_dc) into 8 bits clamped to
 * [rgbMin, rgbMax] *before* the objectModifier colour matrix runs, while the
 * export bake applies the matrix to unclamped colours. Spark's default [0, 1]
 * would therefore make e.g. exposure −1 on an over-bright highlight look dimmer
 * in the preview than in the export. [−0.5, 1.5] covers what 3DGS training
 * typically produces; the cost is an 8-bit step of 2/255 instead of 1/255
 * (the final render is still re-packed and clamped to [0, 1] by SparkRenderer).
 * The rest matches Spark's DEFAULT_SPLAT_ENCODING (not exported).
 *
 * SplatMesh overwrites `packedSplats.splatEncoding` with its own
 * `splatEncoding` option, so every SplatMesh sharing a PackedSplats from
 * `packedSplatsCache` must pass this same object.
 */
export const COMPOSE_SPLAT_ENCODING: SplatEncoding = Object.freeze({
  rgbMin: -0.5,
  rgbMax: 1.5,
  lnScaleMin: LN_SCALE_MIN,
  lnScaleMax: LN_SCALE_MAX,
  sh1Min: -1,
  sh1Max: 1,
  sh2Min: -1,
  sh2Max: 1,
  sh3Min: -1,
  sh3Max: 1,
});

/** Values shared by key, loaded on first acquire and disposed when the last user releases. */
export class RefCountedCache<K, V> {
  private readonly entries = new Map<K, { value: V; refs: number }>();

  constructor(
    private readonly load: (key: K) => V,
    private readonly dispose: (value: V) => void,
  ) {}

  acquire(key: K): V {
    let entry = this.entries.get(key);
    if (!entry) {
      entry = { value: this.load(key), refs: 0 };
      this.entries.set(key, entry);
    }
    entry.refs += 1;
    return entry.value;
  }

  /** Drops one reference; unknown keys are ignored. */
  release(key: K): void {
    const entry = this.entries.get(key);
    if (!entry) return;
    entry.refs -= 1;
    if (entry.refs > 0) return;
    this.entries.delete(key);
    this.dispose(entry.value);
  }

  refCount(key: K): number {
    return this.entries.get(key)?.refs ?? 0;
  }

  get size(): number {
    return this.entries.size;
  }
}

/**
 * PackedSplats.dispose() frees the packed splat textures only; also free the SH
 * textures (extra.sh{1,2,3}Texture, DynoUsampler2DArray wrappers that
 * SplatMesh.ensureShTextures caches on the PackedSplats, so shared too).
 */
export function disposePackedSplats(packed: PackedSplats) {
  const extra = packed.extra as Record<string, { value?: unknown } | undefined> | undefined;
  for (const key of ["sh1Texture", "sh2Texture", "sh3Texture"]) {
    const tex = extra?.[key]?.value;
    if (tex instanceof THREE.Texture) tex.dispose();
  }
  packed.dispose();
}

/**
 * Frees what a SplatMesh owns by itself: the edit SDF texture (private
 * SplatMesh.rgbaDisplaceEdits.sdfTexture). Deliberately does NOT call
 * SplatMesh.dispose(), which would dispose the shared PackedSplats.
 */
export function disposeSplatMeshOwn(mesh: SplatMesh) {
  const edits = (mesh as unknown as { rgbaDisplaceEdits?: { sdfTexture?: THREE.Texture } | null }).rgbaDisplaceEdits;
  edits?.sdfTexture?.dispose();
}

/**
 * One decoded PackedSplats per asset URL, shared by every SplatMesh showing
 * that asset (duplicates, the same asset added twice): the PLY is downloaded,
 * decoded and uploaded to the GPU once. Per-mesh state (transform, crop edits,
 * colour objectModifier, recolor) lives on each SplatMesh, not here.
 */
export const packedSplatsCache = new RefCountedCache<string, PackedSplats>(
  (url) => new PackedSplats({ url, splatEncoding: COMPOSE_SPLAT_ENCODING }),
  disposePackedSplats,
);
