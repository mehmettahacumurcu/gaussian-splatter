import { useEffect, useMemo, useRef, type RefObject } from "react";
import type { ThreeEvent } from "@react-three/fiber";
import * as THREE from "three";
import {
  SplatEdit,
  SplatEditRgbaBlendMode,
  SplatEditSdf,
  SplatEditSdfType,
  SplatMesh,
  dyno,
} from "@sparkjsdev/spark";
import { colorMatrix } from "./colorMath";
import type { ObjectRegistry } from "./registry";
import { COMPOSE_SPLAT_ENCODING, disposeSplatMeshOwn, packedSplatsCache } from "./splatCache";
import { CROP_BOX_HALF_EXTENT, isInsideCrop } from "./transformMath";
import { DEFAULT_COLOR, type CropBox, type SceneObject } from "./types";

interface Props {
  object: SceneObject;
  url: string;
  selected: boolean;
  registry: ObjectRegistry;
  /** True while a gizmo drag/click is in progress: clicks must not change selection. */
  gizmoBusy: RefObject<boolean>;
  onSelect: (id: string) => void;
  onError: (id: string, message: string) => void;
}

type SplatHit = Parameters<SplatMesh["raycast"]>[1][number];

const noRaycast: THREE.Object3D["raycast"] = () => {};

/** Pixels the pointer may travel between down and up and still count as a click. */
const CLICK_SLOP = 4;

/**
 * Splat with crop (mesh-local SplatEdit) and colour matrix (objectModifier).
 *
 * The SplatMesh sits under a group carrying the object transform, so the
 * crop helper (a child of the mesh) lives in the raw file frame, exactly like
 * the bake's crop.
 */
export function SplatObject({ object, url, selected, registry, gizmoBusy, onSelect, onError }: Props) {
  const groupRef = useRef<THREE.Group>(null);
  const colorMat = useMemo(() => new THREE.Matrix3(), []);
  const onErrorRef = useRef(onError);
  onErrorRef.current = onError;
  // Read by the raycast override, which runs outside React.
  const cropRef = useRef<CropBox | null>(object.crop ?? null);
  cropRef.current = object.crop ?? null;
  const visibleRef = useRef(object.visible);
  visibleRef.current = object.visible;

  const mesh = useMemo(() => {
    // The DynoMat3 keeps a reference to `colorMat`; mutating it in place and
    // bumping the mesh version makes Spark regenerate with the new value.
    const uniform = new dyno.DynoMat3({ value: colorMat });
    const modifier = dyno.dynoBlock({ gsplat: dyno.Gsplat }, { gsplat: dyno.Gsplat }, ({ gsplat }) => {
      if (!gsplat) throw new Error("gsplat input missing");
      const { rgb } = dyno.splitGsplat(gsplat).outputs;
      return { gsplat: dyno.combineGsplat({ gsplat, rgb: dyno.mul(uniform, rgb) }) };
    });
    // Splat data is shared per asset URL (see splatCache.ts); every mesh on a
    // shared PackedSplats must pass the same splatEncoding (SplatMesh sets it).
    const m = new SplatMesh({
      packedSplats: packedSplatsCache.acquire(url),
      splatEncoding: COMPOSE_SPLAT_ENCODING,
      objectModifier: modifier,
      editable: true,
    });
    // SplatMesh.raycast ignores edits: drop hits on cropped-away splats here so
    // that both clicks (R3F keeps only the nearest hit per object) and
    // snap-to-ground see the first *visible* splat.
    m.raycast = (raycaster, intersects) => {
      if (!visibleRef.current) return;
      const hits: SplatHit[] = [];
      SplatMesh.prototype.raycast.call(m, raycaster, hits);
      const crop = cropRef.current;
      for (const hit of hits) {
        if (crop) {
          const local = m.worldToLocal(hit.point.clone());
          if (!isInsideCrop(crop, [local.x, local.y, local.z])) continue;
        }
        intersects.push(hit);
      }
    };
    return m;
  }, [url, colorMat]);

  const crop = useMemo(() => {
    const edit = new SplatEdit({ rgbaBlendMode: SplatEditRgbaBlendMode.MULTIPLY, softEdge: 0, sdfSmooth: 0 });
    // Inverted box: everything OUTSIDE gets opacity × 0. SDF scale = half size
    // of a box of half-extent CROP_BOX_HALF_EXTENT (the SDF uses scale directly).
    const sdf = new SplatEditSdf({
      type: SplatEditSdfType.BOX,
      invert: true,
      opacity: 0,
      color: new THREE.Color(1, 1, 1),
    });
    const e = 2 * CROP_BOX_HALF_EXTENT;
    const outline = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(e, e, e)),
      new THREE.LineBasicMaterial({ color: 0xffb347 }),
    );
    outline.raycast = noRaycast;
    sdf.add(outline);
    edit.add(sdf);
    return { edit, sdf, outline };
  }, []);

  useEffect(
    () => () => {
      crop.outline.geometry.dispose();
      crop.outline.material.dispose();
    },
    [crop],
  );

  // Registry entry first: the crop effect below and the async bounds patch
  // both patch the entry this creates.
  useEffect(() => {
    const group = groupRef.current;
    if (!group) return;
    registry.set(object.id, { group, splat: mesh });
    return () => registry.delete(object.id, group);
  }, [object.id, mesh, registry]);

  useEffect(() => {
    const c = object.crop;
    if (!c) {
      crop.edit.removeFromParent();
      registry.patch(object.id, { cropTarget: undefined });
    } else {
      crop.sdf.position.set(...c.center);
      crop.sdf.quaternion.set(...c.quaternion);
      crop.sdf.scale.set(
        c.halfSize[0] / CROP_BOX_HALF_EXTENT,
        c.halfSize[1] / CROP_BOX_HALF_EXTENT,
        c.halfSize[2] / CROP_BOX_HALF_EXTENT,
      );
      if (crop.edit.parent !== mesh) mesh.add(crop.edit);
      registry.patch(object.id, { cropTarget: crop.sdf });
    }
    mesh.updateVersion();
  }, [object.crop, object.id, crop, mesh, registry]);

  useEffect(() => {
    let cancelled = false;
    mesh.initialized
      .then(() => {
        if (!cancelled) registry.patch(object.id, { localBounds: mesh.getBoundingBox(true) });
      })
      .catch((e: unknown) => {
        if (!cancelled) onErrorRef.current(object.id, e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, [mesh, object.id, registry]);

  // Primitives are never auto-disposed by R3F: free the mesh's own resources and
  // drop its reference to the shared splat data (freed with the last user).
  useEffect(
    () => () => {
      disposeSplatMeshOwn(mesh);
      packedSplatsCache.release(url);
    },
    [mesh, url],
  );

  useEffect(() => {
    const m = colorMatrix(object.color ?? DEFAULT_COLOR);
    // Matrix3.set takes row-major arguments, matching colorMatrix's layout.
    colorMat.set(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8]);
    mesh.updateVersion();
  }, [object.color, colorMat, mesh]);

  useEffect(() => {
    crop.outline.visible = selected;
  }, [selected, crop]);

  const handleClick = (e: ThreeEvent<MouseEvent>) => {
    // Orbit drag, gizmo interaction or hidden object: not a selection click.
    if (e.delta > CLICK_SLOP || gizmoBusy.current || !object.visible) return;
    e.stopPropagation();
    onSelect(object.id);
  };

  const t = object.transform;
  return (
    <group
      ref={groupRef}
      position={t.position}
      quaternion={t.quaternion}
      scale={t.scale}
      visible={object.visible}
      onClick={handleClick}
    >
      <primitive object={mesh} />
    </group>
  );
}
