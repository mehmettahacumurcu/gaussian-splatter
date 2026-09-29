import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, TransformControls } from "@react-three/drei";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import type * as THREE from "three";
import { SparkRenderer } from "@sparkjsdev/spark";
import { MeshObject } from "./MeshObject";
import type { ObjectRegistry } from "./registry";
import { SplatObject } from "./SplatObject";
import { CROP_BOX_HALF_EXTENT, readCrop, readTransform, uniformScaleFrom } from "./transformMath";
import type { CropBox, GizmoMode, SceneDoc, SceneObject, Transform } from "./types";

interface ViewportProps {
  doc: SceneDoc;
  active: boolean;
  selectedId: string | null;
  gizmoMode: GizmoMode;
  cropEditing: boolean;
  registry: ObjectRegistry;
  assetUrl: (assetId: string) => string;
  onSelect: (id: string | null) => void;
  onTransformCommit: (id: string, t: Transform) => void;
  onCropCommit: (id: string, crop: CropBox) => void;
  onError: (id: string, message: string) => void;
  onControls: (controls: OrbitControlsImpl | null) => void;
}

const noRaycast: THREE.Object3D["raycast"] = () => {};

function SparkLayer() {
  const gl = useThree((s) => s.gl);
  // SparkRenderer 0.1.10 has no dispose(); its GPU resources go with the
  // WebGL context when the Canvas unmounts.
  const spark = useMemo(() => {
    const s = new SparkRenderer({ renderer: gl });
    s.raycast = noRaycast;
    return s;
  }, [gl]);
  return <primitive object={spark} />;
}

function ControlsBridge({ onControls }: { onControls: ViewportProps["onControls"] }) {
  const controls = useThree((s) => s.controls) as OrbitControlsImpl | null;
  const onControlsRef = useRef(onControls);
  onControlsRef.current = onControls;
  useEffect(() => {
    onControlsRef.current(controls);
    return () => onControlsRef.current(null);
  }, [controls]);
  return null;
}

/** Put the gizmo target back to the committed doc values (after a rejected edit). */
function restoreTarget(target: THREE.Object3D, obj: SceneObject, isCrop: boolean) {
  if (isCrop && obj.crop) {
    const c = obj.crop;
    target.position.set(...c.center);
    target.quaternion.set(...c.quaternion);
    target.scale.set(
      c.halfSize[0] / CROP_BOX_HALF_EXTENT,
      c.halfSize[1] / CROP_BOX_HALF_EXTENT,
      c.halfSize[2] / CROP_BOX_HALF_EXTENT,
    );
  } else {
    const t = obj.transform;
    target.position.set(...t.position);
    target.quaternion.set(...t.quaternion);
    target.scale.setScalar(t.scale);
  }
}

type GizmoProps = Pick<
  ViewportProps,
  "doc" | "selectedId" | "gizmoMode" | "cropEditing" | "registry" | "onTransformCommit" | "onCropCommit"
>;

function Gizmo({ doc, selectedId, gizmoMode, cropEditing, registry, onTransformCommit, onCropCommit }: GizmoProps) {
  const obj = selectedId ? doc.objects.find((o) => o.id === selectedId) : undefined;
  const isCrop = !!obj && cropEditing && !!obj.crop;
  const [target, setTarget] = useState<THREE.Object3D | null>(null);

  // Registry entries appear asynchronously (after mount / load); re-check each frame.
  useFrame(() => {
    let next: THREE.Object3D | null = null;
    if (obj) {
      const entry = registry.get(obj.id);
      if (isCrop) next = entry?.cropTarget ?? null;
      else if (obj.role !== "base") next = entry?.group ?? null;
    }
    if (next !== target) setTarget(next);
  });

  if (!obj || !target) return null;
  return (
    <TransformControls
      object={target}
      mode={gizmoMode}
      space={isCrop ? "local" : "world"}
      onMouseDown={() => {
        target.userData.scaleAtStart = obj.transform.scale;
      }}
      onObjectChange={() => {
        if (!isCrop && gizmoMode === "scale") {
          const prev = (target.userData.scaleAtStart as number | undefined) ?? obj.transform.scale;
          target.scale.setScalar(uniformScaleFrom(prev, target.scale));
        }
      }}
      onMouseUp={() => {
        delete target.userData.scaleAtStart;
        try {
          if (isCrop) onCropCommit(obj.id, readCrop(target));
          else onTransformCommit(obj.id, readTransform(target, obj.transform.scale));
        } catch {
          // Non-finite values from the gizmo: keep the doc as is.
          restoreTarget(target, obj, isCrop);
        }
      }}
    />
  );
}

export function ComposeViewport(props: ViewportProps) {
  const { doc, active, selectedId, registry, assetUrl, onSelect, onError, onControls } = props;
  const upSign = doc.viewUp === "-y" ? -1 : 1;
  return (
    <Canvas
      // Remount on viewUp change: OrbitControls reads camera.up only at construction.
      key={`${doc.id}:${doc.viewUp}`}
      frameloop={active ? "always" : "never"}
      camera={{ position: [0, 1.5 * upSign, 4], up: [0, upSign, 0], fov: 60, near: 0.01, far: 2000 }}
      onPointerMissed={() => onSelect(null)}
      style={{ width: "100%", height: "100%", background: "#101015" }}
    >
      <ambientLight intensity={0.7} />
      <directionalLight position={[5, 10 * upSign, 5]} intensity={1.2} />
      <SparkLayer />
      {doc.objects.map((o) =>
        o.kind === "splat" ? (
          <SplatObject
            key={o.id}
            object={o}
            url={assetUrl(o.asset)}
            selected={o.id === selectedId}
            registry={registry}
            onSelect={onSelect}
            onError={onError}
          />
        ) : (
          <MeshObject
            key={o.id}
            object={o}
            url={assetUrl(o.asset)}
            registry={registry}
            onSelect={onSelect}
            onError={onError}
          />
        ),
      )}
      <OrbitControls makeDefault enableDamping={false} />
      <ControlsBridge onControls={onControls} />
      <Gizmo {...props} />
    </Canvas>
  );
}
