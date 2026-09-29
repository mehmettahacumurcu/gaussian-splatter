import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type RefObject } from "react";
import {
  Canvas,
  events as createPointerEvents,
  useFrame,
  useThree,
  type EventManager,
  type Events,
  type RootStore,
} from "@react-three/fiber";
import { OrbitControls, TransformControls } from "@react-three/drei";
import type { OrbitControls as OrbitControlsImpl, TransformControls as TransformControlsImpl } from "three-stdlib";
import type * as THREE from "three";
import { SparkRenderer } from "@sparkjsdev/spark";
import { MeshObject } from "./MeshObject";
import type { ObjectRegistry } from "./registry";
import { SplatObject } from "./SplatObject";
import { CROP_BOX_HALF_EXTENT, readCrop, readTransform, uniformScaleFrom } from "./transformMath";
import type { CropBox, GizmoMode, SceneDoc, SceneObject, Transform, Vec3, ViewUp } from "./types";

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
  /**
   * `(controls)` when new orbit controls are ready; `(null, released)` when
   * `released` goes away — ignore that unless `released` is still your current one.
   */
  onControls: (controls: OrbitControlsImpl | null, released?: OrbitControlsImpl) => void;
}

const noRaycast: THREE.Object3D["raycast"] = () => {};

/**
 * R3F's default event manager raycasts every interactive object on wheel,
 * pointerup, dblclick and contextmenu too, i.e. a full CPU splat raycast per
 * wheel tick. Selection only needs pointerdown (records initialClick /
 * initialHits, which gate onClick) and click. pointermove is kept because
 * `events.update()` calls it; it only raycasts objects with hover handlers
 * (R3F's filterPointerEvents) and the editor has none, so it raycasts nothing.
 * leave/cancel/lostpointercapture just clear hover/capture state (no raycast).
 */
function editorEvents(store: RootStore): EventManager<HTMLElement> {
  const base = createPointerEvents(store);
  const h = base.handlers as Events;
  const handlers: Partial<Events> = {
    onPointerDown: h.onPointerDown,
    onClick: h.onClick,
    onPointerMove: h.onPointerMove,
    onPointerLeave: h.onPointerLeave,
    onPointerCancel: h.onPointerCancel,
    onLostPointerCapture: h.onLostPointerCapture,
  };
  // connect()/disconnect() iterate store.events.handlers, i.e. this subset.
  return { ...base, handlers: handlers as Events };
}

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

/**
 * Orbit controls that follow `viewUp` without remounting the Canvas (which
 * would reload every asset): camera.up is set first, then the controls are
 * recreated (OrbitControls reads camera.up only at construction), keeping the
 * previous orbit target.
 */
function OrbitRig({ viewUp }: { viewUp: ViewUp }) {
  const camera = useThree((s) => s.camera);
  const [applied, setApplied] = useState<{ viewUp: ViewUp; target: Vec3 } | null>(null);
  // Last mounted controls (kept after unmount so the new ones inherit the target).
  const lastControls = useRef<OrbitControlsImpl | null>(null);
  const keepControls = useCallback((c: OrbitControlsImpl | null) => {
    if (c) lastControls.current = c;
  }, []);

  useLayoutEffect(() => {
    const sign = viewUp === "-y" ? -1 : 1;
    const prev = lastControls.current;
    const target: Vec3 = prev ? [prev.target.x, prev.target.y, prev.target.z] : [0, 0, 0];
    if (camera.up.y !== sign) {
      // Mirror the camera about the target's horizontal plane so the same
      // elevation is kept relative to the new up.
      camera.position.y = 2 * target[1] - camera.position.y;
      camera.up.set(0, sign, 0);
      camera.lookAt(target[0], target[1], target[2]);
    }
    setApplied({ viewUp, target });
  }, [viewUp, camera]);

  if (!applied || applied.viewUp !== viewUp) return null;
  return (
    <OrbitControls key={viewUp} ref={keepControls} makeDefault enableDamping={false} target={applied.target} />
  );
}

function ControlsBridge({ onControls }: { onControls: ViewportProps["onControls"] }) {
  const controls = useThree((s) => s.controls) as OrbitControlsImpl | null;
  const onControlsRef = useRef(onControls);
  onControlsRef.current = onControls;
  useEffect(() => {
    if (!controls) return;
    onControlsRef.current(controls);
    return () => onControlsRef.current(null, controls);
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

function gizmoTarget(obj: SceneObject | undefined, isCrop: boolean, registry: ObjectRegistry): THREE.Object3D | null {
  if (!obj) return null;
  const entry = registry.get(obj.id);
  if (isCrop) return entry?.cropTarget ?? null;
  return obj.role !== "base" ? (entry?.group ?? null) : null;
}

type GizmoProps = Pick<
  ViewportProps,
  "doc" | "selectedId" | "gizmoMode" | "cropEditing" | "registry" | "onTransformCommit" | "onCropCommit"
> & { gizmoBusy: RefObject<boolean> };

/**
 * `gizmoBusy` is set when a gizmo drag starts (so the click ending it doesn't
 * change selection) and cleared at the start of the next gesture: a
 * capture-phase document pointerdown runs before the canvas listeners
 * (TransformControls sets the flag again if that gesture grabs the gizmo).
 */
function GizmoBusyReset({ gizmoBusy }: { gizmoBusy: RefObject<boolean> }) {
  const doc = useThree((s) => s.gl.domElement.ownerDocument);
  useEffect(() => {
    const reset = () => {
      gizmoBusy.current = false;
    };
    doc.addEventListener("pointerdown", reset, true);
    return () => doc.removeEventListener("pointerdown", reset, true);
  }, [doc, gizmoBusy]);
  return null;
}

interface ActiveDrag {
  target: THREE.Object3D;
  obj: SceneObject;
  isCrop: boolean;
}

type GizmoControlsProps = Pick<GizmoProps, "gizmoMode" | "onTransformCommit" | "onCropCommit" | "gizmoBusy"> & {
  target: THREE.Object3D;
  obj: SceneObject;
  isCrop: boolean;
};

/** TransformControls for one target; mounted per target (keyed by the caller). */
function GizmoControls({
  target,
  obj,
  isCrop,
  gizmoMode,
  onTransformCommit,
  onCropCommit,
  gizmoBusy,
}: GizmoControlsProps) {
  const controlsRef = useRef<TransformControlsImpl>(null);
  const dragRef = useRef<ActiveDrag | null>(null);
  const getState = useThree((s) => s.get);

  useEffect(() => {
    // drei creates the three-stdlib controls in useMemo and never disposes
    // them; dispose() removes their canvas/document pointer listeners.
    const controls = controlsRef.current;
    return () => {
      // Detached mid-drag (selection/target change, delete…): three-stdlib
      // only emits mouseUp while attached, so no commit happened. Put the
      // object back to the doc values and re-enable the orbit controls that
      // drei disabled on 'dragging-changed' (its listener is already gone).
      const drag = dragRef.current;
      dragRef.current = null;
      if (drag) {
        delete drag.target.userData.scaleAtStart;
        restoreTarget(drag.target, drag.obj, drag.isCrop);
        const orbit = getState().controls as { enabled?: boolean } | null;
        if (orbit) orbit.enabled = true;
      }
      controls?.dispose();
    };
  }, [getState]);

  return (
    <TransformControls
      ref={controlsRef}
      object={target}
      mode={gizmoMode}
      space={isCrop ? "local" : "world"}
      onMouseDown={() => {
        gizmoBusy.current = true;
        dragRef.current = { target, obj, isCrop };
        target.userData.scaleAtStart = obj.transform.scale;
      }}
      onObjectChange={() => {
        if (!isCrop && gizmoMode === "scale") {
          const prev = (target.userData.scaleAtStart as number | undefined) ?? obj.transform.scale;
          target.scale.setScalar(uniformScaleFrom(prev, target.scale));
        }
      }}
      onMouseUp={() => {
        dragRef.current = null;
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

function Gizmo({
  doc,
  selectedId,
  gizmoMode,
  cropEditing,
  registry,
  onTransformCommit,
  onCropCommit,
  gizmoBusy,
}: GizmoProps) {
  const obj = selectedId ? doc.objects.find((o) => o.id === selectedId) : undefined;
  const isCrop = !!obj && cropEditing && !!obj.crop;
  const [target, setTarget] = useState<THREE.Object3D | null>(null);

  // Registry entries appear asynchronously (after mount / load); re-check each frame.
  useFrame(() => {
    const next = gizmoTarget(obj, isCrop, registry);
    if (next !== target) setTarget(next);
  });

  // The state lags one frame behind selection / mode changes: never attach to a stale target.
  if (!obj || !target || target !== gizmoTarget(obj, isCrop, registry)) return null;
  return (
    <GizmoControls
      key={target.uuid}
      target={target}
      obj={obj}
      isCrop={isCrop}
      gizmoMode={gizmoMode}
      onTransformCommit={onTransformCommit}
      onCropCommit={onCropCommit}
      gizmoBusy={gizmoBusy}
    />
  );
}

export function ComposeViewport(props: ViewportProps) {
  const { doc, active, selectedId, registry, assetUrl, onSelect, onError, onControls } = props;
  const gizmoBusy = useRef(false);
  // Initial camera only (R3F applies `camera` once); OrbitRig handles later viewUp changes.
  const upSign = doc.viewUp === "-y" ? -1 : 1;
  return (
    <Canvas
      key={doc.id}
      events={editorEvents}
      // Spark: antialias doesn't help splats and costs a lot. `flat` = no tone
      // mapping, so meshes match the (un-tone-mapped) splats.
      gl={{ antialias: false }}
      flat
      frameloop={active ? "always" : "never"}
      camera={{ position: [0, 1.5 * upSign, 4], up: [0, upSign, 0], fov: 60, near: 0.01, far: 2000 }}
      onPointerMissed={() => {
        if (!gizmoBusy.current) onSelect(null);
      }}
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
            gizmoBusy={gizmoBusy}
            onSelect={onSelect}
            onError={onError}
          />
        ) : (
          <MeshObject
            key={o.id}
            object={o}
            url={assetUrl(o.asset)}
            registry={registry}
            gizmoBusy={gizmoBusy}
            onSelect={onSelect}
            onError={onError}
          />
        ),
      )}
      <OrbitRig viewUp={doc.viewUp} />
      <ControlsBridge onControls={onControls} />
      <GizmoBusyReset gizmoBusy={gizmoBusy} />
      <Gizmo {...props} gizmoBusy={gizmoBusy} />
    </Canvas>
  );
}
