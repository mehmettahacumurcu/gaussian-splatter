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
import { Vector3 } from "three";
import { SparkRenderer } from "@sparkjsdev/spark";
import { MeshObject } from "./MeshObject";
import type { ObjectRegistry } from "./registry";
import { SplatObject } from "./SplatObject";
import { CROP_BOX_HALF_EXTENT, orbitPositionForUp, readCrop, readTransform, uniformScaleFrom } from "./transformMath";
import type { CropBox, GizmoMode, SceneDoc, SceneObject, Transform, Vec3 } from "./types";
import { effectiveUp, upKey } from "./upVector";

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
 * Orbit controls that follow the scene up vector (any direction) without
 * remounting the Canvas (which would reload every asset): camera.up is set
 * first and the camera re-placed around the kept orbit target, then the
 * controls are recreated (OrbitControls reads camera.up only at construction).
 */
function OrbitRig({ up }: { up: Vec3 }) {
  const camera = useThree((s) => s.camera);
  const key = upKey(up);
  const [applied, setApplied] = useState<{ key: string; target: Vec3 } | null>(null);
  // Last mounted controls (kept after unmount so the new ones inherit the target).
  const lastControls = useRef<OrbitControlsImpl | null>(null);
  const keepControls = useCallback((c: OrbitControlsImpl | null) => {
    if (c) lastControls.current = c;
  }, []);

  useLayoutEffect(() => {
    const prev = lastControls.current;
    const target: Vec3 = prev ? [prev.target.x, prev.target.y, prev.target.z] : [0, 0, 0];
    const newUp = new Vector3(...up);
    if (camera.up.angleTo(newUp) > 1e-4) {
      const pos = orbitPositionForUp(
        [camera.position.x, camera.position.y, camera.position.z],
        target,
        [camera.up.x, camera.up.y, camera.up.z],
        up,
      );
      camera.position.set(...pos);
      camera.up.copy(newUp);
      camera.lookAt(target[0], target[1], target[2]);
    }
    setApplied({ key, target });
    // `key` stands for `up`'s value: a new array with the same numbers is no change.
  }, [key, camera]);

  if (!applied || applied.key !== key) return null;
  return <OrbitControls key={key} ref={keepControls} makeDefault enableDamping={false} target={applied.target} />;
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

/** Start 4 m back and 1.5 m up from the origin, relative to `up`. */
function initialCameraFor(up: Vec3): { position: Vec3; up: Vec3 } {
  return { position: orbitPositionForUp([0, 0, 4], [0, 0, 0], [0, 1, 0], up, Math.atan2(1.5, 4)), up };
}

/** Key light above the scene along `up`, slightly off-axis ([5, ±10, 5] for ±Y). */
function lightPositionFor(up: Vec3): Vec3 {
  return [5 + 10 * up[0], 10 * up[1], 5 + 10 * up[2]];
}

export function ComposeViewport(props: ViewportProps) {
  const { doc, active, selectedId, registry, assetUrl, onSelect, onError, onControls } = props;
  const gizmoBusy = useRef(false);
  const up = effectiveUp(doc);
  const upStr = upKey(up);
  // Initial camera only (R3F applies `camera` once per Canvas, i.e. per doc.id);
  // OrbitRig handles later up changes.
  const initialCamera = useMemo(() => initialCameraFor(up), [doc.id]);
  const lightPosition = useMemo(() => lightPositionFor(up), [upStr]);
  return (
    <Canvas
      key={doc.id}
      events={editorEvents}
      // Spark: antialias doesn't help splats and costs a lot. `flat` = no tone
      // mapping, so meshes match the (un-tone-mapped) splats.
      gl={{ antialias: false }}
      flat
      frameloop={active ? "always" : "never"}
      camera={{ ...initialCamera, fov: 60, near: 0.01, far: 2000 }}
      onPointerMissed={() => {
        if (!gizmoBusy.current) onSelect(null);
      }}
      style={{ width: "100%", height: "100%", background: "#101015" }}
    >
      <ambientLight intensity={0.7} />
      <directionalLight position={lightPosition} intensity={1.2} />
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
      <OrbitRig up={up} />
      <ControlsBridge onControls={onControls} />
      <GizmoBusyReset gizmoBusy={gizmoBusy} />
      <Gizmo {...props} gizmoBusy={gizmoBusy} />
    </Canvas>
  );
}
