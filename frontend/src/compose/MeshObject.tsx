import { useEffect, useRef, useState, type RefObject } from "react";
import type { ThreeEvent } from "@react-three/fiber";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { MeshoptDecoder } from "three/examples/jsm/libs/meshopt_decoder.module.js";
import { gltfRoot } from "./gltfScene";
import type { ObjectRegistry } from "./registry";
import type { SceneObject } from "./types";

interface Props {
  object: SceneObject;
  url: string;
  registry: ObjectRegistry;
  /** True while a gizmo drag/click is in progress: clicks must not change selection. */
  gizmoBusy: RefObject<boolean>;
  onSelect: (id: string) => void;
  onError: (id: string, message: string) => void;
}

/** Pixels the pointer may travel between down and up and still count as a click. */
const CLICK_SLOP = 4;

const UNSUPPORTED_COMPRESSION = "Draco/KTX2-compressed GLB is not supported; re-export without compression";

function loadErrorMessage(e: unknown): string {
  const message = e instanceof Error ? e.message : String(e);
  // GLTFLoader: "No DRACOLoader instance provided." / "setKTX2Loader must be called ..."
  return /DRACO|KTX2/i.test(message) ? UNSUPPORTED_COMPRESSION : message;
}

function disposeMaterial(material: THREE.Material) {
  for (const value of Object.values(material)) {
    if (value instanceof THREE.Texture) value.dispose();
  }
  material.dispose();
}

function disposeTree(root: THREE.Object3D) {
  root.traverse((node) => {
    const mesh = node as THREE.Mesh;
    mesh.geometry?.dispose();
    const mats = Array.isArray(mesh.material) ? mesh.material : mesh.material ? [mesh.material] : [];
    mats.forEach(disposeMaterial);
  });
}

/** A GLB mesh under a group carrying the object transform. */
export function MeshObject({ object, url, registry, gizmoBusy, onSelect, onError }: Props) {
  const groupRef = useRef<THREE.Group>(null);
  const [scene, setScene] = useState<THREE.Group | null>(null);
  const onErrorRef = useRef(onError);
  onErrorRef.current = onError;
  // Read by the raycast overrides, which run outside React.
  const visibleRef = useRef(object.visible);
  visibleRef.current = object.visible;
  const id = object.id;

  useEffect(() => {
    let cancelled = false;
    let loaded: THREE.Group | null = null;
    new GLTFLoader()
      .setMeshoptDecoder(MeshoptDecoder)
      .loadAsync(url)
      .then(gltfRoot)
      .then((root) => {
        loaded = root;
        if (cancelled) {
          disposeTree(loaded);
          return;
        }
        // three's Raycaster ignores `visible`: hidden meshes must not take
        // clicks or snap hits.
        loaded.traverse((node) => {
          const original = node.raycast;
          node.raycast = function (this: THREE.Object3D, raycaster, intersects) {
            if (visibleRef.current) original.call(this, raycaster, intersects);
          };
        });
        setScene(loaded);
      })
      .catch((e: unknown) => {
        if (!cancelled) onErrorRef.current(id, loadErrorMessage(e));
      });
    return () => {
      cancelled = true;
      setScene(null);
      if (loaded) {
        loaded.removeFromParent();
        disposeTree(loaded);
      }
    };
  }, [url, id]);

  useEffect(() => {
    const group = groupRef.current;
    if (!group) return;
    registry.set(id, { group });
    return () => registry.delete(id, group);
  }, [id, registry]);

  const handleClick = (e: ThreeEvent<MouseEvent>) => {
    // Orbit drag, gizmo interaction or hidden object: not a selection click.
    if (e.delta > CLICK_SLOP || gizmoBusy.current || !object.visible) return;
    e.stopPropagation();
    onSelect(id);
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
      {scene && <primitive object={scene} />}
    </group>
  );
}
