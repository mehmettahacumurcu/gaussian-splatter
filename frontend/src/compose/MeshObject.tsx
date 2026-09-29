import { useEffect, useRef, useState } from "react";
import type { ThreeEvent } from "@react-three/fiber";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import type { ObjectRegistry } from "./registry";
import type { SceneObject } from "./types";

interface Props {
  object: SceneObject;
  url: string;
  registry: ObjectRegistry;
  onSelect: (id: string) => void;
  onError: (id: string, message: string) => void;
}

/** Pixels the pointer may travel between down and up and still count as a click. */
const CLICK_SLOP = 4;

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
export function MeshObject({ object, url, registry, onSelect, onError }: Props) {
  const groupRef = useRef<THREE.Group>(null);
  const [scene, setScene] = useState<THREE.Group | null>(null);
  const onErrorRef = useRef(onError);
  onErrorRef.current = onError;
  const id = object.id;

  useEffect(() => {
    let cancelled = false;
    let loaded: THREE.Group | null = null;
    new GLTFLoader()
      .loadAsync(url)
      .then((gltf) => {
        loaded = gltf.scene;
        if (cancelled) disposeTree(loaded);
        else setScene(loaded);
      })
      .catch((e: unknown) => {
        if (!cancelled) onErrorRef.current(id, e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
      setScene(null);
      if (loaded) disposeTree(loaded);
    };
  }, [url, id]);

  useEffect(() => {
    const group = groupRef.current;
    if (!group) return;
    registry.set(id, { group });
    return () => registry.delete(id, group);
  }, [id, registry]);

  const handleClick = (e: ThreeEvent<MouseEvent>) => {
    if (e.delta > CLICK_SLOP || !object.visible) return; // orbit drag / hidden object
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
