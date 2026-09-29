import * as THREE from "three";
import type { GLTF } from "three/examples/jsm/loaders/GLTFLoader.js";

/**
 * The object tree to show for a loaded glTF.
 *
 * `scenes`/`scene` are optional in glTF 2.0; without them GLTFLoader returns
 * `scene: undefined, scenes: []`. Fall back to the first scene, then to a group
 * of the root nodes (nodes nobody lists as a child) — the same roots
 * `backend/compose/glb.py::wrap_with_transform` exports for such files.
 */
export async function gltfRoot(gltf: Pick<GLTF, "scene" | "scenes" | "parser">): Promise<THREE.Group> {
  if (gltf.scene) return gltf.scene;
  if (gltf.scenes?.[0]) return gltf.scenes[0];
  const nodes = (gltf.parser.json.nodes ?? []) as { children?: number[] }[];
  const children = new Set(nodes.flatMap((n) => n.children ?? []));
  const roots = nodes.map((_, i) => i).filter((i) => !children.has(i));
  const objects = (await Promise.all(roots.map((i) => gltf.parser.getDependency("node", i)))) as THREE.Object3D[];
  const group = new THREE.Group();
  for (const obj of objects) group.add(obj);
  return group;
}
