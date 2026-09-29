import { describe, expect, it } from "vitest";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { gltfRoot } from "../gltfScene";

/** A GLB with only a JSON chunk (no BIN): enough for node-only files. */
function minimalGlb(gltf: Record<string, unknown>): ArrayBuffer {
  let json = new TextEncoder().encode(JSON.stringify({ asset: { version: "2.0" }, ...gltf }));
  const padded = Math.ceil(json.length / 4) * 4;
  if (padded !== json.length) {
    const p = new Uint8Array(padded).fill(0x20);
    p.set(json);
    json = p;
  }
  const out = new ArrayBuffer(12 + 8 + json.length);
  const view = new DataView(out);
  view.setUint32(0, 0x46546c67, true); // "glTF"
  view.setUint32(4, 2, true);
  view.setUint32(8, out.byteLength, true);
  view.setUint32(12, json.length, true);
  view.setUint32(16, 0x4e4f534a, true); // "JSON"
  new Uint8Array(out, 20).set(json);
  return out;
}

const NODES = [{ name: "a", children: [1] }, { name: "b" }, { name: "c" }];

describe("gltfRoot", () => {
  it("returns the default scene when the file has one", async () => {
    const gltf = await new GLTFLoader().parseAsync(minimalGlb({ nodes: NODES, scenes: [{ nodes: [2] }], scene: 0 }), "");
    const root = await gltfRoot(gltf);
    expect(root).toBe(gltf.scene);
    expect(root.children.map((c) => c.name)).toEqual(["c"]);
  });

  it("falls back to a group of the root nodes when the file has no scenes", async () => {
    const gltf = await new GLTFLoader().parseAsync(minimalGlb({ nodes: NODES }), "");
    expect(gltf.scene).toBeUndefined();
    const root = await gltfRoot(gltf);
    expect(root.children.map((c) => c.name)).toEqual(["a", "c"]);
    expect(root.children[0].children.map((c) => c.name)).toEqual(["b"]);
  });

  it("returns an empty group for a file without scenes or nodes", async () => {
    const gltf = await new GLTFLoader().parseAsync(minimalGlb({}), "");
    const root = await gltfRoot(gltf);
    expect(root.children).toEqual([]);
  });
});
