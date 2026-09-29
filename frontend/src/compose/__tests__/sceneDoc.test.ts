import { describe, expect, it } from "vitest";
import { composeReducer, initialComposeState, newObjectId } from "../sceneDoc";
import type { SceneDoc, SceneObject } from "../types";

const base: SceneObject = {
  id: "o_base", kind: "splat", asset: "scene__garden", name: "garden", role: "base", visible: true,
  transform: { position: [0, 0, 0], quaternion: [0, 0, 0, 1], scale: 1 },
};
const statue: SceneObject = {
  id: "o_statue", kind: "splat", asset: "a_1", name: "statue", role: "object", visible: true,
  transform: { position: [1, 0, 0], quaternion: [0, 0, 0, 1], scale: 1 },
};
const doc: SceneDoc = { version: 1, id: "s_1", name: "x", viewUp: "y", objects: [base] };

function loaded() {
  return composeReducer(initialComposeState, { type: "load", doc });
}

describe("composeReducer", () => {
  it("loads clean and adding selects + dirties", () => {
    const s0 = loaded();
    expect(s0.dirty).toBe(false);
    const s1 = composeReducer(s0, { type: "add", object: statue });
    expect(s1.doc?.objects.map((o) => o.id)).toEqual(["o_base", "o_statue"]);
    expect(s1.selectedId).toBe("o_statue");
    expect(s1.dirty).toBe(true);
  });

  it("never moves, removes or duplicates the base", () => {
    const s0 = loaded();
    const moved = composeReducer(s0, {
      type: "setTransform", id: "o_base",
      transform: { position: [5, 0, 0], quaternion: [0, 0, 0, 1], scale: 1 },
    });
    expect(moved).toBe(s0);
    expect(composeReducer(s0, { type: "remove", id: "o_base" })).toBe(s0);
    expect(composeReducer(s0, { type: "duplicate", id: "o_base", newId: "o_x" })).toBe(s0);
  });

  it("allows cropping and colouring the base", () => {
    const s = composeReducer(loaded(), {
      type: "setCrop", id: "o_base",
      crop: { center: [0, 0, 0], halfSize: [1, 1, 1], quaternion: [0, 0, 0, 1] },
    });
    expect(s.doc?.objects[0].crop?.halfSize).toEqual([1, 1, 1]);
  });

  it("duplicates, removes and clears selection", () => {
    let s = composeReducer(loaded(), { type: "add", object: statue });
    s = composeReducer(s, { type: "duplicate", id: "o_statue", newId: "o_copy" });
    expect(s.doc?.objects[2]).toMatchObject({ id: "o_copy", name: "statue kopya", asset: "a_1" });
    expect(s.selectedId).toBe("o_copy");
    s = composeReducer(s, { type: "remove", id: "o_copy" });
    expect(s.selectedId).toBeNull();
    expect(s.doc?.objects).toHaveLength(2);
  });

  it("ignores crop/colour on meshes", () => {
    const mesh: SceneObject = { ...statue, id: "o_mesh", kind: "mesh" };
    const s0 = composeReducer(loaded(), { type: "add", object: mesh });
    const s1 = composeReducer(s0, {
      type: "setColor", id: "o_mesh", color: { exposure: 1, tint: [1, 1, 1], saturation: 1 },
    });
    expect(s1).toBe(s0);
  });

  it("markSaved clears dirty", () => {
    const s = composeReducer(composeReducer(loaded(), { type: "add", object: statue }), { type: "markSaved" });
    expect(s.dirty).toBe(false);
  });

  it("generates safe unique ids", () => {
    const a = newObjectId();
    expect(a).toMatch(/^o_[0-9a-f]{12}$/);
    expect(newObjectId()).not.toBe(a);
  });
});
