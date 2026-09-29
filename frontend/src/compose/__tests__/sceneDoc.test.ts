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
    const dirty = composeReducer(loaded(), { type: "add", object: statue });
    const s = composeReducer(dirty, { type: "markSaved", doc: dirty.doc! });
    expect(s.dirty).toBe(false);
  });

  it("markSaved keeps dirty when the doc changed after the saved snapshot", () => {
    const snapshot = composeReducer(loaded(), { type: "add", object: statue });
    const edited = composeReducer(snapshot, { type: "rename", id: "o_statue", name: "renamed" });
    const s = composeReducer(edited, { type: "markSaved", doc: snapshot.doc! });
    expect(s).toBe(edited);
    expect(s.dirty).toBe(true);
  });

  it("markSaved on a clean state returns the same state", () => {
    const s0 = loaded();
    expect(composeReducer(s0, { type: "markSaved", doc: s0.doc! })).toBe(s0);
  });

  it("select returns the same state when unchanged", () => {
    const s0 = composeReducer(loaded(), { type: "select", id: "o_base" });
    expect(composeReducer(s0, { type: "select", id: "o_base" })).toBe(s0);
  });

  it("truncates duplicate names to 128 characters", () => {
    const long: SceneObject = { ...statue, name: "x".repeat(128) };
    let s = composeReducer(loaded(), { type: "add", object: long });
    s = composeReducer(s, { type: "duplicate", id: "o_statue", newId: "o_copy" });
    const name = s.doc!.objects[2].name;
    expect(name.length).toBeLessThanOrEqual(128);
    expect(name.endsWith(" kopya")).toBe(true);
  });

  it("duplicate does not share nested objects with the original", () => {
    const withExtras: SceneObject = {
      ...statue,
      crop: { center: [0, 0, 0], halfSize: [1, 1, 1], quaternion: [0, 0, 0, 1] },
      color: { exposure: 0, tint: [1, 1, 1], saturation: 1 },
    };
    let s = composeReducer(loaded(), { type: "add", object: withExtras });
    s = composeReducer(s, { type: "duplicate", id: "o_statue", newId: "o_copy" });
    const [orig, copy] = [s.doc!.objects[1], s.doc!.objects[2]];
    expect(copy.transform).toEqual(orig.transform);
    expect(copy.transform).not.toBe(orig.transform);
    expect(copy.transform.position).not.toBe(orig.transform.position);
    expect(copy.crop).not.toBe(orig.crop);
    expect(copy.color).not.toBe(orig.color);
  });

  it("ignores adding a duplicate id or a second base", () => {
    const s0 = composeReducer(loaded(), { type: "add", object: statue });
    expect(composeReducer(s0, { type: "add", object: statue })).toBe(s0);
    expect(composeReducer(s0, { type: "add", object: { ...statue, id: "o_b2", role: "base" } })).toBe(s0);
  });

  it("rename trims, caps at 128 and ignores blank names", () => {
    const s0 = composeReducer(loaded(), { type: "add", object: statue });
    expect(composeReducer(s0, { type: "rename", id: "o_statue", name: "   " })).toBe(s0);
    expect(composeReducer(s0, { type: "rename", id: "o_statue", name: "" })).toBe(s0);
    const s1 = composeReducer(s0, { type: "rename", id: "o_statue", name: "  new  " });
    expect(s1.doc!.objects[1].name).toBe("new");
    const s2 = composeReducer(s0, { type: "rename", id: "o_statue", name: "y".repeat(200) });
    expect(s2.doc!.objects[1].name).toHaveLength(128);
  });

  it("ignores invalid transforms", () => {
    const s0 = composeReducer(loaded(), { type: "add", object: statue });
    const bad = (t: Partial<SceneObject["transform"]>) =>
      composeReducer(s0, { type: "setTransform", id: "o_statue", transform: { ...statue.transform, ...t } });
    expect(bad({ position: [NaN, 0, 0] })).toBe(s0);
    expect(bad({ scale: 0 })).toBe(s0);
    expect(bad({ scale: Infinity })).toBe(s0);
    expect(bad({ quaternion: [0, 0, 0, 0] })).toBe(s0);
    expect(bad({ position: [2, 0, 0] })).not.toBe(s0);
  });

  it("generates safe unique ids", () => {
    const a = newObjectId();
    expect(a).toMatch(/^o_[0-9a-f]{12}$/);
    expect(newObjectId()).not.toBe(a);
  });
});
