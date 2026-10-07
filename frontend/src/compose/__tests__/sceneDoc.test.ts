import { describe, expect, it } from "vitest";
import { composeReducer, initialComposeState, newObjectId } from "../sceneDoc";
import { pickMorphSettings, validateMorphSettings } from "../morphSettings";
import type { MorphSettings, SceneDoc, SceneObject } from "../types";

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

const recipe: MorphSettings = {
  sourceId: "o_base", targetId: "o_statue", mode: "shape", duration: 9.5,
  dissolve: 0.35, wave: 0.4, arc: 0.2, targetBlend: 0.85, seed: 4294967295, autoAlign: true,
};

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

  it("setUp normalises, clears on null and dirties", () => {
    const s0 = loaded();
    const s1 = composeReducer(s0, { type: "setUp", up: [0, 0, 2] });
    expect(s1.doc!.up).toEqual([0, 0, 1]);
    expect(s1.dirty).toBe(true);
    // Same up again: unchanged state object.
    expect(composeReducer(s1, { type: "setUp", up: [0, 0, 1] })).toBe(s1);
    const s2 = composeReducer(s1, { type: "setUp", up: null, viewUp: "-y" });
    expect(s2.doc!.up).toBeNull();
    expect(s2.doc!.viewUp).toBe("-y");
    expect(s2.dirty).toBe(true);
    // No up, same viewUp: nothing changes.
    expect(composeReducer(s0, { type: "setUp", up: null, viewUp: "y" })).toBe(s0);
    expect(composeReducer(s0, { type: "setUp", up: null })).toBe(s0);
  });

  it("setUp ignores degenerate vectors", () => {
    const s0 = loaded();
    expect(composeReducer(s0, { type: "setUp", up: [0, 0, 0] })).toBe(s0);
    expect(composeReducer(s0, { type: "setUp", up: [NaN, 1, 0] })).toBe(s0);
    expect(composeReducer(s0, { type: "setUp", up: [Infinity, 0, 0] })).toBe(s0);
    expect(composeReducer(initialComposeState, { type: "setUp", up: [0, 1, 0] })).toBe(initialComposeState);
  });

  it("straighten rotates the current transform about the pivot", () => {
    const s0 = composeReducer(composeReducer(loaded(), { type: "setUp", up: [0, 1, 0] }), { type: "add", object: statue });
    const moved = composeReducer(s0, {
      type: "setTransform", id: "o_statue",
      transform: { position: [5, 0, 0], quaternion: [0, 0, 0, 1], scale: 2 },
    });
    const s1 = composeReducer(moved, { type: "straighten", id: "o_statue", objUp: [0, 0, 1], localPivot: [0, 0, 1] });
    const t = s1.doc!.objects[1].transform;
    // +Z → +Y is −90° about X; the pivot (world (5,0,2)) must stay put.
    expect(t.quaternion[0]).toBeCloseTo(-Math.SQRT1_2, 9);
    expect(t.quaternion[3]).toBeCloseTo(Math.SQRT1_2, 9);
    expect(t.scale).toBe(2);
    expect(t.position[0]).toBeCloseTo(5, 9);
    expect(t.position[1]).toBeCloseTo(-2, 9);
    expect(t.position[2]).toBeCloseTo(2, 9);
    expect(s1.dirty).toBe(true);
  });

  it("straighten ignores the base, meshes, unknown ids and degenerate ups", () => {
    const s0 = composeReducer(loaded(), { type: "add", object: statue });
    const run = (id: string, objUp: [number, number, number]) =>
      composeReducer(s0, { type: "straighten", id, objUp, localPivot: [0, 0, 0] });
    expect(run("o_base", [0, 0, 1])).toBe(s0);
    expect(run("nope", [0, 0, 1])).toBe(s0);
    expect(run("o_statue", [0, 0, 0])).toBe(s0);
    expect(run("o_statue", [NaN, 1, 0])).toBe(s0);
    const s1 = composeReducer(s0, { type: "add", object: { ...statue, id: "o_mesh", kind: "mesh" } });
    expect(composeReducer(s1, { type: "straighten", id: "o_mesh", objUp: [0, 0, 1], localPivot: [0, 0, 0] })).toBe(s1);
  });

  it("generates safe unique ids", () => {
    const a = newObjectId();
    expect(a).toMatch(/^o_[0-9a-f]{12}$/);
    expect(newObjectId()).not.toBe(a);
  });

  it("stores only morph settings, dirties and retains them across save/load", () => {
    const start = composeReducer(loaded(), { type: "add", object: statue });
    const clean = composeReducer(start, { type: "markSaved", doc: start.doc! });
    const runtime = { ...recipe, enabled: true, playing: true, t: 0.9 };
    const edited = composeReducer(clean, { type: "setMorph", morph: runtime });
    expect(edited.dirty).toBe(true);
    expect(edited.doc!.morph).toEqual(recipe);
    expect(edited.doc!.morph).not.toBe(runtime);
    expect(pickMorphSettings(runtime)).toEqual(recipe);
    expect(composeReducer(edited, { type: "setMorph", morph: { ...recipe } })).toBe(edited);
    const saved = composeReducer(edited, { type: "markSaved", doc: edited.doc! });
    const reopened = composeReducer(initialComposeState, { type: "load", doc: JSON.parse(JSON.stringify(saved.doc)) });
    expect(reopened.dirty).toBe(false);
    expect(reopened.doc!.morph).toEqual(recipe);
    expect(composeReducer(reopened, { type: "setMorph", morph: { ...recipe, duration: 10 } }).dirty).toBe(true);
  });

  it("supports absent recipes, partial pairs and clearing", () => {
    const clean = loaded();
    expect(clean.doc!.morph).toBeUndefined();
    expect(composeReducer(clean, { type: "setMorph", morph: null })).toBe(clean);
    const partial = composeReducer(clean, { type: "setMorph", morph: { ...recipe, targetId: null } });
    expect(partial.doc!.morph!.sourceId).toBe("o_base");
    expect(partial.doc!.morph!.targetId).toBeNull();
    const cleared = composeReducer(partial, { type: "setMorph", morph: null });
    expect(cleared.doc!.morph).toBeNull();
    expect(cleared.dirty).toBe(true);
    expect(composeReducer(initialComposeState, { type: "setMorph", morph: recipe })).toBe(initialComposeState);
  });

  it("clears the entire morph when a referenced splat is removed, preserving unrelated recipes", () => {
    const pair = composeReducer(loaded(), { type: "add", object: statue });
    const configured = composeReducer(pair, { type: "setMorph", morph: recipe });
    const extra = composeReducer(configured, { type: "add", object: { ...statue, id: "o_extra" } });
    expect(composeReducer(extra, { type: "remove", id: "o_extra" }).doc!.morph).toEqual(recipe);
    const removed = composeReducer(configured, { type: "remove", id: "o_statue" });
    expect(removed.doc!.morph).toBeNull();
    expect(removed.dirty).toBe(true);
    expect(removed.doc!.objects).toEqual([base]);
    const sourceRemoved = composeReducer(pair, { type: "setMorph", morph: { ...recipe, sourceId: "o_statue", targetId: "o_base" } });
    expect(composeReducer(sourceRemoved, { type: "remove", id: "o_statue" }).doc!.morph).toBeNull();
  });

  it("clears stale morph references when loading a document", () => {
    const stale = composeReducer(initialComposeState, { type: "load", doc: { ...doc, morph: recipe } });
    expect(stale.doc!.morph).toBeNull();
    expect(stale.dirty).toBe(false);
  });

  it.each([
    { duration: NaN }, { duration: Infinity }, { duration: 0.49 }, { duration: 120.1 },
    { dissolve: -0.1 }, { wave: 1.01 }, { arc: Infinity }, { targetBlend: -1 },
    { seed: 1.5 }, { seed: -1 }, { seed: 4294967296 }, { seed: NaN },
    { sourceId: "missing" }, { targetId: "o_base" }, { targetId: "o_mesh" },
    { mode: "unknown" }, { autoAlign: "true" },
  ])("rejects invalid morph settings %j", (patch) => {
    const objects: SceneObject[] = [base, statue, { ...statue, kind: "mesh", id: "o_mesh" }];
    const start = composeReducer(initialComposeState, { type: "load", doc: { ...doc, objects } });
    const bad = { ...recipe, ...patch } as MorphSettings;
    expect(validateMorphSettings(bad, objects)).toBe(false);
    expect(composeReducer(start, { type: "setMorph", morph: bad })).toBe(start);
  });
});
