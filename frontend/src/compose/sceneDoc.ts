import { isValidTransform, straightenTransform } from "./transformMath";
import type { ColorAdjust, CropBox, SceneDoc, SceneObject, Transform, Vec3, ViewUp } from "./types";
import { effectiveUp, normalizeUp } from "./upVector";

export interface ComposeState {
  doc: SceneDoc | null;
  selectedId: string | null;
  dirty: boolean;
}

export const initialComposeState: ComposeState = { doc: null, selectedId: null, dirty: false };

export type ComposeAction =
  | { type: "load"; doc: SceneDoc }
  | { type: "close" }
  | { type: "markSaved"; doc: SceneDoc }
  | { type: "select"; id: string | null }
  | { type: "add"; object: SceneObject }
  | { type: "remove"; id: string }
  | { type: "duplicate"; id: string; newId: string }
  | { type: "setTransform"; id: string; transform: Transform }
  | { type: "setCrop"; id: string; crop: CropBox | null }
  | { type: "setColor"; id: string; color: ColorAdjust | null }
  | { type: "setVisible"; id: string; visible: boolean }
  | { type: "rename"; id: string; name: string }
  /** `up` null = use `viewUp` (optionally changed too); a vector is normalised, degenerate ones ignored. */
  | { type: "setUp"; up: Vec3 | null; viewUp?: ViewUp }
  /**
   * "Dikleştir" a splat object: rotate its CURRENT transform so `objUp` (its
   * own floor up, local frame) matches the scene up, pivoting on `localPivot`.
   */
  | { type: "straighten"; id: string; objUp: Vec3; localPivot: Vec3 };

const MAX_NAME = 128;
const COPY_SUFFIX = " kopya";

function cloneObject(o: SceneObject): SceneObject {
  return {
    ...o,
    transform: { ...o.transform, position: [...o.transform.position], quaternion: [...o.transform.quaternion] },
    ...(o.crop ? { crop: { center: [...o.crop.center], halfSize: [...o.crop.halfSize], quaternion: [...o.crop.quaternion] } } : {}),
    ...(o.color ? { color: { ...o.color, tint: [...o.color.tint] } } : {}),
  } as SceneObject;
}

export function newObjectId(): string {
  const bytes = new Uint8Array(6);
  crypto.getRandomValues(bytes);
  return `o_${Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("")}`;
}

function find(state: ComposeState, id: string): SceneObject | undefined {
  return state.doc?.objects.find((o) => o.id === id);
}

function update(state: ComposeState, id: string, fn: (o: SceneObject) => SceneObject): ComposeState {
  if (!state.doc || !find(state, id)) return state;
  return {
    ...state,
    dirty: true,
    doc: { ...state.doc, objects: state.doc.objects.map((o) => (o.id === id ? fn(o) : o)) },
  };
}

export function composeReducer(state: ComposeState, action: ComposeAction): ComposeState {
  switch (action.type) {
    case "load":
      return { doc: action.doc, selectedId: null, dirty: false };
    case "close":
      return initialComposeState;
    case "markSaved":
      // Only clear dirty if nothing changed since the snapshot that was saved.
      if (!state.dirty || state.doc !== action.doc) return state;
      return { ...state, dirty: false };
    case "select":
      if (state.selectedId === action.id) return state;
      return { ...state, selectedId: action.id };
    case "add":
      if (!state.doc || action.object.role === "base" || find(state, action.object.id)) return state;
      return {
        doc: { ...state.doc, objects: [...state.doc.objects, action.object] },
        selectedId: action.object.id,
        dirty: true,
      };
    case "remove": {
      const target = find(state, action.id);
      if (!state.doc || !target || target.role === "base") return state;
      return {
        doc: { ...state.doc, objects: state.doc.objects.filter((o) => o.id !== action.id) },
        selectedId: state.selectedId === action.id ? null : state.selectedId,
        dirty: true,
      };
    }
    case "duplicate": {
      const target = find(state, action.id);
      if (!state.doc || !target || target.role === "base") return state;
      const name = target.name.slice(0, MAX_NAME - COPY_SUFFIX.length) + COPY_SUFFIX;
      const copy: SceneObject = { ...cloneObject(target), id: action.newId, name };
      return { doc: { ...state.doc, objects: [...state.doc.objects, copy] }, selectedId: copy.id, dirty: true };
    }
    case "setTransform": {
      const target = find(state, action.id);
      if (!target || target.role === "base" || !isValidTransform(action.transform)) return state;
      return update(state, action.id, (o) => ({ ...o, transform: action.transform }));
    }
    case "setCrop": {
      const target = find(state, action.id);
      if (!target || target.kind !== "splat") return state;
      return update(state, action.id, (o) => ({ ...o, crop: action.crop }));
    }
    case "setColor": {
      const target = find(state, action.id);
      if (!target || target.kind !== "splat") return state;
      return update(state, action.id, (o) => ({ ...o, color: action.color }));
    }
    case "setVisible":
      return update(state, action.id, (o) => ({ ...o, visible: action.visible }));
    case "rename": {
      const name = action.name.trim().slice(0, MAX_NAME);
      if (!name) return state;
      return update(state, action.id, (o) => ({ ...o, name }));
    }
    case "setUp": {
      if (!state.doc) return state;
      const up = action.up === null ? null : normalizeUp(action.up);
      if (action.up !== null && !up) return state;
      const viewUp = action.viewUp ?? state.doc.viewUp;
      const prev = state.doc.up ?? null;
      const sameUp = prev === null || up === null ? prev === up : prev.every((c, i) => c === up[i]);
      if (sameUp && viewUp === state.doc.viewUp) return state;
      return { ...state, doc: { ...state.doc, up, viewUp }, dirty: true };
    }
    case "straighten": {
      const target = find(state, action.id);
      const objUp = normalizeUp(action.objUp);
      if (!state.doc || !target || target.role === "base" || target.kind !== "splat" || !objUp) return state;
      if (!action.localPivot.every(Number.isFinite)) return state;
      const transform = straightenTransform(target.transform, objUp, effectiveUp(state.doc), action.localPivot);
      if (!isValidTransform(transform)) return state;
      return update(state, action.id, (o) => ({ ...o, transform }));
    }
  }
}
