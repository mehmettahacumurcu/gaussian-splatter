import type { ColorAdjust, CropBox, SceneDoc, SceneObject, Transform, ViewUp } from "./types";

export interface ComposeState {
  doc: SceneDoc | null;
  selectedId: string | null;
  dirty: boolean;
}

export const initialComposeState: ComposeState = { doc: null, selectedId: null, dirty: false };

export type ComposeAction =
  | { type: "load"; doc: SceneDoc }
  | { type: "close" }
  | { type: "markSaved" }
  | { type: "select"; id: string | null }
  | { type: "add"; object: SceneObject }
  | { type: "remove"; id: string }
  | { type: "duplicate"; id: string; newId: string }
  | { type: "setTransform"; id: string; transform: Transform }
  | { type: "setCrop"; id: string; crop: CropBox | null }
  | { type: "setColor"; id: string; color: ColorAdjust | null }
  | { type: "setVisible"; id: string; visible: boolean }
  | { type: "rename"; id: string; name: string }
  | { type: "setViewUp"; viewUp: ViewUp };

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
      return { ...state, dirty: false };
    case "select":
      return { ...state, selectedId: action.id };
    case "add":
      if (!state.doc) return state;
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
      const copy: SceneObject = { ...structuredClone(target), id: action.newId, name: `${target.name} kopya` };
      return { doc: { ...state.doc, objects: [...state.doc.objects, copy] }, selectedId: copy.id, dirty: true };
    }
    case "setTransform": {
      const target = find(state, action.id);
      if (!target || target.role === "base") return state;
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
    case "rename":
      return update(state, action.id, (o) => ({ ...o, name: action.name }));
    case "setViewUp":
      if (!state.doc || state.doc.viewUp === action.viewUp) return state;
      return { ...state, doc: { ...state.doc, viewUp: action.viewUp }, dirty: true };
  }
}
