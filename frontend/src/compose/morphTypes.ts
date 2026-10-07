import type { MorphSettings } from "./types";

export type MorphMode = MorphSettings["mode"];

/** Persisted morph settings plus transient preview and playback controls. */
export interface MorphState extends MorphSettings {
  enabled: boolean;
  playing: boolean;
  /** Last requested timeline position. Live playback lives in MorphPlayback. */
  t: number;
}

/** Renderer-owned live time, sampled by the panel without rerendering the scene. */
export interface MorphPlayback {
  t: number;
  playing: boolean;
}

export interface MorphStatus {
  phase: "idle" | "loading" | "ready" | "error";
  count?: number;
  precomputeMs?: number;
  /** Mean world endpoint distance divided by the larger robust object diagonal. */
  meanTravel?: number;
  alignmentMs?: number;
  message?: string;
}

export const INITIAL_MORPH_STATE: MorphState = {
  sourceId: null,
  targetId: null,
  enabled: false,
  playing: false,
  t: 0,
  duration: 6,
  mode: "shape",
  dissolve: 1,
  wave: 0,
  arc: 0,
  targetBlend: 1,
  seed: 42,
  autoAlign: false,
};

/** Picking the opposite slot moves an object, so a pair can never repeat an ID. */
export function selectMorphSlot(state: MorphState, slot: "sourceId" | "targetId", id: string): MorphState {
  const other = slot === "sourceId" ? "targetId" : "sourceId";
  return {
    ...state,
    [slot]: state[slot] === id ? null : id,
    [other]: state[other] === id ? null : state[other],
    enabled: false,
    playing: false,
    t: 0,
  };
}
