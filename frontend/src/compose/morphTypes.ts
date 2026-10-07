/** Transient preview controls; these are deliberately not part of SceneDoc. */
export interface MorphState {
  sourceId: string | null;
  targetId: string | null;
  enabled: boolean;
  playing: boolean;
  /** Last requested timeline position. Live playback lives in MorphPlayback. */
  t: number;
  duration: number;
  dissolve: number;
  targetBlend: number;
  seed: number;
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
  message?: string;
}

export const INITIAL_MORPH_STATE: MorphState = {
  sourceId: null,
  targetId: null,
  enabled: false,
  playing: false,
  t: 0,
  duration: 6,
  dissolve: 1,
  targetBlend: 1,
  seed: 42,
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
