import { describe, expect, it } from "vitest";

import {
  buildStaticNotebookRunSpec,
  initialNotebookGeneratorState,
  notebookReducer,
  validateNotebookDraft,
} from "../reducer";
import type {
  NotebookAdvancedDraft,
  NotebookDraft,
  NotebookGeneratorState,
  StaticNotebookPresetsResponse,
} from "../types";

const PRESETS: StaticNotebookPresetsResponse = {
  schema_version: 1,
  default_profile: "balanced_l4",
  profiles: [
    {
      id: "balanced_l4",
      label: "Balanced / L4",
      description: "L4 profile",
      n_iters: 30000,
      max_gaussians: 250000,
      selected_frame_budget: 300,
      resolution_long_edge_cap: 1280,
      intended_gpu: "NVIDIA L4 24 GB",
      minimum_vram_gb: 22,
      foundation_default: true,
      run_eval_default: false,
      warnings: [],
    },
    {
      id: "high",
      label: "High",
      description: "High profile",
      n_iters: 50000,
      max_gaussians: 500000,
      selected_frame_budget: 450,
      resolution_long_edge_cap: 1920,
      intended_gpu: "32 GB class",
      minimum_vram_gb: 30,
      foundation_default: true,
      run_eval_default: false,
      warnings: ["Use 30 GB VRAM"],
    },
  ],
  override_limits: {
    fixed_fps: { min: 1, max: 30 },
    n_iters: { min: 1000, max: 120000 },
    max_gaussians: { min: 50000, max: 6000000 },
  },
};

function readyState(): NotebookGeneratorState {
  const initial = initialNotebookGeneratorState();
  return {
    ...initial,
    draft: { ...initial.draft, inputFolderRaw: "captures/room" },
    presets: { status: "ready" as const, value: PRESETS },
  };
}

function advancedDraft(
  values: Partial<NotebookAdvancedDraft>,
): NotebookDraft {
  const state = readyState();
  return {
    ...state.draft,
    inputFolderRaw: "captures/room",
    quality: {
      ...state.draft.quality,
      advanced: { ...state.draft.quality.advanced, ...values },
    },
  };
}

describe("notebook reducer", () => {
  it("defaults to Smart, FPS 4, and backend Balanced/L4", () => {
    const state = initialNotebookGeneratorState();
    expect(state.draft.frameSelection).toEqual({
      mode: "smart",
      fixedFps: "4",
    });
    expect(state.draft.quality.profile).toBe("balanced_l4");
    expect(state.draft.quality.nIters).toBe("");
  });

  it("keeps explicit overrides across profile changes and reset clears them", () => {
    let state = readyState();
    state = notebookReducer(state, {
      type: "iterations_changed",
      value: "42000",
    });
    state = notebookReducer(state, { type: "profile_changed", value: "high" });
    expect(state.draft.quality.nIters).toBe("42000");
    state = notebookReducer(state, { type: "primary_overrides_reset" });
    expect(state.draft.quality.nIters).toBe("");
    expect(buildStaticNotebookRunSpec(state.draft, PRESETS).quality.n_iters).toBeNull();
  });

  it("compacts empty advanced fields and parses multires schedule", () => {
    const draft = advancedDraft({
      lambda_ssim: "0.25",
      multires_schedule: "0:720,25000:1080",
    });
    expect(buildStaticNotebookRunSpec(draft, PRESETS).quality.advanced).toEqual({
      lambda_ssim: 0.25,
      multires_schedule: [
        [0, 720],
        [25000, 1080],
      ],
    });
  });

  it("any edit invalidates a generated artifact", () => {
    const state = {
      ...readyState(),
      generation: {
        status: "ready" as const,
        artifact: { blob: new Blob(["x"]), filename: "x.ipynb" },
      },
    };
    const edited = notebookReducer(state, {
      type: "input_changed",
      value: "captures/new",
    });
    expect(edited.generation).toEqual({ status: "idle" });
  });

  it("serializes a safe backend FPS while Smart hides a cleared baseline field", () => {
    const state = readyState();
    state.draft.frameSelection.fixedFps = "";
    expect(buildStaticNotebookRunSpec(state.draft, PRESETS).frame_selection).toEqual({
      mode: "smart",
      fixed_fps: 4,
    });
  });
});

describe("draft validation", () => {
  it("uses backend bounds and profile warnings", () => {
    const draft = advancedDraft({});
    draft.quality.profile = "high";
    draft.frameSelection.mode = "fixed_fps";
    draft.frameSelection.fixedFps = "31";
    const result = validateNotebookDraft(draft, PRESETS);
    expect(result.valid).toBe(false);
    expect(result.errors.fixedFps).toContain("30");
    expect(result.warnings).toEqual(["Use 30 GB VRAM"]);
  });

  it("rejects malformed multires and invalid density ordering", () => {
    const draft = advancedDraft({
      multires_schedule: "500:720,0:1080",
      density_start_iter: "20000",
      density_end_iter: "10000",
    });
    const result = validateNotebookDraft(draft, PRESETS);
    expect(result.errors.multires_schedule).toBeTruthy();
    expect(result.errors.density_end_iter).toBeTruthy();
  });
});
