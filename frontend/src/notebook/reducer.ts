import { resolveDriveFolder, type DriveFolderResolution } from "./drivePath";
import type {
  FrameSelectionMode,
  GeneratedNotebook,
  MultiresStage,
  NotebookAdvancedConfig,
  NotebookAdvancedDraft,
  NotebookDraft,
  NotebookGeneratorState,
  NotebookQualityProfileId,
  StaticNotebookPresetsResponse,
  StaticNotebookRunSpec,
} from "./types";

export interface NotebookDraftValidation {
  valid: boolean;
  path: DriveFolderResolution;
  errors: Record<string, string>;
  warnings: string[];
}

export type NotebookAction =
  | { type: "presets_loading" }
  | { type: "presets_succeeded"; value: StaticNotebookPresetsResponse }
  | { type: "presets_failed"; message: string }
  | { type: "input_changed"; value: string }
  | { type: "selection_mode_changed"; value: FrameSelectionMode }
  | { type: "fixed_fps_changed"; value: string }
  | { type: "profile_changed"; value: NotebookQualityProfileId }
  | { type: "iterations_changed"; value: string }
  | { type: "max_gaussians_changed"; value: string }
  | {
      type: "advanced_changed";
      field: keyof NotebookAdvancedDraft;
      value: string | boolean | null;
    }
  | { type: "primary_overrides_reset" }
  | { type: "advanced_reset" }
  | { type: "generation_started" }
  | { type: "generation_succeeded"; artifact: GeneratedNotebook }
  | { type: "generation_failed"; message: string }
  | { type: "generation_reset" };

function emptyAdvancedDraft(): NotebookAdvancedDraft {
  return {
    run_eval: null,
    foundation: null,
    resolution_long_edge_cap: "",
    lambda_ssim: "",
    lambda_lpips: "",
    lambda_depth: "",
    density_start_iter: "",
    density_end_iter: "",
    density_interval: "",
    densify_grad_threshold: "",
    prune_min_opacity: "",
    prune_max_scale: "",
    opacity_reset_interval: "",
    sh_degree: "",
    multires_schedule: "",
  };
}

export function initialNotebookGeneratorState(): NotebookGeneratorState {
  return {
    draft: {
      inputFolderRaw: "",
      frameSelection: { mode: "smart", fixedFps: "4" },
      quality: {
        profile: "balanced_l4",
        nIters: "",
        maxGaussians: "",
        advanced: emptyAdvancedDraft(),
      },
    },
    presets: { status: "loading" },
    generation: { status: "idle" },
  };
}

function edited(
  state: NotebookGeneratorState,
  draft: NotebookDraft,
): NotebookGeneratorState {
  return { ...state, draft, generation: { status: "idle" } };
}

export function notebookReducer(
  state: NotebookGeneratorState,
  action: NotebookAction,
): NotebookGeneratorState {
  switch (action.type) {
    case "presets_loading":
      return { ...state, presets: { status: "loading" } };
    case "presets_succeeded":
      return { ...state, presets: { status: "ready", value: action.value } };
    case "presets_failed":
      return { ...state, presets: { status: "error", message: action.message } };
    case "input_changed":
      return edited(state, { ...state.draft, inputFolderRaw: action.value });
    case "selection_mode_changed":
      return edited(state, {
        ...state.draft,
        frameSelection: { ...state.draft.frameSelection, mode: action.value },
      });
    case "fixed_fps_changed":
      return edited(state, {
        ...state.draft,
        frameSelection: {
          ...state.draft.frameSelection,
          fixedFps: action.value,
        },
      });
    case "profile_changed":
      return edited(state, {
        ...state.draft,
        quality: { ...state.draft.quality, profile: action.value },
      });
    case "iterations_changed":
      return edited(state, {
        ...state.draft,
        quality: { ...state.draft.quality, nIters: action.value },
      });
    case "max_gaussians_changed":
      return edited(state, {
        ...state.draft,
        quality: { ...state.draft.quality, maxGaussians: action.value },
      });
    case "advanced_changed": {
      const advanced = {
        ...state.draft.quality.advanced,
        [action.field]: action.value,
      } as NotebookAdvancedDraft;
      return edited(state, {
        ...state.draft,
        quality: { ...state.draft.quality, advanced },
      });
    }
    case "primary_overrides_reset":
      return edited(state, {
        ...state.draft,
        quality: { ...state.draft.quality, nIters: "", maxGaussians: "" },
      });
    case "advanced_reset":
      return edited(state, {
        ...state.draft,
        quality: { ...state.draft.quality, advanced: emptyAdvancedDraft() },
      });
    case "generation_started":
      return { ...state, generation: { status: "generating" } };
    case "generation_succeeded":
      return {
        ...state,
        generation: { status: "ready", artifact: action.artifact },
      };
    case "generation_failed":
      return {
        ...state,
        generation: { status: "error", message: action.message },
      };
    case "generation_reset":
      return { ...state, generation: { status: "idle" } };
  }
}

function parseNumber(
  raw: string,
  field: string,
  errors: Record<string, string>,
  options: { min: number; max: number; integer?: boolean; exclusiveMin?: boolean },
): number | undefined {
  if (raw.trim() === "") return undefined;
  const value = Number(raw);
  const below = options.exclusiveMin ? value <= options.min : value < options.min;
  if (
    !Number.isFinite(value) ||
    (options.integer && !Number.isInteger(value)) ||
    below ||
    value > options.max
  ) {
    const lower = options.exclusiveMin ? `greater than ${options.min}` : `${options.min}`;
    errors[field] = `Enter ${options.integer ? "a whole number" : "a number"} from ${lower} to ${options.max}.`;
    return undefined;
  }
  return value;
}

function parseMultires(
  raw: string,
  errors: Record<string, string>,
  effectiveIterations: number,
): MultiresStage[] | undefined {
  if (raw.trim() === "") return undefined;
  const stages: MultiresStage[] = [];
  for (const part of raw.split(",")) {
    const match = /^\s*(\d+)\s*:\s*(\d+)\s*$/.exec(part);
    if (!match) {
      errors.multires_schedule = "Use start:edge pairs, for example 0:720,25000:1080.";
      return undefined;
    }
    stages.push([Number(match[1]), Number(match[2])]);
  }
  const starts = stages.map(([start]) => start);
  if (
    stages[0]?.[0] !== 0 ||
    starts.some((start, index) => index > 0 && start <= starts[index - 1]) ||
    stages.some(([, edge]) => edge < 320 || edge > 3840) ||
    stages.at(-1)![0] >= effectiveIterations
  ) {
    errors.multires_schedule =
      "Start at 0, increase stage starts, keep edges 320–3840, and start before the final iteration.";
    return undefined;
  }
  return stages;
}

function parseAdvanced(
  draft: NotebookAdvancedDraft,
  errors: Record<string, string>,
  effectiveIterations: number,
): NotebookAdvancedConfig {
  const result: NotebookAdvancedConfig = {};
  const put = <K extends keyof NotebookAdvancedConfig>(
    key: K,
    value: NotebookAdvancedConfig[K] | undefined,
  ) => {
    if (value !== undefined) result[key] = value;
  };
  if (draft.run_eval !== null) result.run_eval = draft.run_eval;
  if (draft.foundation !== null) result.foundation = draft.foundation;
  put(
    "resolution_long_edge_cap",
    parseNumber(draft.resolution_long_edge_cap, "resolution_long_edge_cap", errors, {
      min: 320,
      max: 3840,
      integer: true,
    }),
  );
  for (const field of ["lambda_ssim", "lambda_lpips", "lambda_depth"] as const) {
    put(field, parseNumber(draft[field], field, errors, { min: 0, max: 1 }));
  }
  for (const field of [
    "density_start_iter",
    "density_end_iter",
    "opacity_reset_interval",
  ] as const) {
    put(
      field,
      parseNumber(draft[field], field, errors, {
        min: 0,
        max: 120000,
        integer: true,
      }),
    );
  }
  put(
    "density_interval",
    parseNumber(draft.density_interval, "density_interval", errors, {
      min: 10,
      max: 5000,
      integer: true,
    }),
  );
  put(
    "densify_grad_threshold",
    parseNumber(draft.densify_grad_threshold, "densify_grad_threshold", errors, {
      min: 0,
      max: 0.1,
      exclusiveMin: true,
    }),
  );
  put(
    "prune_min_opacity",
    parseNumber(draft.prune_min_opacity, "prune_min_opacity", errors, {
      min: 0,
      max: 1,
    }),
  );
  put(
    "prune_max_scale",
    parseNumber(draft.prune_max_scale, "prune_max_scale", errors, {
      min: 0,
      max: 1,
      exclusiveMin: true,
    }),
  );
  put(
    "sh_degree",
    parseNumber(draft.sh_degree, "sh_degree", errors, {
      min: 0,
      max: 3,
      integer: true,
    }),
  );
  put("multires_schedule", parseMultires(draft.multires_schedule, errors, effectiveIterations));

  if (
    result.density_start_iter !== undefined &&
    result.density_end_iter !== undefined &&
    result.density_start_iter >= result.density_end_iter
  ) {
    errors.density_end_iter = "Density end must be greater than density start.";
  }
  if (
    result.density_start_iter !== undefined &&
    result.density_start_iter >= effectiveIterations
  ) {
    errors.density_start_iter = "Density start must be below the effective iterations.";
  }
  if (
    result.density_end_iter !== undefined &&
    result.density_end_iter > effectiveIterations
  ) {
    errors.density_end_iter = "Density end cannot exceed the effective iterations.";
  }
  return result;
}

export function validateNotebookDraft(
  draft: NotebookDraft,
  presets: StaticNotebookPresetsResponse,
): NotebookDraftValidation {
  const errors: Record<string, string> = {};
  const path = resolveDriveFolder(draft.inputFolderRaw);
  if (!path.ok) errors.inputFolder = path.error;
  const profile = presets.profiles.find((item) => item.id === draft.quality.profile);
  if (!profile) errors.profile = "Choose a profile provided by the backend.";

  if (draft.frameSelection.mode === "fixed_fps") {
    parseNumber(draft.frameSelection.fixedFps, "fixedFps", errors, {
      ...presets.override_limits.fixed_fps,
      integer: true,
    });
  }
  const nIters = parseNumber(draft.quality.nIters, "nIters", errors, {
    ...presets.override_limits.n_iters,
    integer: true,
  });
  parseNumber(draft.quality.maxGaussians, "maxGaussians", errors, {
    ...presets.override_limits.max_gaussians,
    integer: true,
  });
  const effectiveIterations = nIters ?? profile?.n_iters ?? 0;
  parseAdvanced(draft.quality.advanced, errors, effectiveIterations);
  return {
    valid: path.ok && Object.keys(errors).length === 0,
    path,
    errors,
    warnings: profile?.warnings ?? [],
  };
}

export function buildStaticNotebookRunSpec(
  draft: NotebookDraft,
  presets: StaticNotebookPresetsResponse,
): StaticNotebookRunSpec {
  const validation = validateNotebookDraft(draft, presets);
  if (!validation.valid || !validation.path.ok) {
    throw new Error("Notebook draft is invalid.");
  }
  const profile = presets.profiles.find((item) => item.id === draft.quality.profile)!;
  const nIters = draft.quality.nIters.trim() === "" ? null : Number(draft.quality.nIters);
  const maxGaussians =
    draft.quality.maxGaussians.trim() === ""
      ? null
      : Number(draft.quality.maxGaussians);
  const errors: Record<string, string> = {};
  const advanced = parseAdvanced(
    draft.quality.advanced,
    errors,
    nIters ?? profile.n_iters,
  );
  return {
    schema_version: 1,
    input_folder: validation.path.canonicalInput,
    frame_selection: {
      mode: draft.frameSelection.mode,
      fixed_fps:
        draft.frameSelection.mode === "smart"
          ? 4
          : Number(draft.frameSelection.fixedFps),
    },
    quality: {
      profile: draft.quality.profile,
      n_iters: nIters,
      max_gaussians: maxGaussians,
      advanced,
    },
    publish: { replace_owned_result: true },
  };
}
