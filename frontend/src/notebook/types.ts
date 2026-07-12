export type FrameSelectionMode = "smart" | "fixed_fps";
export type NotebookQualityProfileId =
  | "balanced_l4"
  | "high"
  | "premium"
  | "ultra";
export type MultiresStage = [startIter: number, longEdge: number];

export interface NotebookAdvancedValues {
  run_eval: boolean;
  foundation: boolean;
  resolution_long_edge_cap: number;
  lambda_ssim: number;
  lambda_lpips: number;
  lambda_depth: number;
  density_start_iter: number;
  density_end_iter: number;
  density_interval: number;
  densify_grad_threshold: number;
  prune_min_opacity: number;
  prune_max_scale: number;
  opacity_reset_interval: number;
  sh_degree: number;
  multires_schedule: MultiresStage[];
}

export type NotebookAdvancedConfig = Partial<NotebookAdvancedValues>;

export interface StaticNotebookRunSpec {
  schema_version: 1;
  input_folder: string;
  frame_selection: { mode: FrameSelectionMode; fixed_fps: number };
  quality: {
    profile: NotebookQualityProfileId;
    n_iters: number | null;
    max_gaussians: number | null;
    advanced: NotebookAdvancedConfig;
  };
  publish: { replace_owned_result: true };
}

export interface NotebookProfileMetadata {
  id: NotebookQualityProfileId;
  label: string;
  description: string;
  n_iters: number;
  max_gaussians: number;
  selected_frame_budget: number;
  resolution_long_edge_cap: number | null;
  intended_gpu: string;
  minimum_vram_gb: number;
  foundation_default: boolean;
  run_eval_default: boolean;
  warnings: string[];
}

export interface StaticNotebookPresetsResponse {
  schema_version: 1;
  default_profile: NotebookQualityProfileId;
  profiles: NotebookProfileMetadata[];
  override_limits: {
    fixed_fps: { min: number; max: number };
    n_iters: { min: number; max: number };
    max_gaussians: { min: number; max: number };
  };
}

export interface GeneratedNotebook {
  blob: Blob;
  filename: string;
}

export interface NotebookAdvancedDraft {
  run_eval: boolean | null;
  foundation: boolean | null;
  resolution_long_edge_cap: string;
  lambda_ssim: string;
  lambda_lpips: string;
  lambda_depth: string;
  density_start_iter: string;
  density_end_iter: string;
  density_interval: string;
  densify_grad_threshold: string;
  prune_min_opacity: string;
  prune_max_scale: string;
  opacity_reset_interval: string;
  sh_degree: string;
  multires_schedule: string;
}

export interface NotebookDraft {
  inputFolderRaw: string;
  frameSelection: { mode: FrameSelectionMode; fixedFps: string };
  quality: {
    profile: NotebookQualityProfileId;
    nIters: string;
    maxGaussians: string;
    advanced: NotebookAdvancedDraft;
  };
}

export type PresetsState =
  | { status: "loading" }
  | { status: "ready"; value: StaticNotebookPresetsResponse }
  | { status: "error"; message: string };

export type GenerationState =
  | { status: "idle" }
  | { status: "generating" }
  | { status: "ready"; artifact: GeneratedNotebook }
  | { status: "error"; message: string };

export interface NotebookGeneratorState {
  draft: NotebookDraft;
  presets: PresetsState;
  generation: GenerationState;
}
