import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { generateStaticNotebook, getStaticNotebookPresets, generatePipelineNotebook, getPipelineNotebookPresets } from "../../api";
import { NotebookGeneratorPanel } from "../NotebookGeneratorPanel";
import type { StaticNotebookPresetsResponse } from "../types";

vi.mock("../../api", () => ({
  getStaticNotebookPresets: vi.fn(),
  generateStaticNotebook: vi.fn(),
  generatePipelineNotebook: vi.fn(),
  getPipelineNotebookPresets: vi.fn(),
}));

const PRESETS: StaticNotebookPresetsResponse = {
  schema_version: 1,
  default_profile: "balanced_l4",
  profiles: [
    ["balanced_l4", "Balanced / L4", 30000, 250000, 300, 1280, 22],
    ["high", "High", 50000, 500000, 450, 1920, 30],
    ["premium", "Premium", 100000, 1000000, 600, 2560, 46],
    ["ultra", "Ultra", 120000, 3000000, 800, null, 75],
  ].map(([id, label, nIters, maxGaussians, frames, resolution, vram]) => ({
    id: id as "balanced_l4" | "high" | "premium" | "ultra",
    label: String(label),
    description: `${label} profile`,
    n_iters: Number(nIters),
    max_gaussians: Number(maxGaussians),
    selected_frame_budget: Number(frames),
    resolution_long_edge_cap: resolution === null ? null : Number(resolution),
    intended_gpu: `${vram} GB class`,
    minimum_vram_gb: Number(vram),
    foundation_default: true,
    run_eval_default: false,
    warnings: id === "ultra" ? ["A100 80 GB class runtime required"] : [],
  })),
  override_limits: {
    fixed_fps: { min: 1, max: 30 },
    n_iters: { min: 1000, max: 120000 },
    max_gaussians: { min: 50000, max: 6000000 },
  },
};

beforeEach(() => {
  const recipe = { iterations: 30000, max_gaussians: 1000000, min_vram: 14, recipe: "medium" };
  const textRecipe = { min_vram: 20, recipe: "sdxl_trellis", image_steps: 25, sparse_steps: 12, slat_steps: 12, resolution: 1024 };
  vi.mocked(getPipelineNotebookPresets).mockResolvedValue({ template_version: 1, presets: {
    hybrid: { baseline: recipe, quality: recipe, ultra: recipe },
    spirula: { baseline: recipe, quality: { ...recipe, iterations: 60000, max_gaussians: 6000000 }, ultra: recipe },
    text_to_splat: { baseline: textRecipe, quality: textRecipe, ultra: { ...textRecipe, min_vram: 38 } },
  }});
  vi.mocked(generatePipelineNotebook).mockResolvedValue({ blob: new Blob(["notebook"]), filename: "room_spirula.ipynb" });
  vi.mocked(getStaticNotebookPresets).mockResolvedValue(PRESETS);
  vi.mocked(generateStaticNotebook).mockResolvedValue({
    blob: new Blob(["notebook"]),
    filename: "myroom_static_splat.ipynb",
  });
  Object.defineProperty(URL, "createObjectURL", {
    configurable: true,
    value: vi.fn(() => "blob:notebook"),
  });
  Object.defineProperty(URL, "revokeObjectURL", {
    configurable: true,
    value: vi.fn(),
  });
  vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
});

describe("NotebookGeneratorPanel", () => {
  it("opens text-to-splat from the pipeline selector and submits text defaults", async () => {
    render(<NotebookGeneratorPanel />);
    fireEvent.change(screen.getByLabelText("Pipeline"), { target: { value: "text_to_splat" } });
    await screen.findByText(/En az 20 GiB/);
    fireEvent.change(screen.getByLabelText("Nesne tarifi (PROMPT)"), { target: { value: "a red sports car" } });
    expect(screen.queryByLabelText("MyDrive path")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Metinden splat notebook’unu indir" }));
    await waitFor(() => expect(generatePipelineNotebook).toHaveBeenCalledWith({
      pipeline: "text_to_splat", input_mode: "text", input_path: "", prompt: "a red sports car",
      negative_prompt: "", style: "", seed: 42, preset: "baseline", output_dir: null, target_splat_count: null,
      image_model: "sdxl", background_model: "u2net", reconstruction_model: "trellis", gpu_preset: "l4",
      image_steps: 25, image_guidance: 7, image_resolution: 1024, trellis_seed: 42,
      sparse_steps: 12, sparse_cfg: 7.5, slat_steps: 12, slat_cfg: 3,
      mesh_views: 24, mesh_fit_iterations: 1500, mesh_splat_cap: 50000,
      trellis2_pipeline_type: "512", mesh_render_resolution: 1024, mesh_sh_degree: 2, mesh_texture_size: 2048,
    }));
    expect(generateStaticNotebook).not.toHaveBeenCalled();
  });

  it("generates Spirula with selected input and training overrides", async () => {
    render(<NotebookGeneratorPanel />);
    fireEvent.change(screen.getByLabelText("Pipeline"), { target: { value: "spirula" } });
    await screen.findByPlaceholderText("30000");
    fireEvent.change(screen.getByLabelText("MyDrive path"), { target: { value: "captures/room.MOV" } });
    fireEvent.change(screen.getByLabelText("Training preset"), { target: { value: "quality" } });
    fireEvent.change(screen.getByLabelText("Training steps"), { target: { value: "65000" } });
    fireEvent.change(screen.getByLabelText("Extraction target FPS"), { target: { value: "8" } });
    fireEvent.click(screen.getByRole("button", { name: "Generate and download notebook" }));
    await waitFor(() => expect(generatePipelineNotebook).toHaveBeenCalledWith(expect.objectContaining({
      pipeline: "spirula", input_mode: "video", input_path: "captures/room.MOV", preset: "quality", iterations: 65000, fps: 8,
    })));
    expect(generateStaticNotebook).not.toHaveBeenCalled();
  });

  it("hybrid accepts prepared datasets and hides video preprocessing controls", async () => {
    render(<NotebookGeneratorPanel />);
    fireEvent.change(screen.getByLabelText("Pipeline"), { target: { value: "hybrid" } });
    await screen.findByPlaceholderText("30000");
    expect(screen.getByLabelText("Input type")).toHaveValue("dataset_zip");
    expect(screen.queryByLabelText("Extraction target FPS")).not.toBeInTheDocument();
    expect(screen.queryByRole("option", { name: "Video" })).not.toBeInTheDocument();
    expect(screen.getByLabelText("COLMAP model folder")).toHaveValue("0");
  });
  it("renders approved order with Smart and Balanced defaults", async () => {
    render(<NotebookGeneratorPanel />);
    await screen.findByText("Balanced / L4");
    const headings = screen
      .getAllByRole("heading", { level: 2 })
      .map((node) => node.textContent);
    expect(headings).toEqual([
      "Google Drive input folder",
      "Frame selection",
      "Quality profile",
      "Iterations and maximum Gaussians",
      "Advanced settings",
      "Review",
    ]);
    expect(screen.getByRole("radio", { name: /Smart/ })).toBeChecked();
    expect(screen.getByRole("radio", { name: /Balanced \/ L4/ })).toBeChecked();
    expect(screen.queryByLabelText("Frames per second")).not.toBeInTheDocument();
  });

  it("shows FPS only for Fixed and submits the exact RunSpec", async () => {
    render(<NotebookGeneratorPanel />);
    const folder = await screen.findByLabelText("Drive folder");
    fireEvent.change(folder, { target: { value: "captures/myroom" } });
    fireEvent.click(screen.getByRole("radio", { name: /Fixed FPS/ }));
    expect(screen.getByLabelText("Frames per second")).toHaveValue(4);
    fireEvent.click(
      screen.getByRole("button", { name: "Generate and download notebook" }),
    );
    await waitFor(() =>
      expect(generateStaticNotebook).toHaveBeenCalledWith({
        schema_version: 1,
        input_folder: "captures/myroom",
        frame_selection: { mode: "fixed_fps", fixed_fps: 4 },
        quality: {
          profile: "balanced_l4",
          n_iters: null,
          max_gaussians: null,
          advanced: {},
        },
        publish: { replace_owned_result: true },
      }),
    );
  });

  it("blocks invalid input and explains the correction", async () => {
    render(<NotebookGeneratorPanel />);
    const folder = await screen.findByLabelText("Drive folder");
    fireEvent.change(folder, { target: { value: "../room" } });
    expect(
      screen.getByRole("button", { name: "Generate and download notebook" }),
    ).toBeDisabled();
    expect(screen.getByRole("alert")).toHaveTextContent(/relative folder/i);
  });
});
