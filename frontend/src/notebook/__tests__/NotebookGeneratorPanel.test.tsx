import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { generateStaticNotebook, getStaticNotebookPresets } from "../../api";
import { NotebookGeneratorPanel } from "../NotebookGeneratorPanel";
import type { StaticNotebookPresetsResponse } from "../types";

vi.mock("../../api", () => ({
  getStaticNotebookPresets: vi.fn(),
  generateStaticNotebook: vi.fn(),
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
