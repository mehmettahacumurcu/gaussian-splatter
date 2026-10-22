import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { EstimateResponse, SpirulaInfo, SpirulaSettings } from "../spirulaApi";

const api = vi.hoisted(() => ({
  getSpirulaInfo: vi.fn(),
  estimateRun: vi.fn(),
  probeInput: vi.fn(),
  startLocalRun: vi.fn(),
}));
const nb = vi.hoisted(() => ({ generatePipelineNotebook: vi.fn() }));
vi.mock("../spirulaApi", () => api);
vi.mock("../../api", () => nb);
vi.mock("../../notebook/NotebookDownload", () => ({
  NotebookDownload: ({ artifact }: { artifact: { filename: string } }) => <p>indirildi {artifact.filename}</p>,
}));

import { NewSplatPage } from "../NewSplatPage";

const INFO: SpirulaInfo = {
  available: true, binary: "E:\\spirula.exe", version: "spirula 2026.9.24", error: null,
  presets: {
    baseline: { label: "Hızlı", quality: "medium", iterations: 30000, cap_max: 1_000_000, sfm_quality: "high" },
    quality: { label: "Önerilen", quality: "high", iterations: 50000, cap_max: 3_000_000, sfm_quality: "extreme" },
    ultra: { label: "Maksimum detay", quality: "ultra", iterations: 80000, cap_max: 10_000_000, sfm_quality: "extreme" },
  },
  defaults: {} as SpirulaSettings,
  gpu_profiles: {} as SpirulaInfo["gpu_profiles"],
  viewer_port: 7007,
};

/** Same shape as the backend: 600 MiB base + 900 B per splat, 6 GB free locally, 40 GB on A100. */
function fakeEstimate(settings: SpirulaSettings): EstimateResponse {
  const cap = settings.cap_max ?? INFO.presets[settings.preset].cap_max;
  const splats = Math.round((cap * 900) / 2 ** 20);
  const row = (gpu: "local" | "A100", available: number) => ({
    gpu, gpu_label: gpu === "local" ? "Bu PC · RTX 3060 Ti" : "Colab A100", vram_mib: 600 + splats, vram_splats_mib: splats,
    vram_base_mib: 600, available_mib: available, fits: 600 + splats <= available - 600,
    max_splats: Math.floor(((available - 1200) * 2 ** 20) / 900), train_seconds: 3240, train_megapixels: 2.07,
  });
  return {
    resolved: { quality: "high", iterations: 50000, cap_max: cap, fps: 4, sfm_quality: "extreme", geometry_model: "moge2-vitb",
      generate_depth: true, appearance: true, floater_suppression: "off", resolution_divisor: 1, allow_partial: false },
    estimates: [row("local", 6144), row("A100", 40960)],
    gpu: { available: true, error: null, gpus: [{ name: "NVIDIA GeForce RTX 3060 Ti", total_mib: 8192, used_mib: 2048, free_mib: 6144, util_pct: 3 }] },
  };
}

beforeEach(() => {
  window.localStorage.clear();
  Object.values(api).forEach((f) => f.mockReset());
  nb.generatePipelineNotebook.mockReset();
  api.getSpirulaInfo.mockResolvedValue(INFO);
  api.estimateRun.mockImplementation((s: SpirulaSettings) => Promise.resolve(fakeEstimate(s)));
  api.probeInput.mockResolvedValue({ kind: "video", frames: 380, width: 1920, height: 1080, duration: 95 });
});

describe("NewSplatPage", () => {
  it("shows the live VRAM estimate and updates it when the splat cap changes", async () => {
    render(<NewSplatPage />);
    expect(await screen.findByRole("button", { name: /Önerilen/ })).toHaveAttribute("aria-pressed", "true");
    await waitFor(() => expect(screen.getByTestId("vram-estimate")).toHaveTextContent("~3.1 GB"));
    expect(screen.getByText(/Sığar/)).toBeInTheDocument();

    fireEvent.change(screen.getByPlaceholderText("3M"), { target: { value: "50m" } });
    await waitFor(() => expect(screen.getByTestId("vram-estimate")).toHaveTextContent("~42 GB"));
    expect(api.estimateRun.mock.calls.at(-1)![0]).toMatchObject({ cap_max: 50_000_000 });
    expect(screen.getByText(/Sığmaz: bu GPU'da en fazla ~5.8M splat/)).toBeInTheDocument();
    // 50M doesn't fit an A100 40 GB either; the comparison row says so.
    expect(screen.getByRole("cell", { name: /42 GB \/ 40 GB/ })).toHaveClass("bad");
  });

  it("starts a local run with the edited settings", async () => {
    api.startLocalRun.mockResolvedValue({ job_id: "j1", run_id: "r1" });
    const onLocalStarted = vi.fn();
    render(<NewSplatPage onLocalStarted={onLocalStarted} />);
    await screen.findByRole("button", { name: /Önerilen/ });
    fireEvent.change(screen.getByPlaceholderText(/\\inputs\\oda/), { target: { value: "E:\\cekim\\oda.MOV" } });
    expect(await screen.findByText(/~380 kare/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Hızlı/ }));
    fireEvent.click(screen.getByRole("button", { name: "Local başlat" }));
    await waitFor(() => expect(onLocalStarted).toHaveBeenCalledWith("j1", "r1"));
    expect(api.startLocalRun).toHaveBeenCalledWith(expect.objectContaining({
      input_mode: "video", input_path: "E:\\cekim\\oda.MOV", settings: expect.objectContaining({ preset: "baseline", cap_max: null }),
    }));
  });

  it("asks before a local run that does not fit in free VRAM", async () => {
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false);
    render(<NewSplatPage />);
    await screen.findByRole("button", { name: /Önerilen/ });
    fireEvent.change(screen.getByPlaceholderText(/\\inputs\\oda/), { target: { value: "E:\\a.MOV" } });
    fireEvent.click(screen.getByRole("button", { name: "50M" }));
    await waitFor(() => expect(screen.getByTestId("vram-estimate")).toHaveTextContent("~42 GB"));
    await screen.findByText(/~380 kare/);
    fireEvent.click(screen.getByRole("button", { name: "Local başlat" }));
    expect(confirm).toHaveBeenCalled();
    expect(api.startLocalRun).not.toHaveBeenCalled();
    confirm.mockRestore();
  });

  it("generates the notebook with the Drive-relative path of the same input", async () => {
    nb.generatePipelineNotebook.mockResolvedValue({ blob: new Blob(["{}"]), filename: "oda_spirula_quality.ipynb" });
    render(<NewSplatPage />);
    await screen.findByRole("button", { name: /Önerilen/ });
    const notebook = screen.getByRole("button", { name: "Notebook üret" });
    fireEvent.change(screen.getByPlaceholderText(/\\inputs\\oda/), { target: { value: "E:\\yerel\\oda.MOV" } });
    expect(notebook).toBeDisabled();
    fireEvent.change(screen.getByPlaceholderText(/\\inputs\\oda/), { target: { value: "G:\\Drive'ım\\GaussianTests\\oda.MOV" } });
    expect(await screen.findByText("Colab'de MyDrive/GaussianTests/oda.MOV")).toBeInTheDocument();
    fireEvent.click(notebook);
    expect(await screen.findByText(/indirildi oda_spirula_quality/)).toBeInTheDocument();
    expect(nb.generatePipelineNotebook).toHaveBeenCalledWith(expect.objectContaining({
      pipeline: "spirula", input_mode: "video", input_path: "GaussianTests/oda.MOV", preset: "quality",
    }));
  });
});
