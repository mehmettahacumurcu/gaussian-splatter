import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { generatePipelineNotebook, getPipelineNotebookPresets } from "../../api";
import { TextToSplatPanel } from "../TextToSplatPanel";
import type { PipelinePresets } from "../PipelineNotebookPanel";

vi.mock("../../api", () => ({ generatePipelineNotebook: vi.fn(), getPipelineNotebookPresets: vi.fn() }));

const training = { iterations: 30000, max_gaussians: 1000000, min_vram: 14, recipe: "medium" };
const text = { min_vram: 20, recipe: "sdxl_trellis", image_steps: 25, sparse_steps: 12, slat_steps: 12, resolution: 1024 };
const catalog: PipelinePresets = { template_version: 2, presets: {
  hybrid: { baseline: training, quality: training, ultra: training },
  spirula: { baseline: training, quality: training, ultra: training },
  text_to_splat: { baseline: text, quality: { ...text, image_steps: 40 }, ultra: { ...text, min_vram: 38 } },
} };
const download = () => screen.getByRole("button", { name: "Metinden splat notebook’unu indir" });
const edit = (label: string, value: string) => fireEvent.change(screen.getByLabelText(label), { target: { value } });

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(getPipelineNotebookPresets).mockResolvedValue(catalog);
  vi.mocked(generatePipelineNotebook).mockResolvedValue({ blob: new Blob(["notebook"]), filename: "car_text_to_splat.ipynb" });
  Object.defineProperty(URL, "createObjectURL", { configurable: true, value: vi.fn(() => "blob:notebook") });
  Object.defineProperty(URL, "revokeObjectURL", { configurable: true, value: vi.fn() });
  vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
});

describe("TextToSplatPanel", () => {
  it("sends Unicode, quotes and multiline text literally with the selected options", async () => {
    render(<TextToSplatPanel />);
    await screen.findByText(/En az 20 GiB/);
    expect(download()).toBeDisabled();
    const prompt = 'a "red" sports car\nİstanbul — "); __import__("os").system("echo no") #';
    edit("Nesne tarifi (PROMPT)", prompt);
    edit("Kalite ayarı", "ultra");
    edit("Tohum (SEED)", "2147483647");
    fireEvent.click(screen.getByText("İsteğe bağlı ayarlar"));
    edit("İstenmeyen özellikler (NEGATIVE)", "blur\nwatermark");
    edit("Görsel tarzı (STYLE)", 'studio "photo"');
    edit("Drive çıktı klasörü (OUTPUT_DIR)", "MyDrive/GaussianTests/car_result");
    edit("Hedef splat sayısı", "125000");
    expect(screen.getByText(/En az 38 GiB/)).toBeInTheDocument();
    fireEvent.click(download());
    await waitFor(() => expect(generatePipelineNotebook).toHaveBeenCalledWith({
      pipeline: "text_to_splat", input_mode: "text", input_path: "", prompt,
      negative_prompt: "blur\nwatermark", style: 'studio "photo"', seed: 2147483647, preset: "ultra",
      output_dir: "GaussianTests/car_result", target_splat_count: 125000,
    }));
    expect(await screen.findByText("car_text_to_splat.ipynb")).toBeVisible();
    edit("Nesne tarifi (PROMPT)", "a blue car");
    expect(screen.queryByText("car_text_to_splat.ipynb")).not.toBeInTheDocument();
  });

  it("blocks empty or oversized prompts, unsafe paths and non-integer budgets before submitting", async () => {
    render(<TextToSplatPanel />);
    await screen.findByText(/En az 20 GiB/);
    edit("Nesne tarifi (PROMPT)", "   \n ");
    expect(download()).toBeDisabled();
    edit("Nesne tarifi (PROMPT)", "x".repeat(2001));
    expect(download()).toBeDisabled();
    edit("Nesne tarifi (PROMPT)", "a car");
    for (const seed of ["", "-1", "0.5", "2147483648"]) {
      edit("Tohum (SEED)", seed);
      expect(download()).toBeDisabled();
    }
    edit("Tohum (SEED)", "0");
    expect(download()).toBeEnabled();
    fireEvent.click(screen.getByText("İsteğe bağlı ayarlar"));
    edit("Drive çıktı klasörü (OUTPUT_DIR)", "../outside");
    expect(download()).toBeDisabled();
    edit("Drive çıktı klasörü (OUTPUT_DIR)", "");
    for (const count of ["999", "2000001", "1500.5"]) {
      edit("Hedef splat sayısı", count);
      expect(download()).toBeDisabled();
    }
    fireEvent.submit(download().closest("form")!);
    expect(generatePipelineNotebook).not.toHaveBeenCalled();
    edit("Hedef splat sayısı", "1000");
    expect(download()).toBeEnabled();
    edit("İstenmeyen özellikler (NEGATIVE)", "x".repeat(2001));
    expect(download()).toBeDisabled();
    edit("İstenmeyen özellikler (NEGATIVE)", "");
    edit("Görsel tarzı (STYLE)", "x".repeat(501));
    expect(download()).toBeDisabled();
  });

  it("retries a failed catalog without losing the prompt", async () => {
    vi.mocked(getPipelineNotebookPresets).mockRejectedValueOnce(new Error("offline"));
    render(<TextToSplatPanel />);
    await screen.findByText(/Hazır ayarlar yüklenemedi/);
    edit("Nesne tarifi (PROMPT)", "a red sports car");
    expect(download()).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Yeniden dene" }));
    await screen.findByText(/En az 20 GiB/);
    expect(screen.getByLabelText("Nesne tarifi (PROMPT)")).toHaveValue("a red sports car");
    expect(download()).toBeEnabled();
  });

  it("explains an old backend catalog instead of crashing", async () => {
    vi.mocked(getPipelineNotebookPresets).mockResolvedValueOnce({ ...catalog, presets: {
      hybrid: catalog.presets.hybrid, spirula: catalog.presets.spirula,
    } } as PipelinePresets);
    render(<TextToSplatPanel />);
    await screen.findByText(/Backend sürümünü kontrol et/);
    expect(download()).toBeDisabled();
  });

  it("shows generation failures and permits retry", async () => {
    vi.mocked(generatePipelineNotebook).mockRejectedValueOnce(new Error("HTTP 422"));
    render(<TextToSplatPanel />);
    await screen.findByText(/En az 20 GiB/);
    edit("Nesne tarifi (PROMPT)", "a car");
    fireEvent.click(download());
    await screen.findByText(/Notebook oluşturulamadı: Error: HTTP 422/);
    expect(download()).toBeEnabled();
    fireEvent.click(download());
    expect(await screen.findByText("car_text_to_splat.ipynb")).toBeVisible();
  });
});
