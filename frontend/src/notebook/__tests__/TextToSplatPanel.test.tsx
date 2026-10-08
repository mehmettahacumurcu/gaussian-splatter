import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { generatePipelineNotebook, getPipelineNotebookPresets } from "../../api";
import { TextToSplatPanel } from "../TextToSplatPanel";
import { MAX_DETAIL, MESH_DEFAULTS } from "../textToSplatSettings";
import type { PipelinePresets } from "../PipelineNotebookPanel";

vi.mock("../../api", () => ({ generatePipelineNotebook: vi.fn(), getPipelineNotebookPresets: vi.fn() }));

const training = { iterations: 30000, max_gaussians: 1000000, min_vram: 14, recipe: "medium" };
const text = { min_vram: 20, recipe: "sdxl_trellis", image_steps: 25, sparse_steps: 12, slat_steps: 12, resolution: 1024 };
const catalog: PipelinePresets = { template_version: 2, presets: {
  hybrid: { baseline: training, quality: training, ultra: training },
  spirula: { baseline: training, quality: training, ultra: training },
  text_to_splat: { baseline: text, quality: { ...text, image_steps: 40, sparse_steps: 20, slat_steps: 20 },
    max_detail: { ...text, ...MAX_DETAIL, min_vram: 60 },
    ultra: { ...text, image_steps: 50, sparse_steps: 25, slat_steps: 25, min_vram: 38 } },
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
    edit("GPU profili", "a100");
    edit("Görsel modeli", "sdxl");
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
      image_model: "sdxl", background_model: "birefnet", reconstruction_model: "trellis", gpu_preset: "a100",
      image_steps: 50, image_guidance: 7, image_resolution: 1024, trellis_seed: 42,
      sparse_steps: 25, sparse_cfg: 7.5, slat_steps: 25, slat_cfg: 3,
      ...MESH_DEFAULTS, mesh_views: 48, mesh_fit_iterations: 3000, mesh_splat_cap: 100000,
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

  it.each([
    ["l4", "sdxl", "u2net", "trellis", 25, 7, 12, 24, 1500, 50000],
    ["a100", "flux1_dev", "birefnet", "trellis", 28, 3.5, 20, 48, 3000, 100000],
    ["h100", "qwen_image", "birefnet", "trellis", 40, 4, 24, 64, 4000, 150000],
    ["rtx_pro_6000", "qwen_image", "birefnet", "trellis2", 40, 4, 24, 160, 30000, 1500000],
  ])("applies the %s GPU profile to every stage and submits its budgets", async (gpu, image, background, route, imageSteps, guidance, samplerSteps, views, iterations, cap) => {
    render(<TextToSplatPanel />);
    await screen.findByText(/En az 20 GiB/);
    edit("Nesne tarifi (PROMPT)", "a ceramic teapot");
    edit("GPU profili", String(gpu));
    expect(screen.getByLabelText("Görsel modeli")).toHaveValue(image);
    expect(screen.getByLabelText("Arka plan kaldırma")).toHaveValue(background);
    expect(screen.getByLabelText("3D üretim yolu")).toHaveValue(route);
    expect(download()).toBeEnabled();
    fireEvent.click(download());
    await waitFor(() => expect(generatePipelineNotebook).toHaveBeenCalledWith(expect.objectContaining({
      gpu_preset: gpu, image_model: image, background_model: background, reconstruction_model: route,
      image_steps: imageSteps, image_guidance: guidance, image_resolution: 1024, trellis_seed: 42,
      sparse_steps: samplerSteps, sparse_cfg: 7.5, slat_steps: samplerSteps, slat_cfg: 3,
      mesh_views: views, mesh_fit_iterations: iterations, mesh_splat_cap: cap,
    })));
  });

  it("blocks insufficient VRAM and defaults Blackwell to max detail while explaining an explicit stable override", async () => {
    render(<TextToSplatPanel />);
    await screen.findByText(/En az 20 GiB/);
    edit("Nesne tarifi (PROMPT)", "a car");
    edit("Görsel modeli", "qwen_image");
    expect(screen.getByRole("alert")).toHaveTextContent(/en az 60 GiB.*yeterli değil/);
    expect(download()).toBeDisabled();
    fireEvent.submit(download().closest("form")!);
    expect(generatePipelineNotebook).not.toHaveBeenCalled();
    expect(screen.getByRole("option", { name: /Hunyuan3D-2.x/ })).toBeDisabled();
    expect(screen.getByText(/Bu sürümde çalıştırılmaz/)).toBeVisible();
    edit("GPU profili", "rtx_pro_6000");
    expect(screen.getByRole("option", { name: "TRELLIS-image-large → Gaussian" })).toBeEnabled();
    expect(screen.getByLabelText("3D üretim yolu")).toHaveValue("trellis2");
    edit("3D üretim yolu", "trellis");
    expect(screen.getByRole("note")).toHaveTextContent(/Blackwell GPU’da çalışmaz/);
    expect(download()).toBeEnabled();
    expect(screen.queryByText(/Yardımcı DINOv3 modeli Meta lisanslıdır/)).not.toBeInTheDocument();
    edit("3D üretim yolu", "trellis2");
    expect(screen.queryByRole("note")).not.toBeInTheDocument();
    expect(screen.getByText(/Yardımcı DINOv3 modeli Meta lisanslıdır/)).toBeVisible();
    fireEvent.click(download());
    await waitFor(() => expect(generatePipelineNotebook).toHaveBeenCalledWith(expect.objectContaining({
      gpu_preset: "rtx_pro_6000", reconstruction_model: "trellis2", image_model: "qwen_image", background_model: "birefnet",
      mesh_views: 160, mesh_fit_iterations: 30000, mesh_splat_cap: 1500000,
    })));
  });

  it.each([
    ["flux1_schnell", 4, 0], ["flux2_klein_4b", 4, 1],
  ])("omits a stale negative prompt and locks guidance for %s", async (model, steps, guidance) => {
    render(<TextToSplatPanel />);
    await screen.findByText(/En az 20 GiB/);
    edit("Nesne tarifi (PROMPT)", "a car");
    fireEvent.click(screen.getByText("İsteğe bağlı ayarlar"));
    edit("İstenmeyen özellikler (NEGATIVE)", "blurry");
    edit("GPU profili", "a100");
    edit("Görsel modeli", String(model));
    expect(screen.queryByLabelText("İstenmeyen özellikler (NEGATIVE)")).not.toBeInTheDocument();
    expect(screen.getByLabelText("Görsel guidance / CFG")).toBeDisabled();
    expect(screen.getByLabelText("Görsel adımları")).toHaveValue(steps);
    if (model === "flux1_schnell") {
      edit("Görsel adımları", "5");
      expect(download()).toBeDisabled();
      expect(screen.getByLabelText("Görsel adımları")).toHaveAttribute("max", "4");
      edit("Görsel adımları", "4");
    }
    fireEvent.click(download());
    await waitFor(() => expect(generatePipelineNotebook).toHaveBeenCalledWith(expect.objectContaining({
      image_model: model, negative_prompt: "", image_steps: steps, image_guidance: guidance,
    })));
    edit("Görsel modeli", "sdxl");
    expect(screen.getByLabelText("İstenmeyen özellikler (NEGATIVE)")).toHaveValue("blurry");
  });

  it.each(["flux1_dev", "qwen_image"])("keeps supported negative prompts for %s", async model => {
    render(<TextToSplatPanel />);
    await screen.findByText(/En az 20 GiB/);
    edit("Nesne tarifi (PROMPT)", "a car");
    edit("GPU profili", "a100");
    edit("Görsel modeli", model);
    fireEvent.click(screen.getByText("İsteğe bağlı ayarlar"));
    edit("İstenmeyen özellikler (NEGATIVE)", "watermark");
    fireEvent.click(download());
    await waitFor(() => expect(generatePipelineNotebook).toHaveBeenCalledWith(expect.objectContaining({
      image_model: model, negative_prompt: "watermark",
    })));
  });

  it("validates stage budgets and sends every custom quality control", async () => {
    render(<TextToSplatPanel />);
    await screen.findByText(/En az 20 GiB/);
    edit("Nesne tarifi (PROMPT)", "a car");
    edit("GPU profili", "h100");
    edit("3D üretim yolu", "trellis2");
    fireEvent.click(screen.getByText("Aşama kalite ayarları"));
    expect(screen.queryByLabelText("Hedef splat sayısı")).not.toBeInTheDocument();
    for (const [label, invalid, corrected] of [
      ["Görsel adımları", "0", "30"], ["Görsel guidance / CFG", "20.1", "5.5"],
      ["TRELLIS tohumu", "2147483648", "123"], ["Sparse-structure adımları", "1.5", "18"],
      ["Sparse-structure CFG", "-1", "8.5"], ["SLAT adımları", "101", "22"], ["SLAT CFG", "", "4.5"],
      ["Mesh render çözünürlüğü", "2049", "2048"], ["Mesh SH derecesi", "4", "3"],
      ["Mesh görüş sayısı", "201", "200"], ["3DGS fit iterasyonları", "30001", "30000"],
      ["Mesh splat üst sınırı", "3000001", "3000000"],
    ]) {
      edit(label, invalid);
      expect(download()).toBeDisabled();
      expect(screen.getByLabelText(label)).toHaveAttribute("aria-invalid", "true");
      fireEvent.submit(download().closest("form")!);
      expect(generatePipelineNotebook).not.toHaveBeenCalled();
      edit(label, corrected);
      expect(download()).toBeEnabled();
    }
    edit("Görsel çözünürlüğü", "768");
    fireEvent.click(download());
    await waitFor(() => expect(generatePipelineNotebook).toHaveBeenCalledWith(expect.objectContaining({
      reconstruction_model: "trellis2",
      image_steps: 30, image_guidance: 5.5, image_resolution: 768, trellis_seed: 123,
      sparse_steps: 18, sparse_cfg: 8.5, slat_steps: 22, slat_cfg: 4.5,
      mesh_views: 200, mesh_render_resolution: 2048, mesh_sh_degree: 3, mesh_fit_iterations: 30000, mesh_splat_cap: 3000000, target_splat_count: null,
    })));
  });

  it("selects max detail on A100 and H100 while preserving their stable defaults", async () => {
    render(<TextToSplatPanel />);
    await screen.findByText(/En az 20 GiB/);
    edit("Nesne tarifi (PROMPT)", "a car");
    for (const gpu of ["a100", "h100"]) {
      edit("GPU profili", gpu);
      expect(screen.getByLabelText("3D üretim yolu")).toHaveValue("trellis");
      edit("Kalite ayarı", "max_detail");
      expect(screen.getByLabelText("3D üretim yolu")).toHaveValue("trellis2");
      expect(screen.getByLabelText("TRELLIS.2 pipeline türü")).toHaveValue("1536_cascade");
      expect(screen.getByLabelText("Mesh render çözünürlüğü")).toHaveValue(1536);
      expect(download()).toBeEnabled();
    }
    fireEvent.click(download());
    await waitFor(() => expect(generatePipelineNotebook).toHaveBeenCalledWith(expect.objectContaining({
      ...MAX_DETAIL, preset: "max_detail",
    })));
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
