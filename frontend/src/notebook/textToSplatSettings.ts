export type ImageModel = "sdxl" | "flux1_dev" | "flux1_schnell" | "flux2_klein_4b" | "qwen_image";
export type BackgroundModel = "u2net" | "birefnet";
export type ReconstructionModel = "trellis" | "trellis2" | "hunyuan3d";
export type Trellis2PipelineType = "512" | "1024_cascade" | "1536_cascade";
export type GpuPreset = "l4" | "a100" | "h100" | "rtx_pro_6000";

export interface TextToSplatModelSettings {
  image_model: ImageModel;
  background_model: BackgroundModel;
  reconstruction_model: ReconstructionModel;
  gpu_preset: GpuPreset;
  image_steps: number;
  image_guidance: number;
  image_resolution: number;
  trellis_seed: number;
  sparse_steps: number;
  sparse_cfg: number;
  slat_steps: number;
  slat_cfg: number;
  trellis2_pipeline_type: Trellis2PipelineType;
  mesh_render_resolution: number;
  mesh_sh_degree: number;
  mesh_texture_size: number;
  mesh_views: number;
  mesh_fit_iterations: number;
  mesh_splat_cap: number;
}

export const IMAGE_MODELS: Record<ImageModel, {
  label: string; description: string; steps: number; guidance: number; negative: boolean; vram: number;
}> = {
  sdxl: {
    label: "SDXL base 1.0",
    description: "Dengeli kalite ve hız; 1024 px için ön kontrol eşiği 12 GiB. CreativeML Open RAIL++-M kullanım koşulları geçerli; açık erişim, HF_TOKEN gerekmez. Negatif tarif desteklenir.",
    steps: 25, guidance: 7, negative: true, vram: 12,
  },
  flux1_dev: {
    label: "FLUX.1-dev · 12B",
    description: "12 milyar parametre; ayrıntı ve tarif uyumu yüksek, SDXL’den yavaş. Ön kontrol eşiği 36 GiB. FLUX.1-dev non-commercial lisansı; Hugging Face erişim koşullarını kabul edip Colab secrets/env içinde HF_TOKEN tanımla. Negatif tarif true CFG ile desteklenir, ek süre ve bellek kullanır.",
    steps: 28, guidance: 3.5, negative: true, vram: 36,
  },
  flux1_schnell: {
    label: "FLUX.1-schnell · 12B",
    description: "12 milyar parametre; 1–4 adımda hızlı önizleme, dev kadar ince kontrol sunmaz. Ön kontrol eşiği 36 GiB. Apache 2.0; Hugging Face erişim koşullarını kabul edip Colab secrets/env içinde HF_TOKEN tanımla. Negatif tarif ve guidance kullanılmaz.",
    steps: 4, guidance: 0, negative: false, vram: 36,
  },
  flux2_klein_4b: {
    label: "FLUX.2 [klein] · 4B",
    description: "4 milyar parametre; 4 adımlı damıtılmış model, hız ve düşük bellek öncelikli. Ön kontrol eşiği 16 GiB. Apache 2.0; açık erişim, HF_TOKEN gerekmez. Negatif tarif desteklenmez; guidance 1 olarak sabittir.",
    steps: 4, guidance: 1, negative: false, vram: 16,
  },
  qwen_image: {
    label: "Qwen-Image · 20B",
    description: "20 milyar parametre; karmaşık tarif ve görsel metin üretiminde güçlü, yavaş ve bellek ihtiyacı yüksek. Ön kontrol eşiği 60 GiB. Apache 2.0; açık erişim, HF_TOKEN gerekmez. Negatif tarif ve true CFG desteklenir.",
    steps: 40, guidance: 4, negative: true, vram: 60,
  },
};

export const BACKGROUND_MODELS: Record<BackgroundModel, { label: string; description: string }> = {
  u2net: {
    label: "U2Net · hızlı",
    description: "Hızlı ve hafif arka plan ayırma; ince saç ve yarı saydam kenarlarda ayrıntı kaybedebilir. CPU’da çalışır, ek GPU belleği gerekmez. rembg kodu MIT, U²-Net kaynak projesi Apache 2.0; dağıtılan ağırlığın ayrı lisans metni doğrulanmadı.",
  },
  birefnet: {
    label: "BiRefNet · ayrıntılı kenarlar",
    description: "220 milyon parametre; saç, kürk ve ince kenarları daha iyi korumayı hedefler, U2Net’ten yavaştır. Görsel modeli bellekten çıktıktan sonra GPU’da çalışır; maske aşaması profilin bellek eşiğine dahildir. ZhengPeng7/BiRefNet ağırlıkları MIT; açık erişim, HF_TOKEN gerekmez.",
  },
};

export const RECONSTRUCTION_MODELS: Record<ReconstructionModel, { label: string; description: string }> = {
  trellis: {
    label: "TRELLIS-image-large → Gaussian",
    description: "L4/A100/H100 profillerinde kararlı varsayılan yol; görselden doğrudan Gaussian üretir. Ön kontrol eşiği 20 GiB. microsoft/TRELLIS-image-large MIT, açık erişim. Eski CUDA bağımlılıkları nedeniyle RTX PRO 6000 Blackwell desteklenmez.",
  },
  trellis2: {
    label: "TRELLIS.2 → mesh → 3DGS · deneysel",
    description: "Yüksek çözünürlüklü deneysel yol; temiz Colab GPU oturumunda uçtan uca doğrulanmadı. 4B model önce mesh üretir; farklı açılardan görüntüler render edilip gsplat ile standart 3DGS PLY eğitilir. Daha uzun sürer; mesh görünüşünü yaklaşık temsil eder. 512/1024 cascade/1536 cascade için geometri eşikleri 28/40/60 GiB; render ve splat bütçesi daha fazla bellek isteyebilir. microsoft/TRELLIS.2-4B MIT; gsplat Apache 2.0. Yardımcı DINOv3 modeli Meta lisanslıdır: Hugging Face üzerinden önceden Meta erişim onayı ve Colab secrets/env içinde HF_TOKEN gerekir. CUDA nvcc ve g++ derlemesi gerektiren ilk kurulum uzundur.",
  },
  hunyuan3d: {
    label: "Hunyuan3D-2.x · deneysel / sonra",
    description: "Bu sürümde çalıştırılmaz. Hunyuan3D-2.1 mesh/doku ve özel rasterizer kurulumunun Colab/CUDA uyumu test edilmeli. Resmî kaynak şekil için 10 GB, doku için 21 GB, birlikte 29 GB bildirir; bu pipeline için ölçülmedi. Ağırlıklar açık erişimli, Tencent Community lisansı geçerli; AB, Birleşik Krallık ve Güney Kore bölge kısıtları bulunur. Aynı mesh → 3DGS yoluna bağlantı sonraya bırakıldı.",
  },
};

export const MESH_DEFAULTS = { trellis2_pipeline_type: "512" as Trellis2PipelineType, mesh_render_resolution: 1024, mesh_sh_degree: 2, mesh_texture_size: 2048 };
export const MAX_DETAIL = { reconstruction_model: "trellis2" as ReconstructionModel, trellis2_pipeline_type: "1536_cascade" as Trellis2PipelineType, mesh_render_resolution: 1536, mesh_texture_size: 4096, mesh_sh_degree: 2, sparse_steps: 24, slat_steps: 24, mesh_views: 160, mesh_fit_iterations: 30000, mesh_splat_cap: 1500000 };
export const PIPELINE_VRAM = { "512": 28, "1024_cascade": 40, "1536_cascade": 60 };

const shared = { ...MESH_DEFAULTS, image_resolution: 1024, trellis_seed: 42, sparse_cfg: 7.5, slat_cfg: 3 };

export const GPU_PRESETS: Record<GpuPreset, {
  label: string; description: string; capacity: number; settings: TextToSplatModelSettings;
}> = {
  l4: {
    label: "L4 · 24 GB", capacity: 24,
    description: "SDXL + U2Net + doğrudan TRELLIS; kısa denemeler için dengeli. Profil seçimi aşağıdaki modelleri ve kalite bütçelerini değiştirir.",
    settings: { ...shared, gpu_preset: "l4", image_model: "sdxl", background_model: "u2net", reconstruction_model: "trellis", image_steps: 25, image_guidance: 7, sparse_steps: 12, slat_steps: 12, mesh_views: 24, mesh_fit_iterations: 1500, mesh_splat_cap: 50000 },
  },
  a100: {
    label: "A100 · 80 GB", capacity: 80,
    description: "FLUX.1-dev + BiRefNet + doğrudan TRELLIS; ayrıntı odaklı. FLUX.1-dev için lisans kabulü ve HF_TOKEN gerekir; A100 40 GB için bu profil önerilmez.",
    settings: { ...shared, gpu_preset: "a100", image_model: "flux1_dev", background_model: "birefnet", reconstruction_model: "trellis", image_steps: 28, image_guidance: 3.5, sparse_steps: 20, slat_steps: 20, mesh_views: 48, mesh_fit_iterations: 3000, mesh_splat_cap: 100000 },
  },
  h100: {
    label: "H100 · 80 GB", capacity: 80,
    description: "Qwen-Image + BiRefNet + kararlı TRELLIS varsayılanı; daha yüksek görsel ve örnekleme bütçesi. TRELLIS.2 deneysel yolunu ayrıca seçebilirsin. GPU modelinin hesabında sunulması Colab kullanılabilirliğine bağlıdır.",
    settings: { ...shared, gpu_preset: "h100", image_model: "qwen_image", background_model: "birefnet", reconstruction_model: "trellis", image_steps: 40, image_guidance: 4, sparse_steps: 24, slat_steps: 24, mesh_views: 64, mesh_fit_iterations: 4000, mesh_splat_cap: 150000 },
  },
  rtx_pro_6000: {
    label: "RTX PRO 6000 · 96 GB", capacity: 96,
    description: "Qwen-Image + BiRefNet + TRELLIS.2 Max detay. Blackwell için modern cu128 ortamı; 1536 cascade, 4096 doku, 160 görüş, 1.5M splat sınırı, 30k adım, SH2. En az 60 GiB; ilk derleme ve eğitim saatler sürebilir, süre ölçülmedi. DINOv3 erişim onayı gerekir.",
    settings: { ...shared, gpu_preset: "rtx_pro_6000", image_model: "qwen_image", background_model: "birefnet", image_steps: 40, image_guidance: 4, ...MAX_DETAIL },
  },
};
// Defaults fill older saved settings; unknown selectors are ignored during hydration.
// API validation remains strict and returns a validation error for invalid requests.
export function restoreTextToSplatSettings(saved: Partial<TextToSplatModelSettings>): TextToSplatModelSettings {
  const gpu = saved.gpu_preset && Object.prototype.hasOwnProperty.call(GPU_PRESETS, saved.gpu_preset) ? saved.gpu_preset : "l4";
  const result = { ...GPU_PRESETS[gpu].settings };
  const selectors = { gpu_preset: GPU_PRESETS, image_model: IMAGE_MODELS,
    background_model: BACKGROUND_MODELS, reconstruction_model: RECONSTRUCTION_MODELS,
    trellis2_pipeline_type: PIPELINE_VRAM };
  for (const [key, value] of Object.entries(saved)) {
    if (!Object.prototype.hasOwnProperty.call(result, key)) continue;
    if (Object.prototype.hasOwnProperty.call(selectors, key)) {
      const options = selectors[key as keyof typeof selectors];
      if (typeof value === "string" && Object.prototype.hasOwnProperty.call(options, value) && value !== "hunyuan3d") Object.assign(result, { [key]: value });
    } else if (typeof value === "number" && Number.isFinite(value)) Object.assign(result, { [key]: value });
  }
  return result;
}

export function estimateTextRequirements(settings: TextToSplatModelSettings, preset: string) {
  const mesh = settings.reconstruction_model === "trellis2";
  const millions = settings.mesh_splat_cap / 1000000;
  const fit = Math.ceil(12 + 6 * millions + 4 * (settings.mesh_render_resolution / 1024) ** 2
    + .25 * millions * (settings.mesh_sh_degree + 1) ** 2);
  const geometry = mesh ? Math.max(PIPELINE_VRAM[settings.trellis2_pipeline_type] ?? 28, fit) : 20;
  const minimum = Math.max(IMAGE_MODELS[settings.image_model].vram, geometry, preset === "ultra" ? 38 : 0);
  const capacity = GPU_PRESETS[settings.gpu_preset].capacity;
  const lower = Object.entries(PIPELINE_VRAM).filter(([, vram]) => mesh
    && vram <= PIPELINE_VRAM[settings.trellis2_pipeline_type]
    && Math.max(vram, fit, IMAGE_MODELS[settings.image_model].vram) <= capacity).at(-1)?.[0];
  return { minimum, lower };
}
