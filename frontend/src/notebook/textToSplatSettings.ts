export type ImageModel = "sdxl" | "flux1_dev" | "flux1_schnell" | "flux2_klein_4b" | "qwen_image";
export type BackgroundModel = "u2net" | "birefnet";
export type ReconstructionModel = "trellis" | "trellis2" | "hunyuan3d";
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
    description: "Görselden doğrudan Gaussian üretir; en kısa 3D yol. Ön kontrol eşiği 20 GiB. microsoft/TRELLIS-image-large MIT, açık erişim. Eski CUDA bağımlılıkları nedeniyle RTX PRO 6000 Blackwell desteklenmez.",
  },
  trellis2: {
    label: "TRELLIS.2 → mesh → 3DGS",
    description: "4B model önce mesh üretir; farklı açılardan görüntüler render edilip gsplat ile standart 3DGS PLY eğitilir. Daha uzun sürer; mesh görünüşünü yaklaşık temsil eder. Ön kontrol eşiği 28 GiB; büyük splat bütçesinde 32 GiB. microsoft/TRELLIS.2-4B MIT; gsplat Apache 2.0. Yardımcı DINOv3 modeli Meta lisanslıdır: Hugging Face üzerinden önceden Meta erişim onayı ve Colab secrets/env içinde HF_TOKEN gerekir. CUDA 12.8 derleyicisi gerektiren ilk kurulum uzundur.",
  },
  hunyuan3d: {
    label: "Hunyuan3D-2.x · deneysel / sonra",
    description: "Bu sürümde çalıştırılmaz. Hunyuan3D-2.1 mesh/doku ve özel rasterizer kurulumunun Colab/CUDA uyumu test edilmeli. Resmî kaynak şekil için 10 GB, doku için 21 GB, birlikte 29 GB bildirir; bu pipeline için ölçülmedi. Ağırlıklar açık erişimli, Tencent Community lisansı geçerli; AB, Birleşik Krallık ve Güney Kore bölge kısıtları bulunur. Aynı mesh → 3DGS yoluna bağlantı sonraya bırakıldı.",
  },
};

const shared = { image_resolution: 1024, trellis_seed: 42, sparse_cfg: 7.5, slat_cfg: 3 };

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
    description: "Qwen-Image + BiRefNet + TRELLIS.2 mesh eğitimi; daha yüksek bütçe ve uzun çalışma süresi. GPU modelinin hesabında sunulması ayrıca Colab kullanılabilirliğine bağlıdır.",
    settings: { ...shared, gpu_preset: "h100", image_model: "qwen_image", background_model: "birefnet", reconstruction_model: "trellis2", image_steps: 40, image_guidance: 4, sparse_steps: 24, slat_steps: 24, mesh_views: 64, mesh_fit_iterations: 4000, mesh_splat_cap: 150000 },
  },
  rtx_pro_6000: {
    label: "RTX PRO 6000 · 96 GB", capacity: 96,
    description: "Qwen-Image + BiRefNet + TRELLIS.2; 72 görüş, 5000 fit adımı. Blackwell için modern CUDA ortamı kullanılır; özgün TRELLIS yolu bu GPU’da seçilemez.",
    settings: { ...shared, gpu_preset: "rtx_pro_6000", image_model: "qwen_image", background_model: "birefnet", reconstruction_model: "trellis2", image_steps: 40, image_guidance: 4, sparse_steps: 24, slat_steps: 24, mesh_views: 72, mesh_fit_iterations: 5000, mesh_splat_cap: 200000 },
  },
};
