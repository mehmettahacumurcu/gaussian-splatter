/**
 * Frontend-only preset metadata.
 *
 * UX hint, NOT a contract with the backend. Backend preset behavior lives in
 * scripts/static_3dgs.py and the dynamic pipeline. Values here are for
 * displaying placeholders, info Details, and PSNR baselines. Drift is fine.
 */
import type { HyperParams, JobMode } from "./api";

export interface PresetMeta {
  useCase: string;
  vram: string;
  diskEstimate: string;
  pickWhen: string[];
  avoidWhen: string[];
  keyHyperparams: string[];
}

export const PRESET_DEFAULTS: Record<JobMode, Record<string, Partial<HyperParams>>> = {
  static: {
    fast:     { iters: 7000,  resolution: "1280x720",  max_gaussians: 100000, lambda_ssim: 0.2 },
    balanced: { iters: 30000, resolution: "1920x1080", max_gaussians: 250000, lambda_ssim: 0.2 },
    high:     { iters: 50000, resolution: "1920x1080", max_gaussians: 500000, lambda_ssim: 0.2 },
    premium:  { iters: 100000,resolution: "2560x1440", max_gaussians: 1000000,lambda_ssim: 0.2 },
  },
  dynamic: {
    micro:       { iters: 200,   resolution: "320x180",  num_timestamps: 5  },
    smoke:       { iters: 500,   resolution: "480x270",  num_timestamps: 10 },
    full:        { iters: 30000, resolution: "640x360",  num_timestamps: 60 },
    high:        { iters: 50000, resolution: "640x360",  num_timestamps: 90, fourier_K: 10, max_gaussians: 60000 },
    ultra:       { iters: 80000, resolution: "720x405",  num_timestamps: 90, fourier_K: 12, hexplane_resolution: 112, mlp_width: 640, mlp_depth: 4, max_gaussians: 80000 },
    ultra_clean: { iters: 80000, resolution: "720x405",  num_timestamps: 90, fourier_K: 12, lambda_aniso: 0.02, dpos_total_cap_frac: 0.05, lambda_rigidity: 0.01 },
    static_max:  { iters: 100000,resolution: "1280x720", num_timestamps: 30, fps: 20, metric3d_model: "metric3d_vit_large", colmap_matching: "exhaustive" },
    cloud:       { iters: 60000, resolution: "1920x1080",num_timestamps: 120 },
  },
};

export const PRESET_BASELINES: Record<JobMode, Record<string, { psnr: number }>> = {
  static: {
    fast: { psnr: 23 }, balanced: { psnr: 26 }, high: { psnr: 28 }, premium: { psnr: 30 },
  },
  dynamic: {
    micro: { psnr: 18 }, smoke: { psnr: 22 }, full: { psnr: 26 }, high: { psnr: 27 },
    ultra: { psnr: 28 }, ultra_clean: { psnr: 28 }, static_max: { psnr: 28 }, cloud: { psnr: 28 },
  },
};

export const PRESET_META: Record<JobMode, Record<string, PresetMeta>> = {
  static: {
    fast: {
      useCase: "Preview / dev iteration",
      vram: "4 GB+",
      diskEstimate: "~400 MB",
      pickWhen: [
        "İlk bakış — pipeline hızlıca çalışıyor mu kontrol",
        "Düşük VRAM (RTX 2060 ve aşağısı)",
      ],
      avoidWhen: ["Sosyal medyaya çıkacak final render"],
      keyHyperparams: ["7k iter", "1280×720", "100k cap", "LPIPS off"],
    },
    balanced: {
      useCase: "Sosyal medya / demo videoları",
      vram: "6 GB+",
      diskEstimate: "~1.5 GB",
      pickWhen: ["Kalite-süre trade-off önemli", "Standart sahneler"],
      avoidWhen: ["Profesyonel rendering — 'high' tercih et"],
      keyHyperparams: ["30k iter", "1920×1080", "250k cap", "LPIPS λ=0.05"],
    },
    high: {
      useCase: "Profesyonel rendering",
      vram: "8 GB+",
      diskEstimate: "~3 GB",
      pickWhen: ["Production-quality static scenes", "≥8GB VRAM kart"],
      avoidWhen: ["Dev iteration", "<8GB VRAM"],
      keyHyperparams: ["50k iter", "1920×1080", "500k cap", "LPIPS λ=0.10", "Multires schedule"],
    },
    premium: {
      useCase: "4dv.ai-tier final çıktı",
      vram: "12 GB+",
      diskEstimate: "~6 GB",
      pickWhen: ["En iyi kalite önemli", "RTX 3090/4090"],
      avoidWhen: ["8GB ve altı kartlar", "Hızlı iterasyon"],
      keyHyperparams: ["100k iter", "2560×1440", "1M cap", "LPIPS λ=0.15", "3-tier multires"],
    },
  },
  dynamic: {
    micro: {
      useCase: "Pre-flight smoke test",
      vram: "4 GB+",
      diskEstimate: "~100 MB",
      pickWhen: ["Pipeline'ı çalıştığını doğrula", "Yeni eklediğin koddan kuşkun var"],
      avoidWhen: ["Demo veya analiz amaçlı kullanma"],
      keyHyperparams: ["200 iter", "320×180", "5 timestamps"],
    },
    smoke: {
      useCase: "Pipeline sağlık kontrolü",
      vram: "4 GB+",
      diskEstimate: "~250 MB",
      pickWhen: ["Bir foundation modeli güncelledin", "Hızlı sanity check"],
      avoidWhen: ["Görsel kaliteye bakacaksan"],
      keyHyperparams: ["500 iter", "480×270", "10 timestamps"],
    },
    full: {
      useCase: "RTX 3060 Ti default — gerçek 4D çıktısı",
      vram: "8 GB+",
      diskEstimate: "~2 GB",
      pickWhen: ["Standart 4D sahne", "8GB civarı kart"],
      avoidWhen: ["12GB+ kartın varsa 'high' kullan"],
      keyHyperparams: ["30k iter", "640×360", "60 timestamps"],
    },
    high: {
      useCase: "Enhanced 4D — daha keskin motion",
      vram: "10 GB+",
      diskEstimate: "~3 GB",
      pickWhen: ["RTX 3080/4070 üstü kart", "Fourier trajectory'i denemek"],
      avoidWhen: ["8GB altı"],
      keyHyperparams: ["50k iter", "640×360", "90 timestamps", "Fourier K=10", "N cap 60k"],
    },
    ultra: {
      useCase: "Yüksek kalite 4D — uzun training",
      vram: "12 GB+",
      diskEstimate: "~5 GB",
      pickWhen: ["RTX 3090/4090", "Final çıktı"],
      avoidWhen: ["12GB altı kartlar — OOM riski"],
      keyHyperparams: ["80k iter", "720×405", "HexPlane 112/56", "MLP 640/4", "K=12", "N cap 80k"],
    },
    ultra_clean: {
      useCase: "Ultra ama anti-streak (v3.8)",
      vram: "12 GB+",
      diskEstimate: "~5 GB",
      pickWhen: ["Streak/spike artifact'ı görüyorsan", "Temiz motion önemli"],
      avoidWhen: ["İlk denemede ultra'yı dene, gerek yoksa atla"],
      keyHyperparams: ["aniso reg", "Δpos clamp", "rigid 5×", "fourier_reg 10×"],
    },
    static_max: {
      useCase: "Eski statik-leaning (legacy)",
      vram: "12 GB+",
      diskEstimate: "~6 GB",
      pickWhen: ["v3.9 davranışını test ediyorsun"],
      avoidWhen: ["Gerçek statik sahne için Static 3D modunu kullan"],
      keyHyperparams: ["fps=20", "vit_large depth", "COLMAP exhaustive"],
    },
    cloud: {
      useCase: "RunPod / RTX 4090 cloud GPU",
      vram: "24 GB",
      diskEstimate: "~8 GB",
      pickWhen: ["Cloud kart kiraladın", "Yüksek çözünürlük gerekli"],
      avoidWhen: ["Local 8GB kartta kullanma — OOM"],
      keyHyperparams: ["60k iter", "1920×1080", "120 timestamps"],
    },
  },
};

export const REFERENCE_PRESET: Record<JobMode, string> = {
  static: "balanced",
  dynamic: "full",
};
