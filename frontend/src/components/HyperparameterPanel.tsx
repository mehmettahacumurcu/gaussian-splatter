/**
 * HyperparameterPanel — tüm training hiperparametrelerini override eder.
 *
 * Gruplar (her biri collapsible):
 *   - Temel: iters, resolution, num_timestamps, fps
 *   - Loss weights: lambda_ssim + motion regularizers
 *   - Learning rates: lr_deform, lr_means
 *   - Density control: start/end/interval/thresholds
 *   - Model: sh_degree, hexplane, mlp_width
 *
 * Her field "optional override": boş bırakılırsa backend default'u kullanır.
 * Preset (smoke/cloud) seçili ise preset'in yazdığı değer üzerine uygulanır.
 */
import { useEffect, useState } from "react";
import type { HyperParams, JobMode } from "../api";

interface Props {
  value: HyperParams;
  onChange: (next: HyperParams) => void;
  /** v6.0 — Static modda 4D-only grup'lar gizlenir. */
  mode?: JobMode;
  /** Salt bilgi amacli — header'da "X preset default'lari" yazar. */
  preset?: string;
}

interface FieldSpec {
  key: keyof HyperParams;
  label: string;
  placeholder: string;
  help?: string;
  kind: "number" | "string";
  min?: number;
  max?: number;
  step?: number;
}

interface GroupSpec {
  title: string;
  icon: string;
  /** True ise sadece 4D modda gosterilir (deformation/Fourier/motion regs). */
  dynamicOnly?: boolean;
  fields: FieldSpec[];
}

const GROUPS: GroupSpec[] = [
  {
    title: "Temel",
    icon: "⚙",
    fields: [
      { key: "iters", label: "Iterations", placeholder: "preset", kind: "number", min: 50, step: 100 },
      { key: "resolution", label: "Resolution", placeholder: "640x360", kind: "string" },
      { key: "num_timestamps", label: "Timestamps (export)", placeholder: "preset", kind: "number", min: 2, step: 1 },
      { key: "fps", label: "Frame extraction FPS", placeholder: "10", kind: "number", min: 1, step: 1 },
      { key: "warmup_iters", label: "Warmup iters", placeholder: "500", kind: "number", min: 0, step: 100, help: "Regularizer'lar linear 0→full over this many iter. v3 full default: 500 (eski: 2000)" },
    ],
  },
  {
    title: "Loss ağırlıkları (mode-shared)",
    icon: "λ",
    fields: [
      { key: "lambda_ssim", label: "λ SSIM", placeholder: "0.2", kind: "number", step: 0.05, help: "0=L1 only, 1=SSIM only" },
      { key: "lambda_scale", label: "λ scale reg", placeholder: "0.005", kind: "number", step: 0.001, help: "ASIMETRIK hinge: sadece scale > 5% × scene_extent olanları cezalandır (outlier-only). Static + Dynamic." },
      { key: "lambda_depth", label: "λ depth (Metric3D)", placeholder: "0.1", kind: "number", step: 0.05, help: "Scale-invariant L1 between rendered & Metric3D depth. Static modda da kullanilabilir (geometric prior). 0 = depth supervision off." },
    ],
  },
  {
    title: "Motion / 4D loss ağırlıkları",
    icon: "🎬",
    dynamicOnly: true,
    fields: [
      { key: "lambda_deform_reg", label: "λ deform L2", placeholder: "0.0003", kind: "number", step: 0.0001, help: "Δpos/Δquat/Δscale mag regularizer." },
      { key: "lambda_smoothness", label: "λ temporal smoothness", placeholder: "0.002", kind: "number", step: 0.001, help: "D(t) vs D(t+dt)." },
      { key: "lambda_rigidity", label: "λ isometric rigidity", placeholder: "0.002", kind: "number", step: 0.001, help: "Local geometry koruma." },
      { key: "lambda_mask_motion", label: "λ mask-weighted recon", placeholder: "1.0", kind: "number", step: 0.1, help: "Dynamic mask'li bölgelerde reconstruction weight boost." },
      { key: "lambda_track", label: "λ track (CoTracker)", placeholder: "0.1", kind: "number", step: 0.01, help: "CoTracker 3D-anchored track L1." },
      { key: "track_sample_k", label: "Track sample K", placeholder: "256", kind: "number", min: 16, step: 32, help: "Her iter kaç track örneklenir." },
    ],
  },
  {
    title: "Learning rates",
    icon: "↓",
    fields: [
      { key: "lr_means", label: "LR means", placeholder: "0.00016", kind: "number", step: 0.0001 },
    ],
  },
  {
    title: "Learning rates (4D)",
    icon: "↓",
    dynamicOnly: true,
    fields: [
      { key: "lr_deform", label: "LR deformation", placeholder: "0.003", kind: "number", step: 0.0005, help: "v3 default: 3e-3 — MLP motion'u 3× daha hızlı öğrenir." },
      { key: "lr_fourier", label: "LR fourier coeffs", placeholder: "0.005", kind: "number", step: 0.001, help: "Per-Gaussian Fourier katsayıları için ayrı LR." },
    ],
  },
  {
    title: "Density control",
    icon: "●",
    fields: [
      { key: "density_start_iter", label: "Density start iter", placeholder: "preset", kind: "number", min: 0, step: 100 },
      { key: "density_end_iter", label: "Density end iter", placeholder: "preset", kind: "number", min: 0, step: 100, help: "v3 default full'de 22k (eski: 15k) — final prune'lar için" },
      { key: "density_interval", label: "Density interval", placeholder: "100", kind: "number", min: 1, step: 10 },
      { key: "densify_grad_threshold", label: "Densify grad threshold", placeholder: "0.0002", kind: "number", step: 0.0001 },
      { key: "prune_min_opacity", label: "Prune min opacity", placeholder: "0.005", kind: "number", step: 0.001 },
      { key: "prune_max_scale", label: "Prune max scale (fraction)", placeholder: "0.02", kind: "number", step: 0.005, help: "v3.2: scene_extent'in FRACTION'u. Örn scene=70 → 0.02×70=1.4 units cap. Önceden absolute idi, bug'tı" },
      { key: "max_gaussians", label: "Max gauss (N hard cap)", placeholder: "0 (unlimited)", kind: "number", min: 0, step: 10000, help: "v3.7.2: bu sayıya ulaşınca split+clone DURDURULUR, sadece prune devam. 0 = sınırsız (preset'in kendi cap'i). 200000 onerilir 1080p 8GB için." },
      { key: "opacity_reset_interval", label: "Opacity reset aralığı", placeholder: "0 (kapalı)", kind: "number", min: 0, step: 500, help: "v3.1 default: 0 (KAPALI, density control zaten yapıyor). >0 koyarsan her N iter reset yapar" },
    ],
  },
  {
    title: "Model — Gaussian (mode-shared)",
    icon: "◇",
    fields: [
      { key: "sh_degree", label: "SH degree", placeholder: "3", kind: "number", min: 0, max: 3, step: 1 },
    ],
  },
  {
    title: "Model — Deformation MLP / Fourier (4D)",
    icon: "◇",
    dynamicOnly: true,
    fields: [
      { key: "hexplane_resolution", label: "HexPlane resolution", placeholder: "96", kind: "number", min: 16, step: 16 },
      { key: "hexplane_feat_dim", label: "HexPlane feature dim", placeholder: "48", kind: "number", min: 8, step: 8 },
      { key: "mlp_width", label: "Deformation MLP width", placeholder: "512", kind: "number", min: 32, step: 32 },
      { key: "mlp_depth", label: "Deformation MLP depth", placeholder: "4", kind: "number", min: 1, max: 8, step: 1 },
      { key: "num_time_freqs", label: "Fourier time freqs", placeholder: "6", kind: "number", min: 0, max: 12, step: 1, help: "0 = kapalı" },
      { key: "deform_pos_mode", label: "Deform pos mode", placeholder: "hybrid", kind: "string", help: "mlp | fourier | hybrid (default: hybrid)" },
      { key: "fourier_K", label: "Fourier trajectory K", placeholder: "8", kind: "number", min: 0, max: 32, step: 1, help: "Per-Gaussian frekans sayısı (0 = kapalı)" },
      { key: "lambda_fourier_reg", label: "λ fourier reg", placeholder: "0.0001", kind: "number", step: 0.0001, help: "High-freq bastırma." },
    ],
  },
  {
    title: "Foundation modeller (Faz 3, 4D only)",
    icon: "🜚",
    dynamicOnly: true,
    fields: [
      { key: "metric3d_model", label: "Metric3D model", placeholder: "metric3d_vit_small", kind: "string", help: "metric3d_vit_small (default, hızlı) | metric3d_vit_large (daha keskin depth) | metric3d_vit_giant2 (en iyi)" },
      { key: "cotracker_num_points", label: "CoTracker nokta sayısı", placeholder: "2048", kind: "number", min: 256, step: 256 },
      { key: "cotracker_grid_size", label: "CoTracker grid NxN", placeholder: "30", kind: "number", min: 10, max: 60, step: 5 },
      { key: "sam2_threshold", label: "SAM2 threshold", placeholder: "0.5", kind: "number", min: 0, max: 1, step: 0.05 },
    ],
  },
  {
    title: "Preprocessing (v3.9 STATIC MAX)",
    icon: "🎬",
    fields: [
      { key: "resize_long_edge", label: "Frame extract long edge (px)", placeholder: "960", kind: "number", min: 480, step: 160, help: "Default 960. 1280-1440 daha çok detail ama COLMAP yavaşlar." },
      { key: "colmap_matching", label: "COLMAP matching", placeholder: "sequential", kind: "string", help: "sequential (default, hızlı) | exhaustive (yavaş N², orbital camera için loop closure sağlar)" },
      { key: "init_subsample_mode", label: "Init subsample mode", placeholder: "random", kind: "string", help: "random (default) | confidence (track length / reproj error tabanlı, kaliteli noktaları korur)" },
    ],
  },
  {
    title: "v6.1 Quality (mode-shared)",
    icon: "✨",
    fields: [
      { key: "mip_scale_floor_frac", label: "Mip-Splatting scale floor frac", placeholder: "0", kind: "number", step: 0.0005, min: 0, max: 0.01, help: "v6.1 Madde 2-B: 3D scale floor = frac × distance_to_nearest_cam. Anti-aliasing approx. 0.001-0.005 onerilen, 0=off (default). Yakın gauss'larin tek-piksel bozulmasini engeller." },
      { key: "sh_progressive_schedule", label: "SH progressive schedule (0/1)", placeholder: "1", kind: "number", min: 0, max: 1, step: 1, help: "v6.1 Madde 11: SH degree iter'a gore 0→3 progresif. 1=ON (default), 0=OFF (sabit max degree). PSNR +0.3-0.5 dB." },
      { key: "init_method", label: "Init method", placeholder: "colmap", kind: "string", help: "v6.1 Madde 3: colmap (default) | dust3r (sparse-view, ~500MB model) | auto (frame<20 ise dust3r)" },
    ],
  },
  {
    title: "v6.1 Quality (4D only)",
    icon: "✨",
    dynamicOnly: true,
    fields: [
      { key: "lambda_accel", label: "λ accel (2nd-order smoothness)", placeholder: "0", kind: "number", step: 0.0001, min: 0, max: 0.01, help: "v6.1 Madde 6: D(t-1)-2D(t)+D(t+1) ceza. Slow-motion render titremesini azalt. 1e-4 to 5e-4 onerilen, 0=off." },
      { key: "cam_grad_clip_norm", label: "Cam refine grad clip", placeholder: "1.0", kind: "number", step: 0.1, min: 0, help: "v6.1 Madde 12: cam_K + cam_w2c grad norm budget. Default 1.0. 0=off. Large-scale scene cam drift'i azalt." },
      { key: "dynamic_densify_scale", label: "Dynamic densify scale", placeholder: "0.5", kind: "number", step: 0.1, min: 0, max: 2, help: "v6.1 Madde 7: dynamic gauss icin densify_grad_threshold * scale. 0.5 = 2× hassas. 1.0 = no-op. Phase 2.1 split'le birlikte calisir." },
    ],
  },
  {
    title: "Anti-streak (4D only)",
    icon: "★",
    dynamicOnly: true,
    fields: [
      { key: "lambda_aniso", label: "λ anisotropy", placeholder: "0", kind: "number", step: 0.005, help: "max/min scale ratio threshold üstü gauss'ları cezalandır. 0=kapalı, 0.02=Ultra Clean default" },
      { key: "aniso_threshold", label: "Aniso threshold", placeholder: "5", kind: "number", min: 1, step: 0.5, help: "max/min < threshold serbest. Düşük=daha agresif streak fix." },
      { key: "dpos_total_cap_frac", label: "Δpos total cap fraction", placeholder: "0.2", kind: "number", min: 0.01, max: 0.5, step: 0.01, help: "Per-iter motion cap × scene_extent. Default 0.2; Ultra Clean 0.05 (4× sıkı)." },
    ],
  },
];

export function HyperparameterPanel({ value, onChange, mode = "dynamic", preset }: Props) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set(["Temel"]));

  // Static modda 4D-only grup'lari gizle
  const visibleGroups = GROUPS.filter((g) => {
    if (mode === "static" && g.dynamicOnly) return false;
    return true;
  });

  const toggle = (title: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(title)) next.delete(title);
      else next.add(title);
      return next;
    });
  };

  const overrideCount = Object.values(value).filter(
    (v) => v !== null && v !== undefined && v !== ""
  ).length;

  return (
    <div className="hparam-panel">
      <div className="hparam-panel-header">
        <span>
          Hiperparametreler{" "}
          <span className="hparam-mode-tag">
            {mode === "static" ? "📸 Static" : "🎬 Dynamic"}
          </span>
        </span>
        <span className="hparam-summary">
          {overrideCount === 0
            ? `${preset ?? "default"} preset default'ları`
            : `${overrideCount} override`}
        </span>
      </div>
      {visibleGroups.map((g) => {
        const isOpen = expanded.has(g.title);
        const groupOverrides = g.fields.filter(
          (f) => value[f.key] !== null && value[f.key] !== undefined && value[f.key] !== ""
        ).length;
        return (
          <div key={g.title} className={`hparam-group ${isOpen ? "open" : ""}`}>
            <button
              className="hparam-group-header"
              onClick={() => toggle(g.title)}
              type="button"
            >
              <span className="hparam-group-icon">{g.icon}</span>
              <span className="hparam-group-title">{g.title}</span>
              {groupOverrides > 0 && (
                <span className="hparam-badge">{groupOverrides}</span>
              )}
              <span className="hparam-chevron">{isOpen ? "▾" : "▸"}</span>
            </button>
            {isOpen && (
              <div className="hparam-fields">
                {g.fields.map((f) => (
                  <NumberOrTextField
                    key={f.key}
                    spec={f}
                    value={value[f.key] as any}
                    onChange={(val) => onChange({ ...value, [f.key]: val })}
                  />
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}


// ---------------------------------------------------------------------------
// NumberOrTextField — locale-safe input (text + decimal mode)
// HTML5 number input Turkce Windows'ta nokta-decimal'i ("0.001") reddediyor.
// Local raw text buffer ile yarim sayilarin (0., 0.0, 0.00) korunmasini saglar.
// ---------------------------------------------------------------------------
function NumberOrTextField({
  spec,
  value,
  onChange,
}: {
  spec: FieldSpec;
  value: number | string | null | undefined;
  onChange: (val: number | string | null) => void;
}) {
  const isNum = spec.kind === "number";
  // Local raw text buffer — controlled input ama yarim sayilar kayboldurmasin
  const [raw, setRaw] = useState<string>(
    value === null || value === undefined ? "" : String(value),
  );
  // Dis prop value degistiginde raw'i senkronize et (preset secimi vs.)
  useEffect(() => {
    const ext = value === null || value === undefined ? "" : String(value);
    // Sadece kullanici-girisi olmayan dis degisikliklerde set et:
    // raw'in parse'i value ile eslesiyorsa zaten senkron; aksi halde override.
    const parsed = isNum ? parseFloat(raw.replace(",", ".")) : NaN;
    const sameAsValue =
      (raw === "" && (value === null || value === undefined)) ||
      (isNum && Number.isFinite(parsed) && parsed === value) ||
      (!isNum && raw === value);
    if (!sameAsValue) setRaw(ext);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const r = e.target.value;
    setRaw(r);  // Her zaman ham metni koru
    if (r === "") {
      onChange(null);
      return;
    }
    if (isNum) {
      const normalized = r.replace(",", ".").trim();
      if (normalized === "-" || normalized === "." || normalized === "-.") {
        // Yarim sayi — parent value degistirme, kullanici devam ediyor
        return;
      }
      const n = parseFloat(normalized);
      if (Number.isFinite(n)) onChange(n);
    } else {
      onChange(r);
    }
  };

  return (
    <label className="hparam-field">
      <span className="hparam-label" title={spec.help}>
        {spec.label}
      </span>
      <input
        type="text"
        inputMode={isNum ? "decimal" : "text"}
        placeholder={spec.placeholder}
        value={raw}
        onChange={handleChange}
      />
    </label>
  );
}
