/**
 * Static3DSubmit — Statik 3D Gaussian Splatting (photo-set / sparse-view).
 *
 * 4D dynamic features kapali: deformation MLP, Fourier trajectory, motion
 * regularizers, depth/track/flow loss'lari hepsi 0. Sadece RGB + SSIM +
 * (opsiyonel) LPIPS. Backend tarafinda mode='static' gonderilir,
 * api.py cfg.train.static_mode=True ayarlar.
 *
 * Preset'ler scripts/static_3dgs.py PRESETS ile birebir:
 *   fast      —  7k iter,  1280x720, 100k cap,  5-8 dk     | preview
 *   balanced  — 30k iter, 1920x1080, 250k cap, 30-45 dk    | sosyal medya
 *   high      — 50k iter, 1920x1080, 500k cap,  2-3 saat   | profesyonel
 *   premium   —100k iter, 2560x1440,   1M cap, 6-10 saat   | 4dv.ai-tier
 */
import { useMemo, useState } from "react";
import type { HyperParams, ProcessResponse, StaticPreset } from "../api";
import { submitJob } from "../api";
import { HyperparameterPanel } from "./HyperparameterPanel";

interface Props {
  onJobSubmitted: (response: ProcessResponse, sceneName: string) => void;
}

interface PresetSpec {
  id: StaticPreset;
  name: string;
  badge?: string;
  duration: string;
  desc: string;
}

const PRESETS: PresetSpec[] = [
  {
    id: "fast",
    name: "Fast",
    badge: "⚡",
    duration: "5-8 dk",
    desc: "7k iter · 1280×720 · 100k cap · LPIPS off · preview kalite",
  },
  {
    id: "balanced",
    name: "Balanced",
    badge: "⭐",
    duration: "30-45 dk",
    desc: "30k iter · 1920×1080 · 250k cap · LPIPS 0.05 · sosyal medya",
  },
  {
    id: "high",
    name: "High",
    badge: "🎯",
    duration: "2-3 saat",
    desc: "50k iter · 1920×1080 · 500k cap · LPIPS 0.10 · multires schedule · profesyonel",
  },
  {
    id: "premium",
    name: "Premium",
    badge: "🔥",
    duration: "6-10 saat",
    desc: "100k iter · 2560×1440 · 1M cap · LPIPS 0.15 · 3-tier multires · 4dv.ai-tier",
  },
];

export function Static3DSubmit({ onJobSubmitted }: Props) {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [scene, setScene] = useState("");
  const [preset, setPreset] = useState<StaticPreset>("balanced");
  // Static 3DGS — depth supervision ON varsayilan (geometric prior yardimci).
  // Tracks/masks/flow zaten static modda otomatik atlanir; bu toggle sadece
  // depth (Metric3D) calistirilsin mi onu kontrol eder.
  const [useDepthSupervision, setUseDepthSupervision] = useState(true);
  // v6.1 — NVS evaluation (held-out cam metrics + orbit mp4)
  const [nvsEval, setNvsEval] = useState(true);
  const [hyperparams, setHyperparams] = useState<HyperParams>({});
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const handleFile = (f: File | null) => {
    setVideoFile(f);
    if (f && !scene) {
      const base = f.name.replace(/\.[^.]+$/, "");
      const safe = base.replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 40);
      setScene(safe || "scene_" + Date.now().toString(36));
    }
  };

  const videoSize = useMemo(() => {
    if (!videoFile) return null;
    return `${(videoFile.size / 1024 / 1024).toFixed(1)} MB`;
  }, [videoFile]);

  // Static modda video.mp4 zorunlu degil — photo-set kullanildiginda
  // sadece sahne adi yeterli (data/<scene>/images/ klasoru olmali).
  const canSubmit = scene.trim() !== "" && !submitting;

  const handleSubmit = async () => {
    if (!scene.trim()) return;
    setSubmitting(true);
    setSubmitError(null);
    try {
      const res = await submitJob(videoFile, {
        scene: scene.trim(),
        mode: "static",
        preset,
        // depth supervision aktif → foundation phase'in depth adimi calisir
        // (tracks/masks/flow yine de static modda otomatik atlanir).
        skip_foundation: !useDepthSupervision,
        nvs_eval: nvsEval,
        hyperparams,
      });
      onJobSubmitted(res, scene.trim());
    } catch (e) {
      setSubmitError(String(e));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="submit-panel">
      <div className="submit-panel-mode-banner submit-panel-mode-static">
        📸 <strong>Static 3D Mode</strong> — photo set / sparse view / tek frame multi-view.
        4D dynamic features (deformation, Fourier, motion regs) kapali.
      </div>

      <h2 className="submit-title">Yeni Static 3D Job</h2>

      {/* Input — video VEYA photo-set */}
      <div className="submit-section">
        <label className="submit-label">Input (opsiyonel video)</label>
        <div className="file-picker">
          <input
            type="file"
            accept="video/mp4,video/quicktime,video/*"
            onChange={(e) => handleFile(e.target.files?.[0] ?? null)}
            id="video-input-static"
          />
          <label htmlFor="video-input-static" className="file-picker-btn">
            {videoFile ? "Değiştir" : "Video seç (opsiyonel)"}
          </label>
          {videoFile && (
            <div className="file-info">
              <div className="file-name">{videoFile.name}</div>
              <div className="file-size">{videoSize}</div>
            </div>
          )}
        </div>
        <p className="submit-hint">
          <strong>Photo set varsa video gerek yok</strong>:{" "}
          <code>data/&lt;sahne&gt;/images/IMG_*.jpg</code> klasorunu hazirla,
          sahne adini gir, submit. Video varsa frame'lere ayrilir.
        </p>
      </div>

      {/* Sahne adi */}
      <div className="submit-section">
        <label className="submit-label">Sahne adı</label>
        <input
          type="text"
          className="submit-input"
          value={scene}
          onChange={(e) => setScene(e.target.value)}
          placeholder="örn. truck, garden, my_object"
        />
        <p className="submit-hint">
          <code>data/&lt;scene&gt;/</code> altinda calisir. Aynı isim varsa uzerine yazar.
        </p>
      </div>

      {/* Preset */}
      <div className="submit-section">
        <label className="submit-label">Preset</label>
        <div className="preset-row">
          {PRESETS.map((p) => (
            <label
              key={p.id}
              className={`preset-chip ${preset === p.id ? "active" : ""}`}
            >
              <input
                type="radio"
                checked={preset === p.id}
                onChange={() => setPreset(p.id)}
              />
              <div>
                <div className="preset-name">
                  {p.name} {p.badge} <span className="preset-duration">({p.duration})</span>
                </div>
                <div className="preset-desc">{p.desc}</div>
              </div>
            </label>
          ))}
        </div>
      </div>

      {/* Foundation depth supervision toggle */}
      <div className="submit-section">
        <label className="submit-checkbox">
          <input
            type="checkbox"
            checked={useDepthSupervision}
            onChange={(e) => setUseDepthSupervision(e.target.checked)}
          />
          Depth supervision (Metric3D) — onerilir
          <span className="submit-hint-inline">
            — geometric prior + sparse-view yardimcisi. Kapatirsan
            sadece RGB+SSIM+LPIPS supervision.
          </span>
        </label>
        <label className="submit-checkbox" style={{ marginTop: 6 }}>
          <input
            type="checkbox"
            checked={nvsEval}
            onChange={(e) => setNvsEval(e.target.checked)}
          />
          NVS Evaluation — onerilir
          <span className="submit-hint-inline">
            — training sonrasi held-out PSNR/SSIM/LPIPS + orbit mp4. Eval tab'da gorulur.
          </span>
        </label>
      </div>

      {/* Advanced hyperparams (mode='static' filter) */}
      <div className="submit-section">
        <button
          className="btn-secondary advanced-toggle"
          onClick={() => setShowAdvanced(!showAdvanced)}
          type="button"
        >
          {showAdvanced ? "▾" : "▸"} Gelişmiş hiperparametreler
        </button>
        {showAdvanced && (
          <HyperparameterPanel
            value={hyperparams}
            onChange={setHyperparams}
            mode="static"
            preset={preset}
          />
        )}
      </div>

      {/* Submit */}
      <div className="submit-actions">
        <button
          className="btn-primary submit-btn"
          disabled={!canSubmit}
          onClick={handleSubmit}
        >
          {submitting ? "Gönderiliyor..." : "Static 3D Job baslat"}
        </button>
        {!scene.trim() && (
          <div className="submit-hint hint-warn">
            ⚠ Sahne adı zorunlu — submit için doldur.
          </div>
        )}
        {submitError && (
          <div className="submit-error" role="alert">
            <strong>✗ Submit failed:</strong> {submitError}
          </div>
        )}
      </div>
    </div>
  );
}
