/**
 * Static3DSubmit — Statik 3D Gaussian Splatting (photo-set / sparse-view).
 *
 * 4D dynamic features kapali: deformation MLP, Fourier trajectory, motion
 * regularizers, depth/track/flow loss'lari hepsi 0. Sadece RGB + SSIM +
 * (opsiyonel) LPIPS. Backend tarafinda mode='static' gonderilir.
 */
import { useMemo, useState } from "react";
import type { HyperParams, ProcessResponse, StaticPreset } from "../api";
import { submitJob } from "../api";
import { HyperparameterPanel } from "./HyperparameterPanel";
import { Card } from "./ui/Card";
import { PresetChip } from "./PresetChip";

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
  { id: "fast",     name: "Fast",     badge: "⚡", duration: "5-8 dk",   desc: "7k iter · 1280×720 · 100k cap · LPIPS off · preview kalite" },
  { id: "balanced", name: "Balanced", badge: "⭐", duration: "30-45 dk", desc: "30k iter · 1920×1080 · 250k cap · LPIPS 0.05 · sosyal medya" },
  { id: "high",     name: "High",     badge: "🎯", duration: "2-3 saat", desc: "50k iter · 1920×1080 · 500k cap · LPIPS 0.10 · multires schedule · profesyonel" },
  { id: "premium",  name: "Premium",  badge: "🔥", duration: "6-10 saat",desc: "100k iter · 2560×1440 · 1M cap · LPIPS 0.15 · 3-tier multires · 4dv.ai-tier" },
];

export function Static3DSubmit({ onJobSubmitted }: Props) {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [scene, setScene] = useState("");
  const [preset, setPreset] = useState<StaticPreset>("balanced");
  const [useDepthSupervision, setUseDepthSupervision] = useState(true);
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
      <h2 className="submit-title">Yeni Static 3D Job</h2>
      <p className="submit-context-line">
        📸 Photo set / sparse view — 4D dynamic features (deformation, Fourier, motion regs) kapalı.
      </p>

      <Card title="Input">
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
          <code>data/&lt;sahne&gt;/images/IMG_*.jpg</code> klasörünü hazırla,
          sahne adını gir, submit. Video varsa frame'lere ayrılır.
        </p>
      </Card>

      <Card title="Sahne adı">
        <input
          type="text"
          className="submit-input"
          value={scene}
          onChange={(e) => setScene(e.target.value)}
          placeholder="örn. truck, garden, my_object"
        />
        <p className="submit-hint">
          <code>data/&lt;scene&gt;/</code> altında çalışır. Aynı isim varsa üzerine yazar.
        </p>
      </Card>

      <Card title="Preset">
        <div className="preset-row">
          {PRESETS.map((p) => (
            <PresetChip
              key={p.id}
              pipeline="static"
              presetId={p.id}
              name={p.name}
              badge={p.badge}
              duration={p.duration}
              desc={p.desc}
              selected={preset === p.id}
              onSelect={() => setPreset(p.id)}
            />
          ))}
        </div>
      </Card>

      <Card title="Options">
        <label className="submit-checkbox">
          <input
            type="checkbox"
            checked={useDepthSupervision}
            onChange={(e) => setUseDepthSupervision(e.target.checked)}
          />
          Depth supervision (Metric3D) — önerilir
          <span className="submit-hint-inline">
            — geometric prior + sparse-view yardımcısı. Kapatırsan sadece RGB+SSIM+LPIPS.
          </span>
        </label>
        <label className="submit-checkbox" style={{ marginTop: 6 }}>
          <input
            type="checkbox"
            checked={nvsEval}
            onChange={(e) => setNvsEval(e.target.checked)}
          />
          NVS Evaluation — önerilir
          <span className="submit-hint-inline">
            — held-out PSNR/SSIM/LPIPS + orbit mp4. Eval tab'ında görülür.
          </span>
        </label>
      </Card>

      <Card
        title="Hiperparametreler"
        actions={
          <button
            className="btn-secondary"
            onClick={() => setShowAdvanced(!showAdvanced)}
            type="button"
          >
            {showAdvanced ? "▾ Gizle" : "▸ Göster"}
          </button>
        }
      >
        {showAdvanced ? (
          <HyperparameterPanel
            value={hyperparams}
            onChange={setHyperparams}
            mode="static"
            preset={preset}
          />
        ) : (
          <p className="submit-hint">
            "{preset}" preset default'larını kullan. Override eklemek için Göster.
          </p>
        )}
      </Card>

      <div className="submit-actions">
        <button
          className="btn-primary submit-btn"
          disabled={!canSubmit}
          onClick={handleSubmit}
        >
          {submitting ? "Gönderiliyor..." : "Static 3D Job başlat"}
        </button>
        {submitError && <div className="submit-error">Hata: {submitError}</div>}
      </div>
    </div>
  );
}
