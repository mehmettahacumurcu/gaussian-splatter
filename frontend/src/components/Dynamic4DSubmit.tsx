/**
 * Dynamic4DSubmit — 4D Gaussian Splatting (zaman-dinamik sahneler).
 *
 * Video input zorunlu (frame extraction → COLMAP → foundation → training).
 * Multi-view auto-detect: data/<scene>/videos/cam*.mp4 varsa N3V mode aktif.
 *
 * Preset'ler backend api.py /process'teki mevcut preset listesini kullanir
 * (eski boolean flag yerine v6.0 mode='dynamic' + preset string).
 */
import { useMemo, useState } from "react";
import type { DynamicPreset, HyperParams, ProcessResponse } from "../api";
import { submitJob } from "../api";
import { HyperparameterPanel } from "./HyperparameterPanel";

interface Props {
  onJobSubmitted: (response: ProcessResponse, sceneName: string) => void;
}

interface PresetSpec {
  id: DynamicPreset;
  name: string;
  badge?: string;
  duration: string;
  desc: string;
}

const PRESETS: PresetSpec[] = [
  {
    id: "micro",
    name: "Micro",
    badge: "⚡",
    duration: "30 sn–10 dk",
    desc: "200 iter · 320×180 · 5 ts · dev iteration / preflight smoke",
  },
  {
    id: "smoke",
    name: "Smoke",
    badge: "💨",
    duration: "5-15 dk",
    desc: "500 iter · 480×270 · 10 ts · pipeline saglik check",
  },
  {
    id: "full",
    name: "Full",
    badge: "🟢",
    duration: "30-60 dk",
    desc: "30k iter · 640×360 · 60 ts · 3060 Ti default",
  },
  {
    id: "safe_4d_8gb",
    name: "Safe 4D (8GB)",
    badge: "🛡",
    duration: "30-60 dk",
    desc: "Full ile esit sure. Stabil deformation: lr_deform/fourier 1e-3, fourier_reg 5x, dpos cap 0.08. LPIPS off → RAM paging yok. PSNR collapse fix.",
  },
  {
    id: "high",
    name: "High",
    badge: "⭐",
    duration: "3-4 saat",
    desc: "50k iter · 640×360 · 90 ts · Fourier K=10 · N cap 60k · enhanced",
  },
  {
    id: "ultra",
    name: "Ultra",
    badge: "🔥",
    duration: "6-9 saat",
    desc: "80k iter · 720×405 · 90 ts · HexPlane 112/56 · MLP 640/4 · K=12 · N cap 80k",
  },
  {
    id: "ultra_clean",
    name: "Ultra Clean",
    badge: "✨",
    duration: "7-9 saat",
    desc: "v3.8 anti-streak · aniso reg + sıkı dpos clamp + rigid 5× + fourier_reg 10×",
  },
  {
    id: "static_max",
    name: "Static Max (legacy)",
    badge: "🎯",
    duration: "10-13 saat",
    desc: "v3.9 4D ama statik-leaning · fps=20 + vit_large depth + COLMAP exhaustive. NOT: gercek statik icin Static 3D modunu kullan.",
  },
  {
    id: "cloud",
    name: "Cloud",
    badge: "☁",
    duration: "RunPod / RTX 4090",
    desc: "60k iter · 1920×1080 · 120 ts · cloud GPU",
  },
];

export function Dynamic4DSubmit({ onJobSubmitted }: Props) {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [scene, setScene] = useState("");
  const [preset, setPreset] = useState<DynamicPreset>("smoke");
  const [skipFoundation, setSkipFoundation] = useState(false);
  // v6.1 — NVS evaluation
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

  // 4D modda multi-view scene icin video upload zorunlu degil (sahne onceden
  // load_n3v.py ile import edilmis olabilir). Sadece sahne adi yeterli.
  const canSubmit = scene.trim() !== "" && !submitting;

  const handleSubmit = async () => {
    if (!scene.trim()) return;
    setSubmitting(true);
    setSubmitError(null);
    try {
      const res = await submitJob(videoFile, {
        scene: scene.trim(),
        mode: "dynamic",
        preset,
        skip_foundation: skipFoundation,
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
      <div className="submit-panel-mode-banner submit-panel-mode-dynamic">
        🎬 <strong>4D Dynamic Mode</strong> — zaman-dinamik sahneler. Deformation MLP +
        per-Gaussian Fourier trajectory + foundation modeller (depth/track/flow) aktif.
        Multi-view auto-detect: <code>data/&lt;sahne&gt;/videos/cam*.mp4</code> varsa N3V mode.
      </div>

      <h2 className="submit-title">Yeni 4D Dynamic Job</h2>

      {/* Video */}
      <div className="submit-section">
        <label className="submit-label">Video dosyası</label>
        <div className="file-picker">
          <input
            type="file"
            accept="video/mp4,video/quicktime,video/*"
            onChange={(e) => handleFile(e.target.files?.[0] ?? null)}
            id="video-input-dynamic"
          />
          <label htmlFor="video-input-dynamic" className="file-picker-btn">
            {videoFile ? "Değiştir" : "Video seç"}
          </label>
          {videoFile && (
            <div className="file-info">
              <div className="file-name">{videoFile.name}</div>
              <div className="file-size">{videoSize}</div>
            </div>
          )}
        </div>
        <p className="submit-hint">
          MP4 veya MOV — kamera hareketli, sahnede belirgin doku. HyperNeRF dataset:{" "}
          <code>scripts/hypernerf_to_mp4.py</code> ile mp4'e cevir. Multi-view scene
          icin sahne onceden hazir ise video upload skip edilebilir.
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
          placeholder="örn. banana_demo, flame_steak"
        />
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

      {/* Foundation toggle */}
      <div className="submit-section">
        <label className="submit-checkbox">
          <input
            type="checkbox"
            checked={skipFoundation}
            onChange={(e) => setSkipFoundation(e.target.checked)}
          />
          Foundation modelleri atla (Metric3D / CoTracker / Farneback / RAFT)
          <span className="submit-hint-inline">
            — micro/smoke preset'leri icin onerilir
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

      {/* Advanced */}
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
            mode="dynamic"
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
          {submitting ? "Gönderiliyor..." : "4D Dynamic Job baslat"}
        </button>
        {!scene.trim() && (
          <div className="submit-hint hint-warn">
            ⚠ Sahne adı zorunlu — submit için doldur.
          </div>
        )}
        {scene.trim() && !videoFile && (
          <div className="submit-hint hint-warn">
            ⚠ Video dosyası seçilmedi. Sahne backend'de yoksa submit 400 ile reddedilir.
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
