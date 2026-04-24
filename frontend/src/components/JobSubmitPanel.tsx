/**
 * JobSubmitPanel — video seç + sahne adı + preset + hyperparams + submit.
 *
 * HTML file input Tauri WebView'da native Explorer dialog'u açar,
 * ayrı plugin kurulumu gerekmez. Dosya seçilince File objesi elde edilir,
 * submitJob() FormData ile upload eder.
 */
import { useMemo, useState } from "react";
import type { HyperParams, ProcessResponse } from "../api";
import { submitJob } from "../api";
import { HyperparameterPanel } from "./HyperparameterPanel";

interface Props {
  onJobSubmitted: (response: ProcessResponse, sceneName: string) => void;
}

type Preset = "smoke" | "full" | "cloud";

export function JobSubmitPanel({ onJobSubmitted }: Props) {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [scene, setScene] = useState("");
  const [preset, setPreset] = useState<Preset>("smoke");
  const [skipFoundation, setSkipFoundation] = useState(false);
  const [hyperparams, setHyperparams] = useState<HyperParams>({});
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  // Video seçilince default scene name öner (dosya adından)
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

  const canSubmit = videoFile !== null && scene.trim() !== "" && !submitting;

  const handleSubmit = async () => {
    if (!videoFile || !scene.trim()) return;
    setSubmitting(true);
    setSubmitError(null);
    try {
      const res = await submitJob(videoFile, {
        scene: scene.trim(),
        smoke_test: preset === "smoke",
        cloud: preset === "cloud",
        skip_foundation: skipFoundation,
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
      <h2 className="submit-title">Yeni Job</h2>

      {/* Video dosyası */}
      <div className="submit-section">
        <label className="submit-label">Video dosyası</label>
        <div className="file-picker">
          <input
            type="file"
            accept="video/mp4,video/quicktime,video/*"
            onChange={(e) => handleFile(e.target.files?.[0] ?? null)}
            id="video-input"
          />
          <label htmlFor="video-input" className="file-picker-btn">
            {videoFile ? "Değiştir" : "Dosya seç"}
          </label>
          {videoFile && (
            <div className="file-info">
              <div className="file-name">{videoFile.name}</div>
              <div className="file-size">{videoSize}</div>
            </div>
          )}
        </div>
        <p className="submit-hint">
          MP4 veya MOV — kamera hareketli, sahnede belirgin doku
          (bkz. HyperNeRF: video'yu önce scripts/hypernerf_to_mp4.py ile çevir)
        </p>
      </div>

      {/* Sahne adı */}
      <div className="submit-section">
        <label className="submit-label">Sahne adı</label>
        <input
          type="text"
          className="submit-input"
          value={scene}
          onChange={(e) => setScene(e.target.value)}
          placeholder="örn. my_scene"
        />
        <p className="submit-hint">
          <code>data/&lt;scene&gt;/</code> altında çalışır. Aynı isim varsa
          üzerine yazar.
        </p>
      </div>

      {/* Preset */}
      <div className="submit-section">
        <label className="submit-label">Preset</label>
        <div className="preset-row">
          <label className={`preset-chip ${preset === "smoke" ? "active" : ""}`}>
            <input
              type="radio"
              checked={preset === "smoke"}
              onChange={() => setPreset("smoke")}
            />
            <div>
              <div className="preset-name">Smoke</div>
              <div className="preset-desc">500 iter, 480x270, 10 ts · ~1-2 dk</div>
            </div>
          </label>
          <label className={`preset-chip ${preset === "full" ? "active" : ""}`}>
            <input
              type="radio"
              checked={preset === "full"}
              onChange={() => setPreset("full")}
            />
            <div>
              <div className="preset-name">Full (3060 Ti)</div>
              <div className="preset-desc">30k iter, 640x360, 60 ts · ~30-60 dk</div>
            </div>
          </label>
          <label className={`preset-chip ${preset === "cloud" ? "active" : ""}`}>
            <input
              type="radio"
              checked={preset === "cloud"}
              onChange={() => setPreset("cloud")}
            />
            <div>
              <div className="preset-name">Cloud (4090)</div>
              <div className="preset-desc">60k iter, 1920x1080, 120 ts</div>
            </div>
          </label>
        </div>
      </div>

      {/* Foundation models toggle */}
      <div className="submit-section">
        <label className="submit-checkbox">
          <input
            type="checkbox"
            checked={skipFoundation}
            onChange={(e) => setSkipFoundation(e.target.checked)}
          />
          Foundation modelleri atla (Metric3D / CoTracker / SAM2)
          <span className="submit-hint-inline">
            — şu an stub, True bırakman önerilir
          </span>
        </label>
      </div>

      {/* Advanced hyperparams */}
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
          {submitting ? "Gönderiliyor..." : "Job'ı başlat"}
        </button>
        {submitError && <div className="submit-error">Hata: {submitError}</div>}
      </div>
    </div>
  );
}
