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

type Preset = "micro" | "smoke" | "full" | "high" | "cloud" | "ultra" | "ultra_clean" | "static_max";

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
        micro_test: preset === "micro",
        smoke_test: preset === "smoke",
        cloud: preset === "cloud",
        high_test: preset === "high",
        ultra_test: preset === "ultra",
        ultra_clean: preset === "ultra_clean",
        static_max: preset === "static_max",
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
        <div style={{ marginBottom: 8, padding: 8, background: "var(--bg-1)", border: "1px solid var(--border-1)", borderRadius: 4, fontSize: 11, color: "var(--text-muted)" }}>
          <strong style={{ color: "var(--text-primary)" }}>Pipeline Mode:</strong>{" "}
          <span>Auto-detect — eğer <code>data/&lt;sahne&gt;/videos/cam*.mp4</code> varsa <strong>multi-view</strong> aktif (N3V/Neural 3D Video format). Aksi halde <strong>single-view</strong>. Multi-view scene için video upload zorunlu değil ama sahne adı önceden hazır olmalı (scripts/load_n3v.py ile import).</span>
        </div>
        <div className="preset-row">
          <label className={`preset-chip ${preset === "micro" ? "active" : ""}`}>
            <input
              type="radio"
              checked={preset === "micro"}
              onChange={() => setPreset("micro")}
            />
            <div>
              <div className="preset-name">Micro ⚡</div>
              <div className="preset-desc">200 iter, 320x180, 5 ts · foundation kapalı ~30 sn · açık ~5-10 dk</div>
            </div>
          </label>
          <label className={`preset-chip ${preset === "smoke" ? "active" : ""}`}>
            <input
              type="radio"
              checked={preset === "smoke"}
              onChange={() => setPreset("smoke")}
            />
            <div>
              <div className="preset-name">Smoke</div>
              <div className="preset-desc">500 iter, 480x270, 10 ts · ~5-15 dk</div>
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
          <label className={`preset-chip ${preset === "high" ? "active" : ""}`}>
            <input
              type="radio"
              checked={preset === "high"}
              onChange={() => setPreset("high")}
            />
            <div>
              <div className="preset-name">High ⭐ (3-4 saat)</div>
              <div className="preset-desc">50k iter, 640x360, 90 ts, Fourier K=10, N cap 60k · enhanced quality</div>
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
          <label className={`preset-chip ${preset === "ultra" ? "active" : ""}`}>
            <input
              type="radio"
              checked={preset === "ultra"}
              onChange={() => setPreset("ultra")}
            />
            <div>
              <div className="preset-name">Ultra 🔥 (6-9 saat)</div>
              <div className="preset-desc">80k iter, 720x405, 90 ts, HexPlane 112/56, MLP 640/4, Fourier K=12, N cap 80k · 3060 Ti optimize</div>
            </div>
          </label>
          <label className={`preset-chip ${preset === "ultra_clean" ? "active" : ""}`}>
            <input
              type="radio"
              checked={preset === "ultra_clean"}
              onChange={() => setPreset("ultra_clean")}
            />
            <div>
              <div className="preset-name">Ultra Clean ✨ (7-9 saat)</div>
              <div className="preset-desc">v3.8 anti-streak: aniso reg + sıkı dpos clamp + rigid 5× + fourier_reg 10× + density_end 30k + sh_degree 2 · banana streak fix</div>
            </div>
          </label>
          <label className={`preset-chip ${preset === "static_max" ? "active" : ""}`}>
            <input
              type="radio"
              checked={preset === "static_max"}
              onChange={() => setPreset("static_max")}
            />
            <div>
              <div className="preset-name">Static Max 🎯 (v3.9, ~3-4 saat preprocessing + 7-9 saat training)</div>
              <div className="preset-desc">Tüm preprocessing iyileştirmeleri: fps=20, vit_large depth, COLMAP exhaustive, confidence init subsample, sh_degree=3 · Ultra Clean fix'leri + max input quality. Banana halo testi.</div>
            </div>
          </label>
        </div>
        {preset === "micro" && (
          <p className="submit-hint">
            ⚡ Dev iteration / preflight.
            Foundation <strong>atla</strong> → ~30 sn (cache hit), sadece recon test.
            Foundation <strong>aç</strong> → ~5-10 dk, tam Stage 2 preflight (MiDaS+CoTracker+tracks/depth loss).
            CoTracker grid 30→15, MiDaS small — tümü hız optimize.
          </p>
        )}
        {preset === "high" && (
          <p className="submit-hint">
            ⭐ <strong>3-4 saat enhanced quality</strong> (Full ile Ultra arası).
            50k iter, 640×360 (Full ile aynı), Fourier K=10, density end @ 35k,
            <strong>N hard cap 60k</strong> (init 42k subsample), num_ts 90.
            Model parametreleri default — render hızı korunmuş.
            Banana/cookie gibi standart sahneler için ideal denge.
          </p>
        )}
        {preset === "ultra" && (
          <p className="submit-hint">
            🔥 <strong>6-9 saat max-quality render</strong> (3060 Ti için optimize).
            v2 (revised — v1 banana'da 164k gaussian patladı, 21 gün ETA).
            80k iter, 720×405, HexPlane 112/56, MLP 640/4, Fourier K=12,
            <strong>N hard cap 80k</strong> (densify threshold 5e-4 ile yavaş büyüme),
            density end @ 50k, num_ts 90, CoTracker grid 25.
            Video 20-40 sn ideal. <strong>Bilgisayarı uyutma</strong>.
          </p>
        )}
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
