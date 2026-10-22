/**
 * Yeni Splat — one settings screen for the Spirula pipeline with two exits:
 * "Local başlat" runs it on this PC, "Notebook üret" writes the same run as a Colab notebook.
 * The cost panel re-estimates VRAM and training time on every change.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { generatePipelineNotebook } from "../api";
import { NotebookDownload } from "../notebook/NotebookDownload";
import type { GeneratedNotebook } from "../notebook/types";
import { CostPanel } from "./CostPanel";
import { driveRelative, formatCount, formatDuration, formatGb, notebookSpec, parseCount } from "./format";
import {
  estimateRun, getSpirulaInfo, probeInput, startLocalRun,
  type EstimateResponse, type GpuName, type InputMode, type Preset, type ProbeResult, type SpirulaInfo,
  type SpirulaSettings,
} from "./spirulaApi";
import "./newsplat.css";

const DRIVE_ROOT_KEY = "4dgs.driveRoot";
const DEFAULT_DRIVE_ROOT = "G:\\Drive'ım";
const ESTIMATE_REFRESH_MS = 5000;
const CAP_CHIPS = [1e6, 3e6, 6e6, 10e6, 50e6];

const DEFAULT_SETTINGS: SpirulaSettings = {
  preset: "quality", iterations: null, cap_max: null, fps: 4, sfm_quality: null, geometry_model: "moge2-vitb",
  generate_depth: true, appearance: true, floater_suppression: "off", resolution_divisor: 1, allow_partial: false,
};

const MODES: { id: InputMode; label: string; hint: string; placeholder: string }[] = [
  { id: "video", label: "Video", hint: "Kareleri en net olanlardan seçer, sonra SfM yapar.", placeholder: "G:\\Drive'ım\\GaussianTests\\inputs\\oda.MOV" },
  { id: "images", label: "Fotoğraflar", hint: "Fotoğraf klasörü (ya da images/ alt klasörü olan klasör).", placeholder: "E:\\cekimler\\oda_fotolar" },
  { id: "dataset", label: "Hazır dataset", hint: "images/ + sparse/ (COLMAP) varsa SfM atlanır.", placeholder: "E:\\datasets\\oda" },
];

function readDriveRoot(): string {
  try {
    return window.localStorage.getItem(DRIVE_ROOT_KEY) || DEFAULT_DRIVE_ROOT;
  } catch {
    return DEFAULT_DRIVE_ROOT;
  }
}

interface Props {
  active?: boolean;
  onLocalStarted?(jobId: string, runId: string): void;
}

export function NewSplatPage({ active = true, onLocalStarted }: Props) {
  const [info, setInfo] = useState<SpirulaInfo | null>(null);
  const [infoError, setInfoError] = useState("");
  const [mode, setMode] = useState<InputMode>("video");
  const [path, setPath] = useState("");
  const [name, setName] = useState("");
  const [settings, setSettings] = useState<SpirulaSettings>(DEFAULT_SETTINGS);
  const [capText, setCapText] = useState("");
  const [probe, setProbe] = useState<ProbeResult | null>(null);
  const [probeError, setProbeError] = useState("");
  const [estimate, setEstimate] = useState<EstimateResponse | null>(null);
  const [target, setTarget] = useState<GpuName>("local");
  const [tick, setTick] = useState(0);
  const [driveRoot, setDriveRoot] = useState(readDriveRoot);
  const [driveOverride, setDriveOverride] = useState("");
  const [busy, setBusy] = useState<"" | "local" | "notebook">("");
  const [error, setError] = useState("");
  const [artifact, setArtifact] = useState<GeneratedNotebook | null>(null);
  const alive = useRef(true);
  useEffect(() => () => { alive.current = false; }, []);

  // The backend may come up after the app; keep asking until it answers.
  useEffect(() => {
    if (info) return;
    let cancelled = false;
    let timer = 0;
    const load = () => getSpirulaInfo().then(
      (value) => { if (!cancelled) { setInfo(value); setInfoError(""); } },
      (e) => { if (!cancelled) { setInfoError(String(e)); timer = window.setTimeout(load, ESTIMATE_REFRESH_MS); } },
    );
    void load();
    return () => { cancelled = true; window.clearTimeout(timer); };
  }, [info]);

  // Probe the capture so the estimate uses its real size and frame count.
  useEffect(() => {
    setProbe(null);
    setProbeError("");
    if (!path.trim()) return;
    const ctrl = new AbortController();
    const id = window.setTimeout(() => {
      probeInput(mode, path, settings.fps, { signal: ctrl.signal }).then(setProbe, (e) => {
        if (!ctrl.signal.aborted) setProbeError(String(e).replace(/^Error: HTTP \d+ [^:]*: /, "").replace(/^\{"detail":"(.*)"\}$/, "$1"));
      });
    }, 500);
    return () => { ctrl.abort(); window.clearTimeout(id); };
  }, [mode, path, settings.fps]);

  // Live free VRAM changes as other apps run; refresh the estimate while the page is visible.
  useEffect(() => {
    if (!active) return;
    const id = window.setInterval(() => setTick((t) => t + 1), ESTIMATE_REFRESH_MS);
    return () => window.clearInterval(id);
  }, [active]);

  useEffect(() => {
    if (!active) return;
    const ctrl = new AbortController();
    const id = window.setTimeout(() => {
      estimateRun(settings, probe?.width, probe?.height, { signal: ctrl.signal }).then(setEstimate, () => {});
    }, 150);
    return () => { ctrl.abort(); window.clearTimeout(id); };
  }, [active, settings, probe?.width, probe?.height, tick]);

  const preset = info?.presets[settings.preset];
  const resolved = estimate?.resolved;
  const local = estimate?.estimates.find((e) => e.gpu === "local");
  const drivePath = useMemo(() => driveOverride.trim() || driveRelative(path, driveRoot) || "", [driveOverride, path, driveRoot]);
  const capError = capText.trim() && parseCount(capText) === null ? "Örn. 3M, 500K veya 2500000" : "";

  function update(patch: Partial<SpirulaSettings>) {
    setSettings((s) => ({ ...s, ...patch }));
    setArtifact(null);
  }

  function setCap(text: string) {
    setCapText(text);
    const n = parseCount(text);
    if (!text.trim()) update({ cap_max: null });
    else if (n !== null) update({ cap_max: Math.min(Math.max(n, 50_000), 100_000_000) });
  }

  function choosePreset(p: Preset) {
    setCapText("");
    update({ preset: p, iterations: null, cap_max: null, sfm_quality: null });
  }

  function saveDriveRoot(value: string) {
    setDriveRoot(value);
    try { window.localStorage.setItem(DRIVE_ROOT_KEY, value); } catch { /* private mode */ }
  }

  async function runLocal() {
    if (local?.fits === false && !window.confirm(
      `Tahmini VRAM (${formatGb(local.vram_mib)}) bu PC'deki boş belleği aşıyor. Yine de başlatılsın mı?`)) return;
    setBusy("local"); setError("");
    try {
      const res = await startLocalRun({ input_mode: mode, input_path: path, settings, name: name.trim() || undefined });
      onLocalStarted?.(res.job_id, res.run_id);
    } catch (e) {
      if (alive.current) setError(String(e));
    } finally {
      if (alive.current) setBusy("");
    }
  }

  async function makeNotebook() {
    setBusy("notebook"); setError(""); setArtifact(null);
    try {
      const nb = await generatePipelineNotebook(notebookSpec(mode, drivePath, settings));
      if (alive.current) setArtifact(nb);
    } catch (e) {
      if (alive.current) setError(String(e));
    } finally {
      if (alive.current) setBusy("");
    }
  }

  const modeInfo = MODES.find((m) => m.id === mode)!;
  const canLocal = !!info?.available && !!path.trim() && !probeError && !busy;
  const notebookBlocked = mode !== "video" && probe && probe.kind !== "video" && !probe.images_subdir && !drivePath.toLowerCase().endsWith(".zip")
    ? "Notebook için klasörde images/ alt klasörü olmalı." : "";
  const canNotebook = !!drivePath && !notebookBlocked && !busy;

  return (
    <div className="ns-page">
      <div className="ns-layout">
        <form className="ns-form" onSubmit={(e) => e.preventDefault()}>
          <header className="ns-head">
            <h2>Yeni splat</h2>
            <p>Bir kez ayarla; bu PC'de çalıştır ya da aynı ayarlarla Colab notebook'u üret.</p>
          </header>

          <section className="ns-section" aria-labelledby="ns-input">
            <h3 id="ns-input">1 · Girdi</h3>
            <div className="ns-segment" role="radiogroup" aria-label="Girdi türü">
              {MODES.map((m) => (
                <button key={m.id} type="button" role="radio" aria-checked={mode === m.id}
                  className={mode === m.id ? "active" : ""} onClick={() => { setMode(m.id); setArtifact(null); }}>
                  {m.label}
                </button>
              ))}
            </div>
            <label className="ns-field">
              <span className="ns-label">{mode === "video" ? "Video dosyası" : "Klasör"}</span>
              <input value={path} onChange={(e) => { setPath(e.target.value); setArtifact(null); }}
                placeholder={modeInfo.placeholder} spellCheck={false} />
              <span className="ns-hint">{modeInfo.hint} Yolu Gezgin'den kopyalayıp yapıştırabilirsin.</span>
            </label>
            {probeError && <p className="ns-msg ns-err" role="alert">{probeError}</p>}
            {probe && <ProbeLine probe={probe} fps={settings.fps} />}
            <label className="ns-field ns-field-narrow">
              <span className="ns-label">Koşu adı <em>isteğe bağlı</em></span>
              <input value={name} onChange={(e) => setName(e.target.value)} placeholder="oda2" />
            </label>
          </section>

          <section className="ns-section" aria-labelledby="ns-quality">
            <h3 id="ns-quality">2 · Kalite</h3>
            <div className="ns-presets">
              {info && (Object.entries(info.presets) as [Preset, SpirulaInfo["presets"][Preset]][]).map(([id, p]) => (
                <button key={id} type="button" className={`ns-preset ${settings.preset === id ? "active" : ""}`}
                  aria-pressed={settings.preset === id} onClick={() => choosePreset(id)}>
                  <strong>{p.label}</strong>
                  <span>{formatCount(p.cap_max)} splat · {formatCount(p.iterations)} adım</span>
                  <span>Spirula {p.quality} · SfM {p.sfm_quality}</span>
                </button>
              ))}
            </div>

            <div className="ns-grid">
              <label className="ns-field">
                <span className="ns-label">Splat sınırı <CostTag text={local ? `≈ ${formatGb(local.vram_splats_mib)} VRAM` : ""} /></span>
                <input value={capText} onChange={(e) => setCap(e.target.value)} inputMode="numeric"
                  placeholder={preset ? formatCount(preset.cap_max) : ""} aria-invalid={!!capError} />
                <span className="ns-chips">
                  {CAP_CHIPS.map((c) => (
                    <button key={c} type="button" className={resolved?.cap_max === c ? "active" : ""}
                      onClick={() => setCap(formatCount(c))}>{formatCount(c)}</button>
                  ))}
                </span>
                <span className="ns-hint">{capError || "Eğitimin büyüyebileceği en fazla Gaussian. VRAM'i en çok bu belirler."}</span>
              </label>
              <label className="ns-field">
                <span className="ns-label">Eğitim adımı <CostTag text={local ? `eğitim ${formatDuration(local.train_seconds)}` : ""} /></span>
                <input type="number" min={1000} max={200000} step={1000} value={settings.iterations ?? ""}
                  placeholder={preset ? String(preset.iterations) : ""}
                  onChange={(e) => update({ iterations: e.target.value ? Number(e.target.value) : null })} />
                <span className="ns-hint">Süreyi doğrusal artırır; VRAM'i değiştirmez.</span>
              </label>
              <label className="ns-field">
                <span className="ns-label">Eğitim çözünürlüğü <CostTag text={local ? `${local.train_megapixels} MP · ${formatGb(local.vram_base_mib)}` : ""} /></span>
                <select value={settings.resolution_divisor} onChange={(e) => update({ resolution_divisor: Number(e.target.value) as 1 | 2 | 4 })}>
                  <option value={1}>Tam (önerilen)</option>
                  <option value={2}>1/2: daha az VRAM, daha hızlı</option>
                  <option value={4}>1/4: hızlı deneme</option>
                </select>
                <span className="ns-hint">Küçültmek detayı da düşürür.</span>
              </label>
              {mode === "video" && (
                <label className="ns-field">
                  <span className="ns-label">Kare hızı</span>
                  <input type="number" min={1} max={30} step={1} value={settings.fps}
                    onChange={(e) => update({ fps: Math.min(30, Math.max(1, Number(e.target.value) || 1)) })} />
                  <span className="ns-hint">Saniyede seçilen kare. Fazlası SfM'yi uzatır; VRAM'e etkisi yok.</span>
                </label>
              )}
            </div>

            <details className="ns-advanced">
              <summary>Gelişmiş</summary>
              <div className="ns-grid">
                <label className="ns-field">
                  <span className="ns-label">SfM kalitesi</span>
                  <select value={settings.sfm_quality ?? ""} onChange={(e) => update({ sfm_quality: (e.target.value || null) as SpirulaSettings["sfm_quality"] })}>
                    <option value="">Preset ({preset?.sfm_quality ?? "…"})</option>
                    <option value="high">high</option>
                    <option value="extreme">extreme: daha çok kamera kaydı, daha yavaş</option>
                  </select>
                </label>
                <label className="ns-field">
                  <span className="ns-label">Geometri modeli</span>
                  <select value={settings.geometry_model} onChange={(e) => update({ geometry_model: e.target.value as SpirulaSettings["geometry_model"] })}>
                    <option value="moge2-vits">MoGe-2 small</option>
                    <option value="moge2-vitb">MoGe-2 base (önerilen)</option>
                    <option value="moge2-vitl">MoGe-2 large</option>
                  </select>
                </label>
                <label className="ns-field">
                  <span className="ns-label">Uçuşan parça temizliği</span>
                  <select value={settings.floater_suppression} onChange={(e) => update({ floater_suppression: e.target.value as SpirulaSettings["floater_suppression"] })}>
                    <option value="off">Kapalı (önerilen)</option>
                    <option value="mild">Hafif</option>
                    <option value="strong">Güçlü</option>
                  </select>
                  <span className="ns-hint">ROOM2'de kapalıyken daha keskin çıktı.</span>
                </label>
                <div className="ns-checks">
                  <Check checked={settings.appearance} onChange={(v) => update({ appearance: v })}
                    label="Fotoğraf başına renk düzeltmesi" hint="Pozlama/beyaz dengesi farklarını emer. Kapatırsan renk farkları splata işlenir." />
                  <Check checked={settings.generate_depth} onChange={(v) => update({ generate_depth: v })}
                    label="Derinlik rehberi" hint="MoGe-2 derinlik haritalarını eğitime katar." />
                  <Check checked={settings.allow_partial} onChange={(v) => update({ allow_partial: v })}
                    label="Kısmi SfM modeline izin ver" hint="Karelerin %80'i kaydedilemezse durmak yerine en büyük parçayla devam eder." />
                </div>
              </div>
            </details>
          </section>

          <section className="ns-section" aria-labelledby="ns-run">
            <h3 id="ns-run">3 · Çalıştır</h3>
            <div className="ns-actions">
              <div className="ns-action">
                <button type="button" className="ns-btn ns-btn-primary" disabled={!canLocal} onClick={() => void runLocal()}>
                  {busy === "local" ? "Başlatılıyor…" : "Local başlat"}
                </button>
                <span className="ns-hint">
                  {!info ? (infoError ? "Backend'e ulaşılamadı." : "Spirula aranıyor…")
                    : info.available ? `Bu PC · ${info.version ?? "spirula"} · canlı önizleme ile`
                    : info.error}
                </span>
              </div>
              <div className="ns-action">
                <button type="button" className="ns-btn" disabled={!canNotebook} onClick={() => void makeNotebook()}>
                  {busy === "notebook" ? "Üretiliyor…" : "Notebook üret"}
                </button>
                <span className="ns-hint">
                  {notebookBlocked || (drivePath ? `Colab'de MyDrive/${drivePath}` : "Girdi Drive klasöründe değil: Drive yolunu aşağıya yaz.")}
                </span>
              </div>
            </div>
            <details className="ns-advanced">
              <summary>Drive ayarı</summary>
              <div className="ns-grid">
                <label className="ns-field">
                  <span className="ns-label">Drive klasörü (bu PC)</span>
                  <input value={driveRoot} onChange={(e) => saveDriveRoot(e.target.value)} spellCheck={false} />
                  <span className="ns-hint">Bunun altındaki yollar notebook için otomatik çevrilir.</span>
                </label>
                <label className="ns-field">
                  <span className="ns-label">MyDrive yolu (elle)</span>
                  <input value={driveOverride} onChange={(e) => setDriveOverride(e.target.value)}
                    placeholder="GaussianTests/inputs/oda.MOV" spellCheck={false} />
                </label>
              </div>
            </details>
            {error && <p className="ns-msg ns-err" role="alert">{error}</p>}
            {artifact && <NotebookDownload artifact={artifact} onRegenerate={() => void makeNotebook()} onEdit={() => setArtifact(null)} />}
          </section>
        </form>

        <CostPanel estimate={estimate} target={target} onTarget={setTarget} frames={probe?.frames ?? null} />
      </div>
    </div>
  );
}

function CostTag({ text }: { text: string }) {
  return text ? <span className="ns-cost-tag">{text}</span> : null;
}

function Check({ checked, onChange, label, hint }: { checked: boolean; onChange(v: boolean): void; label: string; hint: string }) {
  return (
    <label className="ns-check">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span><strong>{label}</strong><span className="ns-hint">{hint}</span></span>
    </label>
  );
}

function ProbeLine({ probe, fps }: { probe: ProbeResult; fps: number }) {
  const size = probe.width && probe.height ? `${probe.width}×${probe.height}` : null;
  const parts = probe.kind === "video"
    ? [probe.duration ? `${Math.floor(probe.duration / 60)}:${String(Math.round(probe.duration % 60)).padStart(2, "0")}` : null,
       size, probe.frames ? `~${probe.frames} kare (${fps} fps)` : null, probe.warning]
    : [`${probe.frames} görüntü`, size, probe.has_sfm ? "SfM modeli var, SfM atlanır" : null];
  return <p className="ns-msg ns-ok">{parts.filter(Boolean).join(" · ")}</p>;
}
