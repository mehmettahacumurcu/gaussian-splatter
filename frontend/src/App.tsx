/**
 * 4DGS Studio — ana uygulama (tab navigation).
 *
 * Ana sekmeler:
 *   - Yeni Splat: Spirula ayarları + canlı VRAM/süre tahmini → local başlat ya da notebook üret
 *   - İşler: tüm job'lar, canlı eğitim önizlemesi, log, sonuç eylemleri
 *   - Viewer: seçilen job'ın splat render'ı
 *   - Kapsama: splatın çekimde az görülen yerleri (ısı haritası, yan yana)
 *   - Sahne Editörü: splat/mesh objeleri bir base sahneye yerleştir + export
 *   - Interactive: gezilebilir splat dünyası (D+E)
 * "Diğer" menüsü: diğer notebook tarifleri, 4D local eğitim, Analiz, Eval.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import "./App.css";
import { SplatViewer } from "./components/SplatViewer";
import { JobSubmitPanel } from "./components/JobSubmitPanel";
import { TrainingAnalytics } from "./components/TrainingAnalytics";
import { NvsEvalPanel } from "./components/NvsEvalPanel";
import { ConnectionSettings } from "./components/ConnectionSettings";
import { InteractivePage } from "./interactive/InteractivePage";
import { ComposePage } from "./compose/ComposePage";
import { CoveragePage, type CoveragePrefill } from "./coverage/CoveragePage";
import { isComposeJob } from "./compose/isComposeJob";
import { NotebookGeneratorPanel } from "./notebook/NotebookGeneratorPanel";
import { NewSplatPage } from "./newsplat/NewSplatPage";
import { JobsPage } from "./jobs/JobsPage";
import {
  getHealth,
  getSplatInfo,
  listJobs,
  listDiskScenes,
  type HealthResponse,
  type Job,
  type SceneListItem,
  type SplatInfo,
} from "./api";
import { getConnection, isLocal, onConnectionChange } from "./connection";

type Tab = "new" | "jobs" | "viewer" | "coverage" | "compose" | "interactive"
  | "notebook" | "submit" | "analytics" | "eval";

const MAIN_TABS: { id: Tab; label: string }[] = [
  { id: "new", label: "Yeni Splat" },
  { id: "jobs", label: "İşler" },
  { id: "viewer", label: "Viewer" },
  { id: "coverage", label: "Kapsama" },
  { id: "compose", label: "Sahne Editörü" },
  { id: "interactive", label: "Interactive" },
];
const MORE_TABS: { id: Tab; label: string; hint: string }[] = [
  { id: "notebook", label: "Diğer notebook'lar", hint: "Native, hybrid, metin → splat, ön işleme" },
  { id: "submit", label: "4D local eğitim", hint: "Eski Trainer4DGS job formu" },
  { id: "analytics", label: "Analiz", hint: "4D eğitim metrikleri" },
  { id: "eval", label: "Eval", hint: "Held-out NVS değerlendirmesi" },
];

type ViewerState =
  | { kind: "idle" }
  | { kind: "loading-info" }
  | { kind: "ready"; info: SplatInfo }
  | { kind: "error"; message: string };

interface SceneProgress {
  loaded: number;
  total: number;
  done: boolean;
}

function formatBytes(b: number): string {
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / 1024 / 1024).toFixed(1)} MB`;
}

function formatTime(ts: number): string {
  try {
    return new Date(ts * 1000).toLocaleString();
  } catch {
    return "-";
  }
}

function App() {
  const [tab, setTab] = useState<Tab>("new");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [coveragePrefill, setCoveragePrefill] = useState<CoveragePrefill | null>(null);
  const [moreOpen, setMoreOpen] = useState(false);
  const moreRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!moreOpen) return;
    const close = (e: MouseEvent) => {
      if (!moreRef.current?.contains(e.target as Node)) setMoreOpen(false);
    };
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setMoreOpen(false); };
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", close);
      document.removeEventListener("keydown", onKey);
    };
  }, [moreOpen]);

  // Viewer state
  const [viewerState, setViewerState] = useState<ViewerState>({ kind: "idle" });
  const [sceneProgress, setSceneProgress] = useState<SceneProgress | null>(null);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [jobIdInput, setJobIdInput] = useState("");
  const [diskScenes, setDiskScenes] = useState<SceneListItem[] | null>(null);
  const [diskPanelOpen, setDiskPanelOpen] = useState(false);
  const [diskLoading, setDiskLoading] = useState(false);
  // v3.7.7 Debug: tek frame yükleme modu — visibility toggle bypass.
  // Setter not currently exposed in UI (no toggle); retained as state so a
  // future settings popover can flip it without a refactor.
  const [singleFrameMode] = useState(false);
  // Analytics — son/aktif job için
  const [analyticsScene, setAnalyticsScene] = useState<string>("");
  // Connection — backend URL + auth token (apiBase shown in topbar; settings modal toggles)
  const [connSettingsOpen, setConnSettingsOpen] = useState(false);
  const [connection, setConnectionState] = useState(() => getConnection());
  useEffect(() => onConnectionChange(setConnectionState), []);
  // Sahne Editörü: ilk açılışta mount edilir, sonra sekme değişse de state kaybolmasın diye açık kalır.
  // Render sırasında set edilir (effect değil) → ilk açılışta boş bir frame olmaz.
  const [composeMounted, setComposeMounted] = useState(false);
  if (tab === "compose" && !composeMounted) setComposeMounted(true);
  // Kapsama: tarama sürerken sekme değişse de ilerleme kaybolmasın; görüntüleyiciler gizliyken kapanır.
  const [coverageMounted, setCoverageMounted] = useState(false);
  if (tab === "coverage" && !coverageMounted) setCoverageMounted(true);

  // Health poll (her 5 sn)
  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      try {
        const h = await getHealth();
        if (!cancelled) setHealth(h);
      } catch (e) {
        if (!cancelled) setHealth(null);
      }
    };
    tick();
    const id = window.setInterval(tick, 5000);
    return () => { cancelled = true; window.clearInterval(id); };
  }, []);

  const loadInViewer = useCallback(async (identifier: string) => {
    const id = identifier.trim();
    if (!id) return;
    setViewerState({ kind: "loading-info" });
    setSceneProgress(null);
    setCurrentFrame(0);
    setDiskPanelOpen(false);
    setTab("viewer");
    try {
      const info = await getSplatInfo(id);
      if (info.num_frames === 0) {
        setViewerState({ kind: "error", message: "Bu sahnenin .ply çıktısı yok" });
        return;
      }
      setSceneProgress({ loaded: 0, total: info.num_frames, done: false });
      setViewerState({ kind: "ready", info });
    } catch (e) {
      setViewerState({ kind: "error", message: String(e) });
    }
  }, []);

  const handleJobSubmitted = useCallback((resp: any, sceneName: string) => {
    // Job başarıyla gönderildi, İşler tab'ına geç — kullanıcı ilerlemeyi takip etsin
    // Analytics'e de aynı sahneyi otomatik bağla
    setAnalyticsScene(sceneName);
    if (resp?.job_id) setSelectedJobId(resp.job_id);
    setTab("jobs");
  }, []);

  const handleLocalStarted = useCallback((jobId: string) => {
    setSelectedJobId(jobId);
    setTab("jobs");
  }, []);

  const openCoverage = useCallback((prefill: CoveragePrefill) => {
    setCoveragePrefill({ ...prefill });
    setTab("coverage");
  }, []);

  const handleViewJob = useCallback((job: Job) => {
    // Job completed mı kontrol et, viewer'a yükle
    if (job.status === "completed") {
      setJobIdInput(job.id);
      loadInViewer(job.id);
    }
  }, [loadInViewer]);

  const loadLatestCompleted = useCallback(async () => {
    try {
      const { jobs } = await listJobs();
      // Compose export job'larının viewer'da açılacak bir splat çıktısı yok.
      const done = jobs.find((j) => j.status === "completed" && !isComposeJob(j));
      if (done) {
        setJobIdInput(done.id);
        loadInViewer(done.id);
      } else {
        setViewerState({ kind: "error", message: "Tamamlanmış job yok" });
      }
    } catch (e) {
      setViewerState({ kind: "error", message: String(e) });
    }
  }, [loadInViewer]);

  const toggleDiskPanel = useCallback(async () => {
    if (diskPanelOpen) {
      setDiskPanelOpen(false);
      return;
    }
    setDiskPanelOpen(true);
    setDiskLoading(true);
    try {
      const res = await listDiskScenes();
      setDiskScenes(res.scenes);
    } catch (e) {
      setViewerState({ kind: "error", message: String(e) });
      setDiskScenes([]);
    } finally {
      setDiskLoading(false);
    }
  }, [diskPanelOpen]);

  const backendBadge = (() => {
    const localTag = isLocal()
      ? <span className="badge-tag badge-tag-local">LOCAL</span>
      : <span className="badge-tag badge-tag-cloud">CLOUD</span>;
    if (!health) {
      return (
        <span className="badge badge-unknown">
          {localTag} Backend: bağlantı yok
        </span>
      );
    }
    const gpu = health.gpu_available ? `GPU: ${health.gpu_name ?? "?"}` : "GPU: yok";
    return (
      <span className="badge badge-ok">
        {localTag} Backend OK · {gpu} · Aktif: {health.active_jobs}
      </span>
    );
  })();

  return (
    <div className="app">
      {/* Üst bar */}
      <header className="app-topbar">
        <div className="app-brand">
          <h1>4DGS Studio</h1>
        </div>
        <nav className="app-tabs">
          {MAIN_TABS.map((t) => (
            <button
              key={t.id}
              className={`tab-btn ${tab === t.id ? "active" : ""}`}
              aria-current={tab === t.id ? "page" : undefined}
              onClick={() => setTab(t.id)}
            >
              {t.label}
              {t.id === "jobs" && health && health.active_jobs > 0 && (
                <span className="tab-badge">{health.active_jobs}</span>
              )}
            </button>
          ))}
          <div className="tab-more" ref={moreRef}>
            <button
              className={`tab-btn ${MORE_TABS.some((t) => t.id === tab) ? "active" : ""}`}
              aria-haspopup="menu"
              aria-expanded={moreOpen}
              onClick={() => setMoreOpen((v) => !v)}
            >
              {MORE_TABS.find((t) => t.id === tab)?.label ?? "Diğer"} ▾
            </button>
            {moreOpen && (
              <div className="tab-more-menu" role="menu">
                {MORE_TABS.map((t) => (
                  <button
                    key={t.id}
                    role="menuitem"
                    className={tab === t.id ? "active" : ""}
                    onClick={() => { setTab(t.id); setMoreOpen(false); }}
                  >
                    <span>{t.label}</span>
                    <small>{t.hint}</small>
                  </button>
                ))}
              </div>
            )}
          </div>
        </nav>
        <div className="app-status">
          {backendBadge}
          <button
            type="button"
            className="conn-settings-btn"
            onClick={() => setConnSettingsOpen(true)}
            title="Backend connection settings"
            aria-label="Backend connection settings"
          >
            ⚙
          </button>
        </div>
      </header>

      {/* İçerik */}
      <main className="app-content">
        {/* Yeni Splat stays mounted so the form survives a trip to İşler and back. */}
        <div className="compose-host" style={{ display: tab === "new" ? "block" : "none" }}>
          <NewSplatPage active={tab === "new"} onLocalStarted={handleLocalStarted} />
        </div>

        {tab === "notebook" && <NotebookGeneratorPanel />}

        {tab === "submit" && (
          <JobSubmitPanel onJobSubmitted={handleJobSubmitted} />
        )}

        {tab === "jobs" && (
          <JobsPage
            selectedId={selectedJobId}
            onSelect={setSelectedJobId}
            onViewJob={handleViewJob}
            onCoverage={openCoverage}
            onNewSplat={() => setTab("new")}
          />
        )}

        {tab === "analytics" && (
          <div className="tab-content">
            <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "14px 20px 0" }}>
              <input
                type="text"
                value={analyticsScene}
                placeholder="sahne adı (örn. cutlemon_v3_1_micro) veya job id"
                onChange={(e) => setAnalyticsScene(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") setAnalyticsScene((v) => v.trim());
                }}
                style={{ flex: 1, maxWidth: 500 }}
              />
              <button
                className="btn-secondary"
                onClick={async () => {
                  try {
                    const { jobs } = await listJobs();
                    const active = jobs.find(
                      (j) => j.status === "running" || j.status === "queued",
                    );
                    if (active) setAnalyticsScene(active.scene);
                    else {
                      const last = jobs[0];
                      if (last) setAnalyticsScene(last.scene);
                    }
                  } catch {
                    /* ignore */
                  }
                }}
              >
                Aktif/son job
              </button>
            </div>
            {analyticsScene ? (
              <TrainingAnalytics scene={analyticsScene} autoRefresh={true} />
            ) : (
              <div style={{ padding: 20, color: "#888" }}>
                Sahne adını gir ve Enter — ya da "Aktif/son job" butonuna bas.
              </div>
            )}
          </div>
        )}

        {tab === "eval" && (
          <div className="tab-content">
            <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "14px 20px 0" }}>
              <input
                type="text"
                value={analyticsScene}
                placeholder="sahne adı veya job id"
                onChange={(e) => setAnalyticsScene(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") setAnalyticsScene((v) => v.trim());
                }}
                style={{ flex: 1, maxWidth: 500 }}
              />
              <button
                className="btn-secondary"
                onClick={async () => {
                  try {
                    const { jobs } = await listJobs();
                    const done = jobs.find((j) => j.status === "completed" && !isComposeJob(j));
                    if (done) setAnalyticsScene(done.scene);
                  } catch { /* ignore */ }
                }}
              >
                Son tamamlanmis
              </button>
            </div>
            <NvsEvalPanel scene={analyticsScene} />
          </div>
        )}

        {tab === "viewer" && (
          <div className="viewer-tab">
            <div className="viewer-controls">
              <input
                type="text"
                value={jobIdInput}
                placeholder="job_id veya sahne adı"
                onChange={(e) => setJobIdInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") loadInViewer(jobIdInput);
                }}
              />
              <button className="btn-primary" onClick={() => loadInViewer(jobIdInput)}>
                Yükle
              </button>
              <button className="btn-secondary" onClick={loadLatestCompleted}>
                Son tamamlanmış
              </button>
              <button className="btn-secondary" onClick={toggleDiskPanel}>
                Diskten {diskPanelOpen ? "▲" : "▼"}
              </button>
              <label
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: 6,
                  marginLeft: 12,
                  fontSize: 12,
                  color: "#888",
                  cursor: "pointer",
                }}
                title="v3.7.7'den itibaren default: cached blob swap. Bu kapatılamaz (multi-mode broken)."
              >
                <input
                  type="checkbox"
                  checked={true}
                  disabled
                  readOnly
                />
                ✅ Cached single-frame swap (auto)
              </label>
            </div>

            {diskPanelOpen && (
              <div className="disk-panel">
                {diskLoading && <p className="hint">Disk taranıyor…</p>}
                {!diskLoading && diskScenes && diskScenes.length === 0 && (
                  <p className="hint hint-err">Diskte .ply çıktısı yok.</p>
                )}
                {!diskLoading && diskScenes && diskScenes.length > 0 && (
                  <ul className="scene-list">
                    {diskScenes.map((s) => (
                      <li
                        key={s.name}
                        className="scene-item"
                        onClick={() => {
                          setJobIdInput(s.name);
                          loadInViewer(s.name);
                        }}
                      >
                        <span className="scene-name">{s.name}</span>
                        <span className="scene-meta">
                          {s.num_frames} frame · {formatBytes(s.total_size_bytes)} · {formatTime(s.modified_ts)}
                        </span>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            )}

            {/* Viewer state hint */}
            {viewerState.kind === "idle" && (
              <div className="viewer-empty">
                <p className="hint">Bir job_id yapıştır veya "Diskten" butonundan seç.</p>
              </div>
            )}
            {viewerState.kind === "loading-info" && (
              <div className="viewer-empty"><p className="hint">Yükleniyor…</p></div>
            )}
            {viewerState.kind === "error" && (
              <div className="viewer-empty">
                <p className="hint hint-err">Hata: {viewerState.message}</p>
              </div>
            )}

            {/* Viewer */}
            {viewerState.kind === "ready" && (
              <>
                <div className="viewer-statusbar">
                  <span>
                    {sceneProgress && !sceneProgress.done
                      ? `Scene'ler yükleniyor: ${sceneProgress.loaded}/${sceneProgress.total}`
                      : `${viewerState.info.num_frames} frame — "${viewerState.info.scene}" (${formatBytes(viewerState.info.total_size_bytes)})`}
                  </span>
                </div>
                <div className="viewer-canvas">
                  <SplatViewer
                    // key includes singleFrameMode → mode değişince viewer remount
                    key={`${viewerState.info.job_id}-${singleFrameMode ? "single" : "multi"}`}
                    jobId={viewerState.info.job_id}
                    numFrames={viewerState.info.num_frames}
                    currentFrame={currentFrame}
                    singleFrameMode={singleFrameMode}
                    onLoadProgress={(loaded, total) =>
                      setSceneProgress({ loaded, total, done: false })
                    }
                    onReady={() =>
                      setSceneProgress((prev) =>
                        prev ? { ...prev, done: true } : prev
                      )
                    }
                    onError={(msg) =>
                      setViewerState({ kind: "error", message: msg })
                    }
                  />
                </div>
              </>
            )}
          </div>
        )}

        {tab === "interactive" && (
          <InteractivePage />
        )}

        {composeMounted && (
          <div className="compose-host" style={{ display: tab === "compose" ? "block" : "none" }}>
            <ComposePage active={tab === "compose"} />
          </div>
        )}

        {coverageMounted && (
          <div className="compose-host" style={{ display: tab === "coverage" ? "block" : "none" }}>
            <CoveragePage active={tab === "coverage"} prefill={coveragePrefill} />
          </div>
        )}
      </main>

      <div className="api-base" title="Click ⚙ in the topbar to change">
        API: {connection.apiBase}
        {connection.authToken && <span className="api-base-auth"> · auth ✓</span>}
      </div>

      <ConnectionSettings
        open={connSettingsOpen}
        onClose={() => setConnSettingsOpen(false)}
      />
    </div>
  );
}

export default App;
