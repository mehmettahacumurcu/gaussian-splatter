/**
 * 4DGS Viewer — ana uygulama (tab navigation).
 *
 * Tab'lar:
 *   - Yeni Job: video seç + hiperparametreler + submit
 *   - Jobs: aktif + geçmiş job listesi, auto-refresh
 *   - Viewer: seçilen job'ın 4D splat render'ı + timeline
 */
import { useCallback, useEffect, useState } from "react";
import "./App.css";
import { SplatViewer } from "./components/SplatViewer";
import { SplatViewerSpark } from "./components/SplatViewerSpark";
import { TimelineSlider } from "./components/TimelineSlider";
import { JobSubmitPanel } from "./components/JobSubmitPanel";
import { JobsList } from "./components/JobsList";
import { TrainingAnalytics } from "./components/TrainingAnalytics";
import { NvsEvalPanel } from "./components/NvsEvalPanel";
import {
  API_BASE,
  getHealth,
  getSplatInfo,
  listJobs,
  listDiskScenes,
  type HealthResponse,
  type Job,
  type JobMode,
  type SceneListItem,
  type SplatInfo,
} from "./api";

type Tab = "submit" | "jobs" | "viewer" | "analytics" | "eval";

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
  const [tab, setTab] = useState<Tab>("submit");
  // v6.0 — Pipeline mode (Static 3D / 4D Dynamic). Submit panel mode'a göre
  // alt component render eder. localStorage'a hatirlatir, refresh'te kalır.
  const [mode, setMode] = useState<JobMode>(() => {
    try {
      const saved = window.localStorage.getItem("4dgs.mode");
      return saved === "static" || saved === "dynamic" ? saved : "dynamic";
    } catch {
      return "dynamic";
    }
  });
  const handleModeChange = useCallback((m: JobMode) => {
    setMode(m);
    try { window.localStorage.setItem("4dgs.mode", m); } catch { /* ignore */ }
  }, []);
  const [health, setHealth] = useState<HealthResponse | null>(null);

  // Viewer state
  const [viewerState, setViewerState] = useState<ViewerState>({ kind: "idle" });
  const [sceneProgress, setSceneProgress] = useState<SceneProgress | null>(null);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [jobIdInput, setJobIdInput] = useState("");
  const [diskScenes, setDiskScenes] = useState<SceneListItem[] | null>(null);
  const [diskPanelOpen, setDiskPanelOpen] = useState(false);
  const [diskLoading, setDiskLoading] = useState(false);
  // v3.7.7 Debug: tek frame yükleme modu — visibility toggle bypass
  const [singleFrameMode, setSingleFrameMode] = useState(false);
  const [viewerEngine, setViewerEngine] = useState<"legacy" | "spark">("legacy");
  // Analytics — son/aktif job için
  const [analyticsScene, setAnalyticsScene] = useState<string>("");

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

  const handleJobSubmitted = useCallback((_resp: any, sceneName: string) => {
    // Job başarıyla gönderildi, Jobs tab'ına geç — kullanıcı ilerlemeyi takip etsin
    // Analytics'e de aynı sahneyi otomatik bağla
    setAnalyticsScene(sceneName);
    setTab("jobs");
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
      const done = jobs.find((j) => j.status === "completed");
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
    if (!health) return <span className="badge badge-unknown">Backend: bağlantı yok</span>;
    const gpu = health.gpu_available ? `GPU: ${health.gpu_name ?? "?"}` : "GPU: yok";
    return (
      <span className="badge badge-ok">
        Backend OK · {gpu} · Aktif: {health.active_jobs}
      </span>
    );
  })();

  return (
    <div className="app">
      {/* Üst bar */}
      <header className="app-topbar">
        <div className="app-brand">
          <h1>4DGS Viewer</h1>
        </div>
        <nav className="app-tabs">
          <button
            className={`tab-btn ${tab === "submit" ? "active" : ""}`}
            onClick={() => setTab("submit")}
          >
            Yeni Job
          </button>
          <button
            className={`tab-btn ${tab === "jobs" ? "active" : ""}`}
            onClick={() => setTab("jobs")}
          >
            Jobs
            {health && health.active_jobs > 0 && (
              <span className="tab-badge">{health.active_jobs}</span>
            )}
          </button>
          <button
            className={`tab-btn ${tab === "viewer" ? "active" : ""}`}
            onClick={() => setTab("viewer")}
          >
            Viewer
          </button>
          <button
            className={`tab-btn ${tab === "analytics" ? "active" : ""}`}
            onClick={() => setTab("analytics")}
          >
            Analiz
          </button>
          <button
            className={`tab-btn ${tab === "eval" ? "active" : ""}`}
            onClick={() => setTab("eval")}
          >
            Eval
          </button>
        </nav>
        <div className="app-status">{backendBadge}</div>
      </header>

      {/* İçerik */}
      <main className="app-content">
        {tab === "submit" && (
          <JobSubmitPanel
            mode={mode}
            onModeChange={handleModeChange}
            onJobSubmitted={handleJobSubmitted}
          />
        )}

        {tab === "jobs" && (
          <div className="tab-content">
            <h2 className="tab-title">Tüm Jobs</h2>
            <JobsList onViewJob={handleViewJob} />
          </div>
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
                    const done = jobs.find((j) => j.status === "completed");
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
                  <span style={{ marginLeft: 16, display: "inline-flex", gap: 6, alignItems: "center" }}>
                    <span style={{ fontSize: 11, color: "var(--text-muted)" }}>Engine:</span>
                    <button
                      type="button"
                      onClick={() => setViewerEngine("legacy")}
                      className={viewerEngine === "legacy" ? "viewer-engine-btn active" : "viewer-engine-btn"}
                    >
                      legacy (mkkellogg)
                    </button>
                    <button
                      type="button"
                      onClick={() => setViewerEngine("spark")}
                      className={viewerEngine === "spark" ? "viewer-engine-btn active" : "viewer-engine-btn"}
                    >
                      Spark (4DGS) ✨
                    </button>
                  </span>
                </div>
                <div className="viewer-canvas">
                  {viewerEngine === "spark" ? (
                    <SplatViewerSpark
                      key={`spark-${viewerState.info.job_id}`}
                      jobId={viewerState.info.job_id}
                      numFrames={viewerState.info.num_frames}
                      currentFrame={currentFrame}
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
                  ) : (
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
                  )}
                </div>
                <div className="viewer-timeline">
                  <TimelineSlider
                    numFrames={viewerState.info.num_frames}
                    currentFrame={currentFrame}
                    onFrameChange={setCurrentFrame}
                    baseFps={10}
                  />
                </div>
              </>
            )}
          </div>
        )}
      </main>

      <div className="api-base">API: {API_BASE}</div>
    </div>
  );
}

export default App;
