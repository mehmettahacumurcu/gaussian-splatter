/**
 * 4DGS Viewer - ana uygulama.
 *
 * KRITIK NOKTA: SplatViewer bir kere mount olduktan sonra UNMOUNT ETMEMELI.
 * Unmount etmek viewer.dispose() cagirir, mid-load ise scene disposed hatasi
 * + removeChild hatasi atar. Bu yuzden loadState "ready"ye alinir alinmaz
 * orada kalir; scene yukleme ilerlemesi ayrir bir state'te tutulur.
 */
import { useEffect, useState, useCallback } from "react";
import "./App.css";
import { SplatViewer } from "./components/SplatViewer";
import { TimelineSlider } from "./components/TimelineSlider";
import {
  API_BASE,
  getHealth,
  getSplatInfo,
  listJobs,
  listDiskScenes,
  type HealthResponse,
  type SceneListItem,
  type SplatInfo,
} from "./api";

type LoadState =
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
  const [jobIdInput, setJobIdInput] = useState("");
  const [loadState, setLoadState] = useState<LoadState>({ kind: "idle" });
  const [sceneProgress, setSceneProgress] = useState<SceneProgress | null>(null);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [diskScenes, setDiskScenes] = useState<SceneListItem[] | null>(null);
  const [diskPanelOpen, setDiskPanelOpen] = useState(false);
  const [diskLoading, setDiskLoading] = useState(false);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch((e) => console.warn("Health check fail:", e));
  }, []);

  const loadJob = useCallback(async (identifier: string) => {
    const id = identifier.trim();
    if (!id) {
      setLoadState({ kind: "error", message: "Bos identifier" });
      return;
    }
    // Yeni yukleme: onceki viewer'i varsa unmount olacak, yeni ready ile tekrar mount olacak
    setLoadState({ kind: "loading-info" });
    setSceneProgress(null);
    setCurrentFrame(0);
    setDiskPanelOpen(false);
    try {
      const info = await getSplatInfo(id);
      if (info.num_frames === 0) {
        setLoadState({ kind: "error", message: "Bu sahnenin .ply ciktisi yok" });
        return;
      }
      setSceneProgress({ loaded: 0, total: info.num_frames, done: false });
      setLoadState({ kind: "ready", info });
    } catch (e) {
      setLoadState({ kind: "error", message: String(e) });
    }
  }, []);

  const loadLatest = useCallback(async () => {
    try {
      const { jobs } = await listJobs();
      const done = jobs.find((j) => j.status === "completed");
      if (!done) {
        setLoadState({
          kind: "error",
          message:
            "Registry'de tamamlanmis job yok. 'Diskten yukle' ile eski sahneleri de gorebilirsin.",
        });
        return;
      }
      setJobIdInput(done.id);
      loadJob(done.id);
    } catch (e) {
      setLoadState({ kind: "error", message: String(e) });
    }
  }, [loadJob]);

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
      setLoadState({ kind: "error", message: String(e) });
      setDiskScenes([]);
    } finally {
      setDiskLoading(false);
    }
  }, [diskPanelOpen]);

  const renderStatus = () => {
    // Scene loading'i oncelik ver
    if (sceneProgress && !sceneProgress.done && loadState.kind === "ready") {
      return (
        <p className="hint">
          Scene'ler yukleniyor: {sceneProgress.loaded}/{sceneProgress.total}
        </p>
      );
    }
    switch (loadState.kind) {
      case "idle":
        return (
          <p className="hint">
            Bir job_id yapistir, "Diskten yukle" ile sahne sec, ya da
            "Son tamamlanmis" butonuna bas.
          </p>
        );
      case "loading-info":
        return <p className="hint">Sahne bilgisi aliniyor...</p>;
      case "ready":
        return (
          <p className="hint hint-ok">
            {loadState.info.num_frames} frame yuklendi - "{loadState.info.scene}"
            ({formatBytes(loadState.info.total_size_bytes)})
            {loadState.info.source === "disk" ? " - diskten" : " - registry'den"}
          </p>
        );
      case "error":
        return <p className="hint hint-err">Hata: {loadState.message}</p>;
    }
  };

  const backendBadge = (() => {
    if (!health) return <span className="badge badge-unknown">Backend: baglanti yok</span>;
    const gpu = health.gpu_available ? `GPU: ${health.gpu_name ?? "?"}` : "GPU: yok";
    return (
      <span className="badge badge-ok">
        Backend OK - {gpu} - Aktif job: {health.active_jobs}
      </span>
    );
  })();

  return (
    <div className="app">
      <header className="app-header">
        <div className="title-row">
          <h1>4DGS Viewer</h1>
          {backendBadge}
        </div>
        <div className="control-row">
          <input
            type="text"
            value={jobIdInput}
            placeholder="job_id veya sahne adi (orn. api_test3)"
            onChange={(e) => setJobIdInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") loadJob(jobIdInput);
            }}
          />
          <button className="btn-primary" onClick={() => loadJob(jobIdInput)}>
            Yukle
          </button>
          <button className="btn-secondary" onClick={loadLatest}>
            Son tamamlanmis
          </button>
          <button className="btn-secondary" onClick={toggleDiskPanel}>
            Diskten yukle {diskPanelOpen ? "UP" : "DOWN"}
          </button>
        </div>

        {diskPanelOpen && (
          <div className="disk-panel">
            {diskLoading && <p className="hint">Disk taraniyor...</p>}
            {!diskLoading && diskScenes && diskScenes.length === 0 && (
              <p className="hint hint-err">
                Diskte .ply ciktisi olan sahne bulunamadi.
              </p>
            )}
            {!diskLoading && diskScenes && diskScenes.length > 0 && (
              <ul className="scene-list">
                {diskScenes.map((s) => (
                  <li
                    key={s.name}
                    className="scene-item"
                    onClick={() => {
                      setJobIdInput(s.name);
                      loadJob(s.name);
                    }}
                    title={`${s.num_frames} frame - ${formatBytes(s.total_size_bytes)}`}
                  >
                    <span className="scene-name">{s.name}</span>
                    <span className="scene-meta">
                      {s.num_frames} frame - {formatBytes(s.total_size_bytes)} -{" "}
                      {formatTime(s.modified_ts)}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}

        {renderStatus()}
      </header>

      <main className="app-main">
        {loadState.kind === "ready" && (
          <SplatViewer
            key={loadState.info.job_id}
            jobId={loadState.info.job_id}
            numFrames={loadState.info.num_frames}
            currentFrame={currentFrame}
            onLoadProgress={(loaded, total) =>
              setSceneProgress({ loaded, total, done: false })
            }
            onReady={() =>
              setSceneProgress((prev) =>
                prev ? { ...prev, done: true } : prev
              )
            }
            onError={(msg) => setLoadState({ kind: "error", message: msg })}
          />
        )}
      </main>

      {loadState.kind === "ready" && (
        <footer className="app-footer">
          <TimelineSlider
            numFrames={loadState.info.num_frames}
            currentFrame={currentFrame}
            onFrameChange={setCurrentFrame}
            baseFps={10}
          />
        </footer>
      )}

      <div className="api-base">API: {API_BASE}</div>
    </div>
  );
}

export default App;
