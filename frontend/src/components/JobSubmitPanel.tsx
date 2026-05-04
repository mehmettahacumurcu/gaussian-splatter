/**
 * JobSubmitPanel — mode-aware wrapper.
 *
 * v6.0 cleanup: tek devasa form yerine mode picker (Static 3D / 4D Dynamic)
 * + her mod icin ayri component. Preset'ler ve hyperparams alanlari moda
 * gore filtrelenir.
 *
 * v6.2: "Obje Silme" (edit) modu eklendi — EditSubmit component.
 */
import { useEffect, useState } from "react";
import type { JobMode, ProcessResponse } from "../api";
import { Static3DSubmit } from "./Static3DSubmit";
import { Dynamic4DSubmit } from "./Dynamic4DSubmit";
import { EditSubmit } from "./EditSubmit";
import { listDiskScenes, getSceneInfo } from "../api";

/** Extends JobMode with "edit" for the submit panel picker. */
type SubmitMode = JobMode | "edit";

interface Props {
  mode: JobMode;
  onModeChange: (mode: JobMode) => void;
  onJobSubmitted: (response: ProcessResponse, sceneName: string) => void;
}

export function JobSubmitPanel({ mode, onModeChange, onJobSubmitted }: Props) {
  // "edit" is a local-only mode, not persisted in localStorage via JobMode
  const [submitMode, setSubmitMode] = useState<SubmitMode>(mode);

  // Edit mode: let user pick a scene from disk scenes
  const [editScene, setEditScene] = useState<string>("");
  const [diskScenes, setDiskScenes] = useState<string[] | null>(null);
  const [loadingScenes, setLoadingScenes] = useState(false);
  // Real frame count for the chosen scene (drives EditFramePicker stride).
  // Defaults to a sentinel until /scenes/<scene>/info answers.
  const [sceneNFrames, setSceneNFrames] = useState<number | null>(null);

  // Whenever editScene changes (and we're in edit mode), re-fetch n_frames so
  // the framepicker requests indices that actually exist on disk.
  useEffect(() => {
    if (submitMode !== "edit" || !editScene) return;
    let cancelled = false;
    setSceneNFrames(null);
    getSceneInfo(editScene)
      .then((info) => { if (!cancelled) setSceneNFrames(info.n_frames); })
      .catch((err) => {
        console.error("[JobSubmitPanel] getSceneInfo failed:", err);
        if (!cancelled) setSceneNFrames(0);
      });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [editScene, submitMode]);

  const handleModeClick = async (m: SubmitMode) => {
    setSubmitMode(m);
    if (m !== "edit") {
      onModeChange(m);
    } else {
      // Load disk scenes for the edit scene picker
      if (diskScenes === null) {
        setLoadingScenes(true);
        try {
          const res = await listDiskScenes();
          setDiskScenes(res.scenes.map((s) => s.name));
          if (res.scenes.length > 0 && !editScene) {
            setEditScene(res.scenes[0].name);
          }
        } catch {
          setDiskScenes([]);
        } finally {
          setLoadingScenes(false);
        }
      }
    }
  };

  return (
    <div className="submit-mode-wrapper">
      {/* Mode picker */}
      <div className="mode-picker">
        <button
          type="button"
          className={`mode-btn mode-btn-static ${submitMode === "static" ? "active" : ""}`}
          onClick={() => handleModeClick("static")}
        >
          <span className="mode-btn-icon">📸</span>
          <span className="mode-btn-content">
            <span className="mode-btn-name">Static 3D</span>
            <span className="mode-btn-sub">Photo set / sparse view</span>
          </span>
        </button>
        <button
          type="button"
          className={`mode-btn mode-btn-dynamic ${submitMode === "dynamic" ? "active" : ""}`}
          onClick={() => handleModeClick("dynamic")}
        >
          <span className="mode-btn-icon">🎬</span>
          <span className="mode-btn-content">
            <span className="mode-btn-name">4D Dynamic</span>
            <span className="mode-btn-sub">Video / multi-view</span>
          </span>
        </button>
        <button
          type="button"
          className={`mode-btn ${submitMode === "edit" ? "active" : ""}`}
          onClick={() => handleModeClick("edit")}
          style={{ borderColor: submitMode === "edit" ? "#4a8" : undefined }}
        >
          <span className="mode-btn-icon">✂️</span>
          <span className="mode-btn-content">
            <span className="mode-btn-name">Obje Silme</span>
            <span className="mode-btn-sub">Sahne'den nesne kaldir</span>
          </span>
        </button>
      </div>

      {/* Mode-spesifik form */}
      {submitMode === "static" && (
        <Static3DSubmit onJobSubmitted={onJobSubmitted} />
      )}
      {submitMode === "dynamic" && (
        <Dynamic4DSubmit onJobSubmitted={onJobSubmitted} />
      )}
      {submitMode === "edit" && (
        <div className="submit-panel">
          {/* Scene picker for edit mode */}
          <div className="submit-section">
            <label className="submit-label">Sahne sec</label>
            {loadingScenes && <p className="submit-hint">Sahneler yukleniyor...</p>}
            {diskScenes !== null && diskScenes.length === 0 && (
              <p className="submit-hint hint-warn">
                Diskte sahne bulunamadi. Once Static 3D job calistir.
              </p>
            )}
            {diskScenes !== null && diskScenes.length > 0 && (
              <select
                className="submit-input"
                value={editScene}
                onChange={(e) => setEditScene(e.target.value)}
              >
                {diskScenes.map((s) => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
            )}
            {diskScenes === null && !loadingScenes && (
              <input
                type="text"
                className="submit-input"
                value={editScene}
                onChange={(e) => setEditScene(e.target.value)}
                placeholder="sahne adi (örn. garden)"
              />
            )}
          </div>

          {editScene && sceneNFrames === null && (
            <p className="submit-hint">Sahne bilgisi yukleniyor...</p>
          )}
          {editScene && sceneNFrames === 0 && (
            <p className="submit-hint hint-warn">
              Sahne'de frame bulunamadi (data/{editScene}/frames/ bos veya yok).
              Once Static 3D job calistirip frames uret.
            </p>
          )}
          {editScene && sceneNFrames !== null && sceneNFrames > 0 && (
            <EditSubmit
              scene={editScene}
              sourceCkpt="output/ckpt/ckpt_final.pt"
              totalFrames={sceneNFrames}
              onSubmitted={(jobId) => {
                console.log("[App] edit job submitted:", jobId);
                onJobSubmitted(
                  { job_id: jobId, status: "queued", status_url: `/status/${jobId}`, message: "Edit job queued" },
                  editScene,
                );
              }}
            />
          )}
        </div>
      )}
    </div>
  );
}
