/**
 * JobSubmitPanel — mode-aware wrapper.
 *
 * v6.0 cleanup: tek devasa form yerine mode picker (Static 3D / 4D Dynamic)
 * + her mod icin ayri component. Preset'ler ve hyperparams alanlari moda
 * gore filtrelenir.
 */
import type { JobMode, ProcessResponse } from "../api";
import { Static3DSubmit } from "./Static3DSubmit";
import { Dynamic4DSubmit } from "./Dynamic4DSubmit";

interface Props {
  mode: JobMode;
  onModeChange: (mode: JobMode) => void;
  onJobSubmitted: (response: ProcessResponse, sceneName: string) => void;
}

export function JobSubmitPanel({ mode, onModeChange, onJobSubmitted }: Props) {
  return (
    <div className="submit-mode-wrapper">
      {/* Mode picker */}
      <div className="mode-picker">
        <button
          type="button"
          className={`mode-btn mode-btn-static ${mode === "static" ? "active" : ""}`}
          onClick={() => onModeChange("static")}
        >
          <span className="mode-btn-icon">📸</span>
          <span className="mode-btn-content">
            <span className="mode-btn-name">Static 3D</span>
            <span className="mode-btn-sub">Photo set / sparse view</span>
          </span>
        </button>
        <button
          type="button"
          className={`mode-btn mode-btn-dynamic ${mode === "dynamic" ? "active" : ""}`}
          onClick={() => onModeChange("dynamic")}
        >
          <span className="mode-btn-icon">🎬</span>
          <span className="mode-btn-content">
            <span className="mode-btn-name">4D Dynamic</span>
            <span className="mode-btn-sub">Video / multi-view</span>
          </span>
        </button>
      </div>

      {/* Mode-spesifik form */}
      {mode === "static" ? (
        <Static3DSubmit onJobSubmitted={onJobSubmitted} />
      ) : (
        <Dynamic4DSubmit onJobSubmitted={onJobSubmitted} />
      )}
    </div>
  );
}
