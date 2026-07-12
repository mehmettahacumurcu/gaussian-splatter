import type { FrameSelectionMode } from "./types";

interface Props {
  mode: FrameSelectionMode;
  fixedFps: string;
  fixedFpsError?: string;
  onModeChange(mode: FrameSelectionMode): void;
  onFixedFpsChange(value: string): void;
}

export function FrameSelectionSection(props: Props) {
  return (
    <section className="nb-stage is-complete">
      <div className="nb-stage-dot" aria-hidden="true" />
      <div className="nb-stage-body">
        <p className="nb-stage-kicker">02 / frames</p>
        <h2>Frame selection</h2>
        <div className="nb-choice-row">
          <label className={`nb-choice ${props.mode === "smart" ? "is-selected" : ""}`}>
            <input
              type="radio"
              name="frame-selection"
              checked={props.mode === "smart"}
              onChange={() => props.onModeChange("smart")}
            />
            <span><strong>Smart</strong><small>Recommended · quality and overlap scored</small></span>
          </label>
          <label className={`nb-choice ${props.mode === "fixed_fps" ? "is-selected" : ""}`}>
            <input
              type="radio"
              name="frame-selection"
              checked={props.mode === "fixed_fps"}
              onChange={() => props.onModeChange("fixed_fps")}
            />
            <span><strong>Fixed FPS</strong><small>Baseline · uniform timeline sampling</small></span>
          </label>
        </div>
        {props.mode === "fixed_fps" && (
          <label className="nb-field nb-compact-field">
            <span>Frames per second</span>
            <input
              type="number"
              value={props.fixedFps}
              onChange={(event) => props.onFixedFpsChange(event.target.value)}
              aria-invalid={Boolean(props.fixedFpsError)}
            />
            {props.fixedFpsError && <small className="nb-error">{props.fixedFpsError}</small>}
          </label>
        )}
      </div>
    </section>
  );
}
