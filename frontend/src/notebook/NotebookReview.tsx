import type { DriveFolderResolution } from "./drivePath";
import type { NotebookProfileMetadata, StaticNotebookRunSpec } from "./types";

interface Props {
  spec: StaticNotebookRunSpec | null;
  profile: NotebookProfileMetadata | null;
  path: DriveFolderResolution;
  warnings: readonly string[];
  generating: boolean;
  onGenerate(): void;
}

export function NotebookReview(props: Props) {
  const iterations = props.spec?.quality.n_iters ?? props.profile?.n_iters;
  const gaussians = props.spec?.quality.max_gaussians ?? props.profile?.max_gaussians;
  return (
    <section className={`nb-stage nb-review-stage ${props.spec ? "is-complete" : ""}`}>
      <div className="nb-stage-dot" aria-hidden="true" />
      <div className="nb-stage-body">
        <p className="nb-stage-kicker">06 / handoff</p>
        <h2>Review</h2>
        <div className="nb-review-grid">
          <span>Input</span><code>{props.path.ok ? props.path.displayInput : "—"}</code>
          <span>Result</span><code>{props.path.ok ? props.path.displayOutput : "—"}</code>
          <span>Frames</span><strong>{props.spec?.frame_selection.mode === "fixed_fps" ? `Fixed ${props.spec.frame_selection.fixed_fps} FPS` : "Smart selection"}</strong>
          <span>Compute</span><strong>{props.profile?.label ?? "—"} · {iterations?.toLocaleString() ?? "—"} iter · {gaussians?.toLocaleString() ?? "—"} Gaussians</strong>
        </div>
        {props.warnings.map((warning) => <p className="nb-warning" key={warning}>{warning}</p>)}
        <p className="nb-no-viewer">Output is a standard portable PLY. Open it in SuperSplat or the 4DGS Studio viewer; no web viewer is bundled.</p>
        <button
          type="button"
          className="nb-generate-button"
          disabled={!props.spec || props.generating}
          onClick={props.onGenerate}
        >
          {props.generating ? "Generating notebook…" : "Generate and download notebook"}
        </button>
      </div>
    </section>
  );
}
