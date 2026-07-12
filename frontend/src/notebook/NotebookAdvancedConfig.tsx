import type { NotebookAdvancedDraft } from "./types";

interface Props {
  value: NotebookAdvancedDraft;
  errors: Partial<Record<keyof NotebookAdvancedDraft, string>>;
  onChange<K extends keyof NotebookAdvancedDraft>(key: K, value: NotebookAdvancedDraft[K]): void;
}

const NUMERIC_FIELDS: Array<[keyof NotebookAdvancedDraft, string, string]> = [
  ["resolution_long_edge_cap", "Resolution long edge", "e.g. 1920"],
  ["lambda_ssim", "SSIM weight", "0–1"],
  ["lambda_lpips", "LPIPS weight", "0–1"],
  ["lambda_depth", "Depth weight", "0–1"],
  ["density_start_iter", "Density start", "iteration"],
  ["density_end_iter", "Density end", "iteration"],
  ["density_interval", "Density interval", "iterations"],
  ["densify_grad_threshold", "Densify gradient threshold", "0–0.1"],
  ["prune_min_opacity", "Prune minimum opacity", "0–1"],
  ["prune_max_scale", "Prune maximum scale", "0–1"],
  ["opacity_reset_interval", "Opacity reset interval", "iteration"],
  ["sh_degree", "Spherical harmonics degree", "0–3"],
];

export function NotebookAdvancedConfig({ value, errors, onChange }: Props) {
  return (
    <section className="nb-stage is-complete">
      <div className="nb-stage-dot" aria-hidden="true" />
      <div className="nb-stage-body">
        <p className="nb-stage-kicker">05 / tuning</p>
        <h2>Advanced settings</h2>
        <details className="nb-advanced">
          <summary>Approved quality controls <span>collapsed by default</span></summary>
          <div className="nb-advanced-grid">
            {(["foundation", "run_eval"] as const).map((field) => (
              <label className="nb-field" key={field}>
                <span>{field === "foundation" ? "Foundation depth" : "Holdout evaluation"}</span>
                <select
                  value={value[field] === null ? "default" : String(value[field])}
                  onChange={(event) =>
                    onChange(
                      field,
                      event.target.value === "default" ? null : event.target.value === "true",
                    )
                  }
                >
                  <option value="default">Profile default</option>
                  <option value="true">Enabled</option>
                  <option value="false">Disabled</option>
                </select>
              </label>
            ))}
            {NUMERIC_FIELDS.map(([field, label, placeholder]) => (
              <label className="nb-field" key={field}>
                <span>{label}</span>
                <input
                  type="text"
                  inputMode="decimal"
                  value={value[field] as string}
                  placeholder={placeholder}
                  onChange={(event) => onChange(field, event.target.value)}
                  aria-invalid={Boolean(errors[field])}
                />
                {errors[field] && <small className="nb-error">{errors[field]}</small>}
              </label>
            ))}
            <label className="nb-field nb-wide-field">
              <span>Multires schedule</span>
              <input
                type="text"
                value={value.multires_schedule}
                placeholder="0:720,25000:1080"
                onChange={(event) => onChange("multires_schedule", event.target.value)}
                aria-invalid={Boolean(errors.multires_schedule)}
              />
              {errors.multires_schedule && <small className="nb-error">{errors.multires_schedule}</small>}
            </label>
          </div>
        </details>
      </div>
    </section>
  );
}
