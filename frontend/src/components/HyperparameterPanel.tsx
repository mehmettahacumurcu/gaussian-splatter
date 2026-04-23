/**
 * HyperparameterPanel — tüm training hiperparametrelerini override eder.
 *
 * Gruplar (her biri collapsible):
 *   - Temel: iters, resolution, num_timestamps, fps
 *   - Loss weights: lambda_ssim + motion regularizers
 *   - Learning rates: lr_deform, lr_means
 *   - Density control: start/end/interval/thresholds
 *   - Model: sh_degree, hexplane, mlp_width
 *
 * Her field "optional override": boş bırakılırsa backend default'u kullanır.
 * Preset (smoke/cloud) seçili ise preset'in yazdığı değer üzerine uygulanır.
 */
import { useState } from "react";
import type { HyperParams } from "../api";

interface Props {
  value: HyperParams;
  onChange: (next: HyperParams) => void;
  preset: "smoke" | "full" | "cloud";
}

interface FieldSpec {
  key: keyof HyperParams;
  label: string;
  placeholder: string;
  help?: string;
  kind: "number" | "string";
  min?: number;
  max?: number;
  step?: number;
}

const GROUPS: { title: string; icon: string; fields: FieldSpec[] }[] = [
  {
    title: "Temel",
    icon: "⚙",
    fields: [
      { key: "iters", label: "Iterations", placeholder: "preset", kind: "number", min: 50, step: 100 },
      { key: "resolution", label: "Resolution", placeholder: "640x360", kind: "string" },
      { key: "num_timestamps", label: "Timestamps (export)", placeholder: "preset", kind: "number", min: 2, step: 1 },
      { key: "fps", label: "Frame extraction FPS", placeholder: "10", kind: "number", min: 1, step: 1 },
    ],
  },
  {
    title: "Loss ağırlıkları",
    icon: "λ",
    fields: [
      { key: "lambda_ssim", label: "λ SSIM", placeholder: "0.2", kind: "number", step: 0.05, help: "0=L1 only, 1=SSIM only" },
      { key: "lambda_deform_reg", label: "λ deform L2", placeholder: "0.001", kind: "number", step: 0.001, help: "Δpos/Δquat/Δscale mag regularizer" },
      { key: "lambda_smoothness", label: "λ temporal smoothness", placeholder: "0.01", kind: "number", step: 0.005, help: "D(t) vs D(t+dt)" },
      { key: "lambda_rigidity", label: "λ isometric rigidity", placeholder: "0.01", kind: "number", step: 0.005, help: "local geometry koruma" },
    ],
  },
  {
    title: "Learning rates",
    icon: "↓",
    fields: [
      { key: "lr_deform", label: "LR deformation", placeholder: "0.001", kind: "number", step: 0.0005 },
      { key: "lr_means", label: "LR means", placeholder: "0.00016", kind: "number", step: 0.0001 },
    ],
  },
  {
    title: "Density control",
    icon: "●",
    fields: [
      { key: "density_start_iter", label: "Density start iter", placeholder: "preset", kind: "number", min: 0, step: 100 },
      { key: "density_end_iter", label: "Density end iter", placeholder: "preset", kind: "number", min: 0, step: 100 },
      { key: "density_interval", label: "Density interval", placeholder: "100", kind: "number", min: 1, step: 10 },
      { key: "densify_grad_threshold", label: "Densify grad threshold", placeholder: "0.0002", kind: "number", step: 0.0001 },
      { key: "prune_min_opacity", label: "Prune min opacity", placeholder: "0.005", kind: "number", step: 0.001 },
      { key: "prune_max_scale", label: "Prune max scale", placeholder: "0.1", kind: "number", step: 0.01 },
    ],
  },
  {
    title: "Model mimarisi",
    icon: "◇",
    fields: [
      { key: "sh_degree", label: "SH degree", placeholder: "3", kind: "number", min: 0, max: 3, step: 1 },
      { key: "hexplane_resolution", label: "HexPlane resolution", placeholder: "64", kind: "number", min: 16, step: 16 },
      { key: "hexplane_feat_dim", label: "HexPlane feature dim", placeholder: "32", kind: "number", min: 8, step: 8 },
      { key: "mlp_width", label: "Deformation MLP width", placeholder: "256", kind: "number", min: 32, step: 32 },
    ],
  },
];

export function HyperparameterPanel({ value, onChange, preset }: Props) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set(["Temel"]));

  const toggle = (title: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(title)) next.delete(title);
      else next.add(title);
      return next;
    });
  };

  const updateField = (key: keyof HyperParams, raw: string, kind: "number" | "string") => {
    const next = { ...value };
    if (raw === "") {
      next[key] = null;
    } else if (kind === "number") {
      const n = parseFloat(raw);
      next[key] = Number.isFinite(n) ? (n as any) : null;
    } else {
      next[key] = raw as any;
    }
    onChange(next);
  };

  const overrideCount = Object.values(value).filter(
    (v) => v !== null && v !== undefined && v !== ""
  ).length;

  return (
    <div className="hparam-panel">
      <div className="hparam-panel-header">
        <span>Hiperparametreler</span>
        <span className="hparam-summary">
          {overrideCount === 0
            ? `${preset} preset default'ları`
            : `${overrideCount} override`}
        </span>
      </div>
      {GROUPS.map((g) => {
        const isOpen = expanded.has(g.title);
        const groupOverrides = g.fields.filter(
          (f) => value[f.key] !== null && value[f.key] !== undefined && value[f.key] !== ""
        ).length;
        return (
          <div key={g.title} className={`hparam-group ${isOpen ? "open" : ""}`}>
            <button
              className="hparam-group-header"
              onClick={() => toggle(g.title)}
              type="button"
            >
              <span className="hparam-group-icon">{g.icon}</span>
              <span className="hparam-group-title">{g.title}</span>
              {groupOverrides > 0 && (
                <span className="hparam-badge">{groupOverrides}</span>
              )}
              <span className="hparam-chevron">{isOpen ? "▾" : "▸"}</span>
            </button>
            {isOpen && (
              <div className="hparam-fields">
                {g.fields.map((f) => {
                  const v = value[f.key];
                  return (
                    <label key={f.key} className="hparam-field">
                      <span className="hparam-label" title={f.help}>
                        {f.label}
                      </span>
                      <input
                        type={f.kind === "number" ? "number" : "text"}
                        placeholder={f.placeholder}
                        value={v === null || v === undefined ? "" : String(v)}
                        min={f.min}
                        max={f.max}
                        step={f.step}
                        onChange={(e) => updateField(f.key, e.target.value, f.kind)}
                      />
                    </label>
                  );
                })}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
