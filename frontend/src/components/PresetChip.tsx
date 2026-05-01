import { InfoButton } from "./ui/InfoButton";
import { PRESET_META, PRESET_DEFAULTS, REFERENCE_PRESET, type PresetMeta } from "../presets";
import type { JobMode, HyperParams } from "../api";

interface Props {
  pipeline: JobMode;
  presetId: string;
  name: string;
  badge?: string;
  duration: string;
  desc: string;
  selected: boolean;
  onSelect: () => void;
}

export function PresetChip({
  pipeline, presetId, name, badge, duration, desc, selected, onSelect,
}: Props) {
  const meta = PRESET_META[pipeline]?.[presetId];
  const refId = REFERENCE_PRESET[pipeline];
  const refDefaults = PRESET_DEFAULTS[pipeline]?.[refId];
  const myDefaults = PRESET_DEFAULTS[pipeline]?.[presetId];

  return (
    <label className={`preset-chip ${selected ? "active" : ""}`}>
      <div className="preset-chip-row">
        <input
          type="radio"
          checked={selected}
          onChange={onSelect}
        />
        <div className="preset-chip-content">
          <div className="preset-name">
            {name} {badge && <span aria-hidden>{badge}</span>}{" "}
            <span className="preset-duration">({duration})</span>
          </div>
          <div className="preset-desc">{desc}</div>
        </div>
        {meta && (
          <InfoButton label={`${name} preset details`}>
            {renderDetails(meta, presetId, refId, myDefaults, refDefaults)}
          </InfoButton>
        )}
      </div>
    </label>
  );
}

function renderDetails(
  meta: PresetMeta,
  presetId: string,
  refId: string,
  myDefaults: Partial<HyperParams> | undefined,
  refDefaults: Partial<HyperParams> | undefined,
) {
  return (
    <>
      <div><strong>Use case:</strong> {meta.useCase}</div>
      <div><strong>VRAM:</strong> {meta.vram} · <strong>Disk:</strong> {meta.diskEstimate}</div>
      <div><strong>Key hyperparams:</strong> {meta.keyHyperparams.join(" · ")}</div>
      <div>
        <strong>Pick when:</strong>
        <ul>{meta.pickWhen.map((s) => <li key={s}>{s}</li>)}</ul>
      </div>
      <div>
        <strong>Avoid when:</strong>
        <ul>{meta.avoidWhen.map((s) => <li key={s}>{s}</li>)}</ul>
      </div>
      {presetId !== refId && myDefaults && refDefaults && (
        <div className="preset-vs">
          <strong>vs {refId}:</strong> {diffSummary(myDefaults, refDefaults)}
        </div>
      )}
    </>
  );
}

function diffSummary(
  a: Partial<HyperParams>,
  b: Partial<HyperParams>,
): string {
  const parts: string[] = [];
  const keys = new Set<string>([...Object.keys(a), ...Object.keys(b)]);
  for (const k of keys) {
    const av = (a as Record<string, unknown>)[k];
    const bv = (b as Record<string, unknown>)[k];
    if (av !== undefined && av !== bv) {
      parts.push(`${k}: ${bv ?? "—"} → ${av}`);
    }
  }
  return parts.length ? parts.join(" · ") : "(same as reference)";
}
