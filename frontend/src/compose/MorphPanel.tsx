import { useEffect, useState } from "react";
import type { MorphPlayback, MorphState, MorphStatus } from "./morphTypes";
import type { SceneObject } from "./types";

interface Props {
  morph: MorphState;
  playback: MorphPlayback;
  status: MorphStatus;
  objects: SceneObject[];
  onChange: (patch: Partial<MorphState>) => void;
  recording?: { progress: number } | null;
  onRecord?: () => void;
  onCancelRecording?: () => void;
  recordingError?: string | null;
}

function NumericControl({
  label,
  value,
  min,
  max,
  step,
  disabled = false,
  onCommit,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  disabled?: boolean;
  onCommit: (value: number) => void;
}) {
  const [text, setText] = useState(String(value));
  useEffect(() => setText(String(value)), [value, disabled]);
  const commit = () => {
    if (disabled) {
      setText(String(value));
      return;
    }
    const parsed = Number(text);
    const next = text.trim() && Number.isFinite(parsed) ? Math.max(min, Math.min(max, parsed)) : value;
    const rounded = step === 1 ? Math.round(next) : next;
    setText(String(rounded));
    if (rounded !== value) onCommit(rounded);
  };
  return (
    <label className="compose-field">
      <span>{label}</span>
      <input
        type="number"
        value={text}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        onChange={(event) => { if (!disabled) setText(event.target.value); }}
        onBlur={commit}
        onKeyDown={(event) => {
          if (event.key === "Enter") event.currentTarget.blur();
        }}
      />
    </label>
  );
}

export function MorphPanel({ morph, playback, status, objects, onChange, recording, onRecord, onCancelRecording, recordingError }: Props) {
  const locked = !!recording;
  const change = (patch: Partial<MorphState>) => { if (!locked) onChange(patch); };
  const [liveTime, setLiveTime] = useState(morph.t);
  useEffect(() => {
    setLiveTime(morph.t);
    if (!morph.playing) return;
    const timer = window.setInterval(() => setLiveTime(playback.t), 100);
    return () => window.clearInterval(timer);
  }, [morph.t, morph.playing, playback]);

  const source = objects.find((object) => object.id === morph.sourceId && object.kind === "splat");
  const target = objects.find((object) => object.id === morph.targetId && object.kind === "splat");
  const hasPair = !!source && !!target && source.id !== target.id;
  const ready = hasPair && morph.enabled && status.phase === "ready";
  const time = morph.playing ? liveTime : morph.t;

  return (
    <section className="compose-inspector compose-morph" aria-label="Morph">
      <h3>Morph <small>Önizleme</small></h3>
      <p className="muted compose-morph-hint">Obje listesindeki A ve B düğmeleriyle iki splat seç.</p>
      <div className="compose-morph-pair">
        <span title={source?.name}><b>A</b> {source?.name ?? "Başlangıç seçilmedi"}</span>
        <span title={target?.name}><b>B</b> {target?.name ?? "Hedef seçilmedi"}</span>
      </div>
      <div className="compose-row">
        <button
          type="button"
          className={morph.enabled ? "btn-secondary" : "btn-primary"}
          disabled={locked || !hasPair}
          onClick={() => change({ enabled: !morph.enabled, playing: false, t: 0 })}
        >
          {morph.enabled ? "Önizlemeyi kapat" : "Morph hazırla"}
        </button>
        <button
          type="button"
          className="btn-secondary"
          disabled={locked || !hasPair}
          aria-label="Morph A ve B yer değiştir"
          onClick={() => change({
            sourceId: morph.targetId,
            targetId: morph.sourceId,
            enabled: false,
            playing: false,
            t: 0,
          })}
        >
          A ↔ B
        </button>
      </div>
      {morph.enabled && (
        <p className={`compose-morph-status${status.phase === "error" ? " error" : ""}`} role="status" aria-label="Morph durumu">
          {status.phase === "loading" || status.phase === "idle" ? "Splatlar eşleştiriliyor…" : null}
          {status.phase === "ready" ? `${(status.count ?? 0).toLocaleString()} parçacık${status.precomputeMs !== undefined ? ` · ${(status.precomputeMs / 1000).toFixed(2)} sn hazırlık` : ""}` : null}
          {status.phase === "error" ? status.message ?? "Morph hazırlanamadı." : null}
        </p>
      )}
      <div className="compose-row">
        <button
          type="button"
          className="btn-primary"
          disabled={locked || !ready}
          onClick={() => change({ playing: !morph.playing, t: morph.playing ? playback.t : morph.t >= 1 ? 0 : morph.t })}
        >
          {morph.playing ? "Duraklat" : "Oynat"}
        </button>
        <button
          type="button"
          className="btn-secondary"
          disabled={locked || !ready}
          onClick={() => change({ t: 0, playing: false })}
        >
          Başa dön
        </button>
        <output className="compose-morph-time" aria-label="Morph ilerleme" aria-live="off">{Math.round(time * 100)}%</output>
      </div>
      <label className="compose-field">
        <span>Zaman çizelgesi</span>
        <input
          type="range"
          min={0}
          max={1}
          step={0.001}
          value={time}
          disabled={locked || !ready}
          onChange={(event) => change({ t: Number(event.target.value), playing: false })}
        />
      </label>
      <NumericControl label="Süre (sn)" value={morph.duration} min={0.5} max={120} step={0.5} disabled={locked} onCommit={(duration) => change({ duration })} />
      <label className="compose-field">
        <span>Dissolve · {Math.round(morph.dissolve * 100)}%</span>
        <input
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={morph.dissolve}
          disabled={locked}
          onChange={(event) => change({ dissolve: Number(event.target.value) })}
        />
        <small className="muted">Parçacık bulutunun yayılması</small>
      </label>
      <label className="compose-field">
        <span>Target blend · {Math.round(morph.targetBlend * 100)}%</span>
        <input
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={morph.targetBlend}
          disabled={locked}
          onChange={(event) => change({ targetBlend: Number(event.target.value) })}
        />
        <small className="muted">B hedefine yaklaşma miktarı</small>
      </label>
      <NumericControl label="Seed" value={morph.seed} min={0} max={4294967295} step={1} disabled={locked} onCommit={(seed) => change({ seed, playing: false, t: 0 })} />
      <div className="compose-row">
        <button
          type="button"
          className="btn-primary"
          disabled={locked || !ready || !onRecord}
          onClick={onRecord}
        >
          {recording ? `Kaydediliyor… %${Math.round(Math.max(0, Math.min(1, recording.progress)) * 100)}` : "Videoyu kaydet"}
        </button>
        {recording && (
          <button type="button" className="btn-secondary" onClick={onCancelRecording} disabled={!onCancelRecording}>
            İptal
          </button>
        )}
      </div>
      {recordingError && <p className="compose-morph-status error" role="alert">{recordingError}</p>}
      <p className="muted compose-morph-hint">Geçiş sırasında DC renk kullanılır. Önizleme sahne kaydına ve export’a dahil değildir.</p>
    </section>
  );
}
