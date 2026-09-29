import { useLayoutEffect, useRef, useState } from "react";
import { Euler, Quaternion } from "three";
import { hexToTint, tintToHex } from "./colorMath";
import { DEFAULT_COLOR, type ColorAdjust, type SceneObject, type Transform } from "./types";

interface Props {
  object: SceneObject;
  cropEditing: boolean;
  onRename: (name: string) => void;
  onTransform: (t: Transform) => void;
  onToggleCrop: (enabled: boolean) => void;
  onCropEditing: (editing: boolean) => void;
  onColor: (c: ColorAdjust | null) => void;
  onSnap: () => void;
  /** "Dikleştir" (non-base splats): align the object's own floor up with the scene up. */
  onStraighten?: () => void;
  /** A backend action is running (disables actions that start another one). */
  busy?: boolean;
}

const RAD = Math.PI / 180;
const AXES = ["X", "Y", "Z"];

/**
 * Numeric input that commits on blur / Enter only (never per keystroke).
 *
 * After an accepted commit the text is left as typed until the new `value`
 * prop arrives (synced in a layout effect, before any later event can run);
 * `committed` remembers the last committed text so a blur right after Enter
 * is a no-op instead of re-committing a stale value.
 */
function NumberField({
  label,
  value,
  step = 0.01,
  disabled,
  isValid,
  onCommit,
}: {
  label: string;
  value: number;
  step?: number;
  disabled?: boolean;
  /** Extra acceptance test; rejected values are reverted in the field. */
  isValid?: (v: number) => boolean;
  onCommit: (v: number) => void;
}) {
  const [text, setText] = useState(value.toFixed(3));
  // Last text/number that the field committed or received as a prop.
  const committed = useRef({ text: value.toFixed(3), value });
  // Bumped on each accepted commit so the next render re-syncs from the prop
  // even when the parent's value came back unchanged (e.g. a rejected edit).
  const [syncTick, setSyncTick] = useState(0);
  useLayoutEffect(() => {
    const shown = value.toFixed(3);
    committed.current = { text: shown, value };
    setText(shown);
  }, [value, syncTick]);

  const commit = () => {
    // Unedited (the text is only the rounded value) or already committed.
    if (text === committed.current.text) return;
    const v = parseFloat(text);
    if (!Number.isFinite(v) || v === committed.current.value || (isValid && !isValid(v))) {
      setText(committed.current.text);
      return;
    }
    committed.current = { text, value: v };
    onCommit(v);
    setSyncTick((n) => n + 1);
  };
  return (
    <input
      type="number"
      aria-label={label}
      step={step}
      value={text}
      disabled={disabled}
      onChange={(e) => setText(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === "Enter") commit();
      }}
    />
  );
}

export function InspectorPanel({
  object,
  cropEditing,
  onRename,
  onTransform,
  onToggleCrop,
  onCropEditing,
  onColor,
  onSnap,
  onStraighten,
  busy = false,
}: Props) {
  const [name, setName] = useState(object.name);
  useLayoutEffect(() => setName(object.name), [object.name]);
  const t = object.transform;
  const locked = object.role === "base";
  const euler = new Euler().setFromQuaternion(new Quaternion(...t.quaternion));
  const deg = [euler.x / RAD, euler.y / RAD, euler.z / RAD];
  const color = object.color ?? DEFAULT_COLOR;

  const commitName = () => {
    const trimmed = name.trim();
    if (trimmed && trimmed !== object.name) onRename(trimmed);
    else setName(object.name);
  };
  const setPos = (i: number, v: number) => {
    const position = [...t.position] as Transform["position"];
    position[i] = v;
    onTransform({ ...t, position });
  };
  const setRot = (i: number, v: number) => {
    const d = [...deg];
    d[i] = v;
    const q = new Quaternion().setFromEuler(new Euler(d[0] * RAD, d[1] * RAD, d[2] * RAD));
    onTransform({ ...t, quaternion: [q.x, q.y, q.z, q.w] });
  };
  const setColor = (patch: Partial<ColorAdjust>) => onColor({ ...color, ...patch });

  return (
    <div className="compose-inspector">
      <label className="compose-field">
        <span>Ad</span>
        <input
          value={name}
          maxLength={128}
          onChange={(e) => setName(e.target.value)}
          onBlur={commitName}
          onKeyDown={(e) => {
            if (e.key === "Enter") e.currentTarget.blur();
          }}
        />
      </label>

      <h4>Transform {locked && <small>(base kilitli)</small>}</h4>
      <div className="compose-vec">
        <span>Konum</span>
        {t.position.map((v, i) => (
          <NumberField key={i} label={`Konum ${AXES[i]}`} value={v} disabled={locked} onCommit={(n) => setPos(i, n)} />
        ))}
      </div>
      <div className="compose-vec">
        <span>Dönüş °</span>
        {deg.map((v, i) => (
          <NumberField
            key={i}
            label={`Dönüş ${AXES[i]}`}
            value={v}
            step={1}
            disabled={locked}
            onCommit={(n) => setRot(i, n)}
          />
        ))}
      </div>
      <div className="compose-vec">
        <span>Ölçek</span>
        <NumberField
          label="Ölçek"
          value={t.scale}
          disabled={locked}
          isValid={(n) => n > 0}
          onCommit={(n) => onTransform({ ...t, scale: n })}
        />
      </div>
      {!locked && (
        <div className="compose-row">
          <button type="button" className="btn-secondary" onClick={onSnap}>
            Zemine oturt
          </button>
          {object.kind === "splat" && onStraighten && (
            <button
              type="button"
              className="btn-secondary"
              disabled={busy}
              title="Objeyi kendi zemin düzlemine göre sahnenin yukarı yönüne çevir"
              onClick={onStraighten}
            >
              Dikleştir
            </button>
          )}
        </div>
      )}

      {object.kind === "splat" && (
        <>
          <h4>Crop</h4>
          <label className="compose-check">
            <input type="checkbox" checked={!!object.crop} onChange={(e) => onToggleCrop(e.target.checked)} /> Crop kutusu
          </label>
          {object.crop && (
            <label className="compose-check">
              <input type="checkbox" checked={cropEditing} onChange={(e) => onCropEditing(e.target.checked)} />{" "}
              Kutuyu gizmo ile düzenle
            </label>
          )}

          <h4>Renk</h4>
          <label className="compose-field">
            <span>Pozlama {color.exposure.toFixed(2)}</span>
            <input
              type="range"
              min={-3}
              max={3}
              step={0.05}
              value={color.exposure}
              onChange={(e) => setColor({ exposure: parseFloat(e.target.value) })}
            />
          </label>
          <label className="compose-field">
            <span>Doygunluk {color.saturation.toFixed(2)}</span>
            <input
              type="range"
              min={0}
              max={2}
              step={0.05}
              value={color.saturation}
              onChange={(e) => setColor({ saturation: parseFloat(e.target.value) })}
            />
          </label>
          <label className="compose-field">
            <span>Ton</span>
            <input type="color" value={tintToHex(color.tint)} onChange={(e) => setColor({ tint: hexToTint(e.target.value) })} />
          </label>
          <button type="button" className="btn-secondary" disabled={!object.color} onClick={() => onColor(null)}>
            Rengi sıfırla
          </button>
        </>
      )}
    </div>
  );
}
