import { useEffect, useState } from "react";
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
}

const RAD = Math.PI / 180;

/** Numeric input that commits on blur / Enter only (never per keystroke). */
function NumberField({
  value,
  step = 0.01,
  disabled,
  onCommit,
}: {
  value: number;
  step?: number;
  disabled?: boolean;
  onCommit: (v: number) => void;
}) {
  const [text, setText] = useState(value.toFixed(3));
  useEffect(() => setText(value.toFixed(3)), [value]);
  const commit = () => {
    // Unedited field: the text is only the rounded value, don't write it back.
    if (text === value.toFixed(3)) return;
    const v = parseFloat(text);
    if (Number.isFinite(v) && v !== value) onCommit(v);
    // Accepted → the new `value` prop replaces this; rejected → show the old value again.
    setText(value.toFixed(3));
  };
  return (
    <input
      type="number"
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

export function InspectorPanel({ object, cropEditing, onRename, onTransform, onToggleCrop, onCropEditing, onColor, onSnap }: Props) {
  const [name, setName] = useState(object.name);
  useEffect(() => setName(object.name), [object.name]);
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
          <NumberField key={i} value={v} disabled={locked} onCommit={(n) => setPos(i, n)} />
        ))}
      </div>
      <div className="compose-vec">
        <span>Dönüş °</span>
        {deg.map((v, i) => (
          <NumberField key={i} value={v} step={1} disabled={locked} onCommit={(n) => setRot(i, n)} />
        ))}
      </div>
      <div className="compose-vec">
        <span>Ölçek</span>
        <NumberField value={t.scale} disabled={locked} onCommit={(n) => n > 0 && onTransform({ ...t, scale: n })} />
      </div>
      {!locked && (
        <button type="button" className="btn-secondary" onClick={onSnap}>
          Zemine oturt
        </button>
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
