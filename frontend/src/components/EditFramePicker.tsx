import { useState } from "react";
import { getApiBase } from "../connection";

interface Props {
  scene: string;
  totalFrames: number;
  onSelect: (frameIdx: number) => void;
}

export function EditFramePicker({ scene, totalFrames, onSelect }: Props) {
  const [selected, setSelected] = useState<number | null>(null);
  const stride = Math.max(1, Math.floor(totalFrames / 12));
  const indices = Array.from({ length: 12 }, (_, i) => Math.min(i * stride, totalFrames - 1));

  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 8, padding: 12 }}>
      {indices.map((idx) => (
        <div
          key={idx}
          onClick={() => { setSelected(idx); onSelect(idx); }}
          style={{
            border: selected === idx ? "3px solid #4a8" : "1px solid #444",
            cursor: "pointer", padding: 4, borderRadius: 4,
          }}
        >
          <img
            src={`${getApiBase()}/scenes/${scene}/frame/${idx}`}
            alt={`Frame ${idx}`}
            style={{ width: 160, height: "auto", display: "block" }}
          />
          <div style={{ textAlign: "center", color: "#aaa", fontSize: 11 }}>#{idx}</div>
        </div>
      ))}
    </div>
  );
}
