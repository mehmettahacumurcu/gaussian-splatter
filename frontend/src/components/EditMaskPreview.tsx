import { useRef } from "react";
import { getApiBase } from "../connection";

interface Props {
  scene: string;
  frameIdx: number;
  onClick: (xNorm: number, yNorm: number) => void;
  maskUrl?: string;  // optional overlay returned from a SAM2 preview call
}

export function EditMaskPreview({ scene, frameIdx, onClick, maskUrl }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  return (
    <div
      ref={containerRef}
      style={{ position: "relative", display: "inline-block", cursor: "crosshair" }}
      onClick={(e) => {
        const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
        const x = (e.clientX - rect.left) / rect.width;
        const y = (e.clientY - rect.top) / rect.height;
        onClick(x, y);
      }}
    >
      <img
        src={`${getApiBase()}/scenes/${scene}/frame/${frameIdx}`}
        alt={`Frame ${frameIdx}`}
        style={{ display: "block", maxWidth: "100%", maxHeight: "70vh" }}
      />
      {maskUrl && (
        <img
          src={maskUrl}
          alt="mask preview"
          style={{
            position: "absolute", left: 0, top: 0,
            width: "100%", height: "100%",
            opacity: 0.5, mixBlendMode: "multiply",
            pointerEvents: "none",
          }}
        />
      )}
    </div>
  );
}
