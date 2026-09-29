import { useRef, useState } from "react";
import type { Asset, ObjectKind } from "./types";

interface Props {
  assets: Asset[];
  kinds?: ObjectKind[];
  busy?: boolean;
  actionLabel: string;
  onPick: (asset: Asset) => void;
  onUpload: (file: File) => void;
}

function formatMb(bytes: number) {
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export function AssetPicker({ assets, kinds = ["splat", "mesh"], busy, actionLabel, onPick, onUpload }: Props) {
  const [assetId, setAssetId] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);
  const options = assets.filter((a) => kinds.includes(a.kind));
  const accept = kinds.map((k) => (k === "splat" ? ".ply" : ".glb")).join(",");
  const chosen = options.find((a) => a.id === assetId);

  return (
    <div className="compose-asset-picker">
      <select value={assetId} onChange={(e) => setAssetId(e.target.value)} disabled={busy}>
        <option value="">Asset seç…</option>
        {options.map((a) => (
          <option key={a.id} value={a.id}>
            {a.kind === "splat" ? "◉" : "▲"} {a.name} {a.source === "pipeline" ? "(pipeline)" : ""} · {formatMb(a.size_bytes)}
          </option>
        ))}
      </select>
      <div className="compose-row">
        <button type="button" className="btn-primary" disabled={!chosen || busy} onClick={() => chosen && onPick(chosen)}>
          {actionLabel}
        </button>
        <button type="button" className="btn-secondary" disabled={busy} onClick={() => fileRef.current?.click()}>
          Dosya yükle
        </button>
        <input
          ref={fileRef}
          type="file"
          accept={accept}
          hidden
          onChange={(e) => {
            const file = e.target.files?.[0];
            e.target.value = "";
            if (file) onUpload(file);
          }}
        />
      </div>
    </div>
  );
}
