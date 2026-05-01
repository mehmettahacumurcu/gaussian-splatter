import { useEffect, useRef, useState } from "react";

export type ViewerEngine = "legacy" | "spark";

interface Props {
  engine: ViewerEngine;
  onEngineChange: (e: ViewerEngine) => void;
  hudVisible: boolean;
  onHudToggle: (v: boolean) => void;
  singleFrameMode: boolean;
  onSingleFrameToggle: (v: boolean) => void;
}

export function ViewerSettings(props: Props) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    window.addEventListener("mousedown", onClick);
    return () => window.removeEventListener("mousedown", onClick);
  }, [open]);

  return (
    <div className="viewer-settings" ref={ref}>
      <button
        type="button"
        className="viewer-settings-btn"
        aria-label="Viewer settings"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
      >
        ⚙
      </button>
      {open && (
        <div className="viewer-settings-popover">
          <div className="viewer-settings-section">
            <div className="viewer-settings-label">Engine</div>
            <label className="viewer-settings-row">
              <input
                type="radio"
                checked={props.engine === "legacy"}
                onChange={() => props.onEngineChange("legacy")}
              />
              legacy (mkkellogg)
            </label>
            <label className="viewer-settings-row">
              <input
                type="radio"
                checked={props.engine === "spark"}
                onChange={() => props.onEngineChange("spark")}
              />
              Spark (4DGS) ✨
            </label>
          </div>
          <div className="viewer-settings-section">
            <div className="viewer-settings-label">Overlays</div>
            <label className="viewer-settings-row">
              <input
                type="checkbox"
                checked={props.hudVisible}
                onChange={(e) => props.onHudToggle(e.target.checked)}
              />
              Performance HUD
            </label>
          </div>
          <div className="viewer-settings-section">
            <div className="viewer-settings-label">Debug</div>
            <label className="viewer-settings-row">
              <input
                type="checkbox"
                checked={props.singleFrameMode}
                onChange={(e) => props.onSingleFrameToggle(e.target.checked)}
              />
              Single-frame mode
            </label>
          </div>
        </div>
      )}
    </div>
  );
}
