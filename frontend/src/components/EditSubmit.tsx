import { useState } from "react";
import { submitEditJob } from "../api";
import { EditFramePicker } from "./EditFramePicker";
import { EditMaskPreview } from "./EditMaskPreview";
import { EditQualityDialog } from "./EditQualityDialog";

type State = "idle" | "selecting_frame" | "click_target" | "quality" | "submitted";

interface Props {
  scene: string;
  sourceCkpt: string;
  totalFrames: number;
  onSubmitted: (jobId: string) => void;
}

export function EditSubmit({ scene, sourceCkpt, totalFrames, onSubmitted }: Props) {
  const [state, setState] = useState<State>("idle");
  const [frameIdx, setFrameIdx] = useState<number | null>(null);
  const [click, setClick] = useState<[number, number] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleClick = (x: number, y: number) => {
    setClick([x, y]);
    setState("quality");
  };

  const handleQualitySelect = async (mode: "A" | "B") => {
    if (frameIdx === null || click === null) return;
    try {
      const res = await submitEditJob({
        scene, source_ckpt: sourceCkpt,
        frame_idx: frameIdx,
        click_x: click[0], click_y: click[1],
        quality_mode: mode,
      });
      onSubmitted(res.job_id);
      setState("submitted");
    } catch (e) {
      setError(String(e));
    }
  };

  return (
    <div className="submit-panel">
      <div className="submit-panel-mode-banner" style={{ background: "#1a2a1a", borderColor: "#4a8" }}>
        Obje Silme — Sahne'deki bir nesneyi sec ve kaldir. Kaynak checkpoint'ten yeni bir PLY olusturulur.
      </div>

      <h2 className="submit-title">Obje Silme</h2>
      <p>Sahne: <code>{scene}</code> · Kaynak: <code>{sourceCkpt}</code></p>

      {state === "idle" && (
        <div className="submit-actions">
          <button className="btn-primary submit-btn" onClick={() => setState("selecting_frame")}>
            Silmeye basla
          </button>
        </div>
      )}

      {state === "selecting_frame" && (
        <div className="submit-section">
          <h4>Nesnenin gorundugu frame'i sec</h4>
          <EditFramePicker
            scene={scene}
            totalFrames={totalFrames}
            onSelect={(idx) => { setFrameIdx(idx); setState("click_target"); }}
          />
        </div>
      )}

      {state === "click_target" && frameIdx !== null && (
        <div className="submit-section">
          <h4>Silinecek nesneye tikla (frame #{frameIdx})</h4>
          <p className="submit-hint">Gorsel uzerinde nesneye tiklayarak koordinat gonder.</p>
          <EditMaskPreview
            scene={scene}
            frameIdx={frameIdx}
            onClick={handleClick}
          />
          <button
            className="btn-secondary"
            style={{ marginTop: 8 }}
            onClick={() => setState("selecting_frame")}
          >
            Geri — baska frame sec
          </button>
        </div>
      )}

      {state === "quality" && (
        <EditQualityDialog
          onSelect={handleQualitySelect}
          onCancel={() => setState("click_target")}
        />
      )}

      {state === "submitted" && (
        <div className="submit-section">
          <p>Job gonderildi. Ilerlemeyi Jobs tabinda takip et.</p>
          <button className="btn-secondary" onClick={() => { setState("idle"); setFrameIdx(null); setClick(null); setError(null); }}>
            Yeni silme islemi
          </button>
        </div>
      )}

      {error && (
        <div className="submit-error" role="alert">
          <strong>Hata:</strong> {error}
        </div>
      )}
    </div>
  );
}
