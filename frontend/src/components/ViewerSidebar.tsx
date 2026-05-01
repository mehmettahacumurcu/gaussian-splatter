import { useEffect, useState } from "react";
import { Sidebar } from "./ui/Sidebar";
import { StatPill } from "./ui/StatPill";
import { getJobSummary, type SplatInfo } from "../api";

interface Props {
  info: SplatInfo;
}

export function ViewerSidebar({ info }: Props) {
  const [summary, setSummary] = useState<Record<string, unknown> | null>(null);
  const [loadingSummary, setLoadingSummary] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setSummary(null);
    setLoadingSummary(true);
    getJobSummary(info.scene)
      .then((s) => { if (!cancelled) setSummary(s as Record<string, unknown>); })
      .catch(() => { /* summary may not exist (still training); ignore */ })
      .finally(() => { if (!cancelled) setLoadingSummary(false); });
    return () => { cancelled = true; };
  }, [info.scene]);

  const wide = typeof window !== "undefined" && window.innerWidth >= 1280;

  const mode = (summary?.mode as string | undefined) ?? "—";
  const preset = (summary?.preset as string | undefined) ?? "—";
  const psnr = summary?.final_psnr as number | undefined;
  const trainingDuration = summary?.training_duration_sec as number | undefined;
  const overrides = (summary?.hyperparam_overrides as Record<string, unknown> | undefined) ?? {};
  const overrideEntries = Object.entries(overrides).filter(([, v]) => v !== null && v !== undefined);

  return (
    <Sidebar side="right" defaultOpen={wide} width={300}>
      <div className="vsidebar">
        <div className="vsidebar-title">{info.scene}</div>
        <div className="vsidebar-mode">
          <span className={`job-mode-chip job-mode-${mode === "static" ? "static" : "dynamic"}`}>
            {mode === "static" ? "📸 static" : mode === "dynamic" ? "🎬 dynamic" : mode}
          </span>
          {preset !== "—" && <span className="vsidebar-preset">preset: {preset}</span>}
        </div>

        <div className="vsidebar-stats">
          <StatPill label="Frames" value={info.num_frames} />
          <StatPill label="Size" value={formatBytes(info.total_size_bytes)} />
          {psnr !== undefined && <StatPill label="PSNR" value={`${psnr.toFixed(2)} dB`} tone="accent" />}
          {trainingDuration !== undefined && (
            <StatPill label="Training" value={formatDuration(trainingDuration)} />
          )}
        </div>

        {overrideEntries.length > 0 && (
          <div className="vsidebar-section">
            <div className="vsidebar-section-title">Hyperparam overrides</div>
            <ul className="vsidebar-kv">
              {overrideEntries.slice(0, 8).map(([k, v]) => (
                <li key={k}>
                  <span className="vsidebar-k">{k}</span>
                  <span className="vsidebar-v">{String(v)}</span>
                </li>
              ))}
              {overrideEntries.length > 8 && (
                <li className="vsidebar-more">+{overrideEntries.length - 8} more</li>
              )}
            </ul>
          </div>
        )}

        <div className="vsidebar-section">
          <div className="vsidebar-section-title">Files</div>
          <a
            className="btn-secondary"
            href={`http://127.0.0.1:8000/download/${info.job_id}`}
            target="_blank"
            rel="noreferrer"
          >
            .zip indir ({info.num_frames} ply)
          </a>
        </div>

        {loadingSummary && <div className="hint">Summary yükleniyor…</div>}
        {!loadingSummary && summary === null && (
          <div className="hint">Summary yok (training devam ediyor olabilir).</div>
        )}
      </div>
    </Sidebar>
  );
}

function formatBytes(b: number): string {
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  if (b < 1024 * 1024 * 1024) return `${(b / 1024 / 1024).toFixed(1)} MB`;
  return `${(b / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

function formatDuration(sec: number): string {
  if (sec < 60) return `${sec.toFixed(0)}s`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m`;
  return `${Math.floor(sec / 3600)}h ${Math.floor((sec % 3600) / 60)}m`;
}
