/**
 * İşler — every job on the backend, with a detail pane for the selected one:
 * phase steps, training stats, Spirula's live viewer while training, the run log and result actions.
 */
import { useEffect, useRef, useState } from "react";
import { cancelJob, listJobs, type Job } from "../api";
import { getApiBase, withTokenParam } from "../connection";
import type { CoveragePrefill } from "../coverage/CoveragePage";
import { getRunLog, type RunLog } from "../newsplat/spirulaApi";
import {
  KIND_LABEL, SPIRULA_STEPS, STATUS_LABEL, elapsedSeconds, formatElapsed, jobKind, jobTitle, spirulaRunId,
} from "./jobKind";
import "./jobs.css";

const POLL_MS = 2000;
const num = (n: number) => n.toLocaleString("tr-TR");

interface Props {
  active?: boolean;
  selectedId: string | null;
  onSelect(id: string | null): void;
  onViewJob(job: Job): void;
  onCoverage(prefill: CoveragePrefill): void;
  onNewSplat(): void;
}

export function JobsPage({ active = true, selectedId, onSelect, onViewJob, onCoverage, onNewSplat }: Props) {
  const [jobs, setJobs] = useState<Job[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    const tick = () => listJobs().then(
      (res) => { if (!cancelled) { setJobs(res.jobs); setError(null); } },
      (e) => { if (!cancelled) setError(String(e)); },
    );
    void tick();
    const id = window.setInterval(tick, POLL_MS);
    return () => { cancelled = true; window.clearInterval(id); };
  }, [active]);

  const selected = jobs?.find((j) => j.id === selectedId) ?? jobs?.[0] ?? null;

  if (error && !jobs) {
    return <div className="jb-empty"><p className="ns-err">İşler yüklenemedi: {error}</p><p className="ns-hint">Backend çalışıyor mu?</p></div>;
  }
  if (!jobs) return <div className="jb-empty"><p className="ns-hint">Yükleniyor…</p></div>;
  if (jobs.length === 0) {
    return (
      <div className="jb-empty">
        <p>Henüz iş yok.</p>
        <button type="button" className="ns-btn ns-btn-primary" onClick={onNewSplat}>Yeni splat</button>
      </div>
    );
  }

  return (
    <div className="jb-page">
      <ul className="jb-list" aria-label="İşler">
        {jobs.map((job) => (
          <li key={job.id}>
            <button type="button" className={`jb-item ${job.id === selected?.id ? "active" : ""}`}
              aria-current={job.id === selected?.id} onClick={() => onSelect(job.id)}>
              <span className="jb-item-top">
                <span className="jb-item-title">{jobTitle(job)}</span>
                <StatusPill job={job} />
              </span>
              <span className="jb-item-sub">{KIND_LABEL[jobKind(job)]} · {formatElapsed(elapsedSeconds(job))}</span>
              {(job.status === "running" || job.status === "queued") && (
                <span className="jb-mini-bar"><span style={{ width: `${job.overall_progress * 100}%` }} /></span>
              )}
            </button>
          </li>
        ))}
      </ul>
      {selected && <JobDetail key={selected.id} job={selected} active={active} onViewJob={onViewJob} onCoverage={onCoverage} />}
    </div>
  );
}

function JobDetail({ job, active, onViewJob, onCoverage }: { job: Job; active: boolean } & Pick<Props, "onViewJob" | "onCoverage">) {
  const kind = jobKind(job);
  const runId = spirulaRunId(job);
  const running = job.status === "running" || job.status === "queued";
  type TrainStats = { step?: number; total?: number; splats?: number; psnr?: number; eta?: string };
  const result = (job.result ?? {}) as { splat_path?: string; workspace?: string; train?: TrainStats; sfm?: { selected_reconstruction?: string; registered_images?: number; input_images?: number } };
  // Finished runs keep their last training numbers in the result.
  const details = { ...(job.status === "completed" ? result.train : {}), ...job.phase.details } as TrainStats & { viewer_url?: string };
  const [cancelling, setCancelling] = useState(false);
  const [message, error] = splitError(job.error);
  const cancelled = isCancelled(job);

  return (
    <section className="jb-detail" aria-label={jobTitle(job)}>
      <header className="jb-head">
        <div>
          <h2>{jobTitle(job)}</h2>
          <p className="ns-hint">{KIND_LABEL[kind]} · {new Date(job.created_at * 1000).toLocaleString()} · {formatElapsed(elapsedSeconds(job))}</p>
        </div>
        <StatusPill job={job} />
      </header>

      {kind === "spirula" && <Steps job={job} />}

      {running && (
        <div className="jb-progress">
          <div className="jb-bar"><span style={{ width: `${job.overall_progress * 100}%` }} /></div>
          <div className="jb-progress-line">
            <span>%{Math.round(job.overall_progress * 100)}</span>
            <span className="jb-msg">{job.cancel_requested ? "İptal ediliyor…" : job.phase.message}</span>
            <button type="button" className="ns-btn jb-cancel" disabled={cancelling || job.cancel_requested}
              onClick={async () => {
                if (!window.confirm(`"${jobTitle(job)}" iptal edilsin mi?`)) return;
                setCancelling(true);
                try { await cancelJob(job.id); } finally { setCancelling(false); }
              }}>
              İptal
            </button>
          </div>
        </div>
      )}

      {details.step != null && details.total != null && (
        <dl className="jb-stats">
          <div><dt>Adım</dt><dd>{num(details.step)} / {num(details.total)}</dd></div>
          {details.splats != null && <div><dt>Splat</dt><dd>{num(details.splats)}</dd></div>}
          {details.psnr != null && <div><dt>PSNR</dt><dd>{details.psnr.toFixed(2)} dB</dd></div>}
          {running && details.eta && <div><dt>Kalan</dt><dd>{details.eta}</dd></div>}
        </dl>
      )}

      {kind === "spirula" && running && (
        <LivePreview url={job.phase.name === "spirula_train" ? details.viewer_url : undefined} />
      )}

      {cancelled && <p className="ns-hint">İptal edildi.</p>}

      {job.status === "failed" && !cancelled && (
        <div className="jb-error" role="alert">
          <p>{message}</p>
          {error && <details><summary>Teknik ayrıntı</summary><pre>{error}</pre></details>}
        </div>
      )}

      {job.status === "completed" && (
        <div className="jb-actions">
          {kind !== "compose" && kind !== "coverage" && (
            <button type="button" className="ns-btn ns-btn-primary" onClick={() => onViewJob(job)}>Viewer'da aç</button>
          )}
          {kind === "spirula" && result.workspace && (
            <button type="button" className="ns-btn" onClick={() => onCoverage({
              path: result.workspace!, plyPath: result.splat_path, sparseDir: result.sfm?.selected_reconstruction,
            })}>Kapsama analizi</button>
          )}
          {job.download_url && (
            <a className="ns-btn" href={withTokenParam(`${getApiBase()}${job.download_url}`)} target="_blank" rel="noreferrer">
              {kind === "spirula" ? "PLY indir" : "İndir"}
            </a>
          )}
          {result.sfm?.registered_images != null && (
            <span className="ns-hint">SfM: {result.sfm.registered_images}/{result.sfm.input_images} kare kayıtlı</span>
          )}
        </div>
      )}

      {runId && <RunLogView runId={runId} live={active && running} />}
    </section>
  );
}

function isCancelled(job: Job): boolean {
  return job.status === "failed" && !!job.cancel_requested;
}

function StatusPill({ job }: { job: Job }) {
  return isCancelled(job)
    ? <span className="jb-pill">İptal edildi</span>
    : <span className={`jb-pill jb-${job.status}`}>{STATUS_LABEL[job.status]}</span>;
}

function Steps({ job }: { job: Job }) {
  const current = SPIRULA_STEPS.findIndex((s) => s.phase === job.phase.name);
  const done = job.status === "completed";
  return (
    <ol className="jb-steps">
      {SPIRULA_STEPS.map((s, i) => {
        const state = done || i < current ? "done" : i === current ? (job.status !== "failed" ? "now" : isCancelled(job) ? "todo" : "failed") : "todo";
        return <li key={s.phase} className={`jb-step jb-step-${state}`}>{s.label}</li>;
      })}
    </ol>
  );
}

function LivePreview({ url }: { url?: string }) {
  if (!url) {
    return <div className="jb-live jb-live-wait"><p className="ns-hint">Canlı önizleme eğitim başlayınca burada açılır.</p></div>;
  }
  return (
    <div className="jb-live">
      <div className="jb-live-bar">
        <span>Canlı eğitim · Spirula görüntüleyici</span>
        <a href={url} target="_blank" rel="noreferrer">Ayrı pencerede aç</a>
      </div>
      <iframe src={url} title="Spirula canlı eğitim önizlemesi" />
    </div>
  );
}

function RunLogView({ runId, live }: { runId: string; live: boolean }) {
  const [log, setLog] = useState<RunLog | null>(null);
  const pre = useRef<HTMLPreElement>(null);
  // Follow the tail while running, unless the user scrolled up to read.
  useEffect(() => {
    const el = pre.current;
    if (el && live && el.scrollHeight - el.scrollTop - el.clientHeight < 80) el.scrollTop = el.scrollHeight;
  }, [log, live]);
  useEffect(() => {
    let cancelled = false;
    const tick = () => getRunLog(runId, 120).then((l) => { if (!cancelled) setLog(l); }, () => {});
    void tick();
    if (!live) return () => { cancelled = true; };
    const id = window.setInterval(tick, POLL_MS);
    return () => { cancelled = true; window.clearInterval(id); };
  }, [runId, live]);
  if (!log?.label) return null;
  return (
    <details className="jb-log" open={live}>
      <summary>Log · {log.label}.log</summary>
      <pre ref={pre}>{log.lines.join("\n")}</pre>
    </details>
  );
}

/** Backend errors are "<message>\n\n<traceback>"; show the message, fold the rest. */
function splitError(raw: string | null): [string, string] {
  if (!raw) return ["", ""];
  const i = raw.indexOf("\n\nTraceback");
  return i < 0 ? [raw, ""] : [raw.slice(0, i), raw.slice(i + 2)];
}
