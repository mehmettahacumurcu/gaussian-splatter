/**
 * JobsList — aktif + geçmiş job'ları listele, 2 sn'de bir auto-refresh.
 *
 * Her row:
 *   - scene name + status badge
 *   - progress bar (overall_progress)
 *   - phase + message (running ise)
 *   - "Viewer'da aç" butonu (completed ise)
 */
import { useEffect, useState } from "react";
import type { Job } from "../api";
import { downloadUrl, listJobs } from "../api";

interface Props {
  onViewJob: (job: Job) => void;
}

export function JobsList({ onViewJob }: Props) {
  const [jobs, setJobs] = useState<Job[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      try {
        const res = await listJobs();
        if (!cancelled) {
          setJobs(res.jobs);
          setError(null);
        }
      } catch (e) {
        if (!cancelled) setError(String(e));
      }
    };
    tick();
    const id = window.setInterval(tick, 2000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, []);

  if (error) {
    return (
      <div className="jobs-list-empty">
        <p className="hint hint-err">Jobs yüklenemedi: {error}</p>
        <p className="hint">Backend çalışıyor mu? (http://127.0.0.1:8000)</p>
      </div>
    );
  }

  if (jobs === null) {
    return <div className="jobs-list-empty"><p className="hint">Yükleniyor…</p></div>;
  }

  if (jobs.length === 0) {
    return (
      <div className="jobs-list-empty">
        <p className="hint">Hiç job yok.</p>
        <p className="hint">"Yeni Job" sekmesinden bir video gönder.</p>
      </div>
    );
  }

  return (
    <div className="jobs-list">
      {jobs.map((job) => (
        <JobRow
          key={job.id}
          job={job}
          expanded={expanded === job.id}
          onToggleExpand={() => setExpanded(expanded === job.id ? null : job.id)}
          onView={() => onViewJob(job)}
        />
      ))}
    </div>
  );
}

interface RowProps {
  job: Job;
  expanded: boolean;
  onToggleExpand: () => void;
  onView: () => void;
}

function JobRow({ job, expanded, onToggleExpand, onView }: RowProps) {
  const duration = (() => {
    const start = job.started_at ?? job.created_at;
    const end = job.finished_at ?? Date.now() / 1000;
    return end - start;
  })();

  return (
    <div className={`job-row job-${job.status}`}>
      <div className="job-row-top" onClick={onToggleExpand}>
        <div className="job-row-left">
          <span className={`job-badge job-badge-${job.status}`}>
            {job.status}
          </span>
          <span className="job-scene">{job.scene}</span>
          {job.smoke_test && <span className="job-tag">smoke</span>}
        </div>
        <div className="job-row-right">
          <span className="job-duration">
            {formatDuration(duration)}
          </span>
          <span className="job-chevron">{expanded ? "▾" : "▸"}</span>
        </div>
      </div>

      {/* Progress bar */}
      {(job.status === "running" || job.status === "queued") && (
        <div className="job-progress-row">
          <div className="progress-bar">
            <div
              className={`progress-fill progress-fill-${job.status}`}
              style={{ width: `${job.overall_progress * 100}%` }}
            />
          </div>
          <span className="progress-pct">
            %{(job.overall_progress * 100).toFixed(0)}
          </span>
        </div>
      )}

      {/* Phase + message */}
      {job.status === "running" && (
        <div className="job-phase-row">
          <span className="job-phase-name">{job.phase.name}</span>
          <span className="job-phase-msg">{job.phase.message}</span>
        </div>
      )}

      {/* Expand: details + actions */}
      {expanded && (
        <div className="job-expand">
          <div className="job-meta">
            <span>ID: <code>{job.id}</code></span>
            <span>Oluşturuldu: {new Date(job.created_at * 1000).toLocaleString()}</span>
            {job.finished_at && (
              <span>Bitiş: {new Date(job.finished_at * 1000).toLocaleString()}</span>
            )}
          </div>

          {job.error && (
            <details className="job-error">
              <summary>Hata</summary>
              <pre>{job.error}</pre>
            </details>
          )}

          {job.status === "completed" && (
            <div className="job-actions">
              <button className="btn-primary" onClick={onView}>
                Viewer'da aç
              </button>
              <a
                className="btn-secondary"
                href={downloadUrl(job.id)}
                target="_blank"
                rel="noreferrer"
              >
                .zip indir
              </a>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function formatDuration(sec: number): string {
  if (sec < 60) return `${sec.toFixed(0)}s`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m ${Math.floor(sec % 60)}s`;
  return `${Math.floor(sec / 3600)}h ${Math.floor((sec % 3600) / 60)}m`;
}
