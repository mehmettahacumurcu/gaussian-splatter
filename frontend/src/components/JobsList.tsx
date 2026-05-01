/**
 * JobsList — aktif + geçmiş job'ları listele, 2 sn'de bir auto-refresh.
 *
 * T7: pipeline filter chips (current / all / static / dynamic), mode chip per row.
 */
import { useEffect, useMemo, useState } from "react";
import type { Job, JobMode } from "../api";
import { listJobs } from "../api";

interface Props {
  pipeline: JobMode;
  onViewJob: (job: Job) => void;
}

type Filter = "current" | "all" | "static" | "dynamic";

export function JobsList({ pipeline, onViewJob }: Props) {
  const [jobs, setJobs] = useState<Job[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [filter, setFilter] = useState<Filter>("current");

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
    return () => { cancelled = true; window.clearInterval(id); };
  }, []);

  const filtered = useMemo(() => {
    if (!jobs) return null;
    const target =
      filter === "current" ? pipeline :
      filter === "all" ? null :
      filter;
    if (target === null) return jobs;
    return jobs.filter((j) => (j.mode ?? "dynamic") === target);
  }, [jobs, filter, pipeline]);

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

  return (
    <>
      <div className="jobs-filter-row">
        <FilterChip
          label={pipeline === "static" ? "📸 Static (current)" : "🎬 Dynamic (current)"}
          active={filter === "current"}
          onClick={() => setFilter("current")}
        />
        <FilterChip label="All" active={filter === "all"} onClick={() => setFilter("all")} />
        <FilterChip label="📸 Static" active={filter === "static"} onClick={() => setFilter("static")} />
        <FilterChip label="🎬 Dynamic" active={filter === "dynamic"} onClick={() => setFilter("dynamic")} />
      </div>
      {filtered && filtered.length === 0 ? (
        <div className="jobs-list-empty">
          <p className="hint">Bu filtrede iş yok.</p>
          <p className="hint">"Submit" sekmesinden bir job başlat.</p>
        </div>
      ) : (
        <div className="jobs-list">
          {filtered!.map((job) => (
            <JobRow
              key={job.id}
              job={job}
              expanded={expanded === job.id}
              onToggleExpand={() => setExpanded(expanded === job.id ? null : job.id)}
              onView={() => onViewJob(job)}
            />
          ))}
        </div>
      )}
    </>
  );
}

function FilterChip({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      type="button"
      className={`jobs-filter-chip ${active ? "active" : ""}`}
      onClick={onClick}
    >
      {label}
    </button>
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
  const mode = job.mode ?? "dynamic";
  return (
    <div className={`job-row job-${job.status}`}>
      <div className="job-row-top" onClick={onToggleExpand}>
        <div className="job-row-left">
          <span className={`job-badge job-badge-${job.status}`}>{job.status}</span>
          <span className={`job-mode-chip job-mode-${mode}`}>
            {mode === "static" ? "📸 static" : "🎬 dynamic"}
          </span>
          <span className="job-scene">{job.scene}</span>
          {job.smoke_test && <span className="job-tag">smoke</span>}
        </div>
        <div className="job-row-right">
          <span className="job-duration">{formatDuration(duration)}</span>
          <span className="job-chevron">{expanded ? "▾" : "▸"}</span>
        </div>
      </div>
      {(job.status === "running" || job.status === "queued") && (
        <div className="job-progress-row">
          <div className="progress-bar">
            <div
              className={`progress-fill progress-fill-${job.status}`}
              style={{ width: `${job.overall_progress * 100}%` }}
            />
          </div>
          <span className="progress-pct">%{(job.overall_progress * 100).toFixed(0)}</span>
        </div>
      )}
      {job.status === "running" && (
        <div className="job-phase-row">
          <span className="job-phase-name">{job.phase.name}</span>
          <span className="job-phase-msg">{job.phase.message}</span>
        </div>
      )}
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
              <button className="btn-primary" onClick={onView}>Viewer'da aç</button>
              <a
                className="btn-secondary"
                href={`http://127.0.0.1:8000/download/${job.id}`}
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
