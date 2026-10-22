/**
 * Kapsama: where did the capture under-observe the scene?
 *
 * Pick a training run folder (Spirula/Colab result synced from Drive, or a
 * local scene); the backend counts per splat how many cameras saw it and
 * writes a heatmap. The splat and its heatmap are shown side by side with one
 * camera, and the views that see the most weakly covered splats are listed so
 * the next video can target them.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getJobStatus } from "../api";
import {
  coveragePhotoUrl,
  coverageSplatUrl,
  deleteCoverage,
  getCoverage,
  listCoverage,
  startCoverage,
  type CoverageDetail,
  type CoverageMeta,
  type CoverageView,
} from "./coverageApi";
import { DualSplatViewer } from "./DualSplatViewer";
import "./coverage.css";

const POLL_MS = 1000;
const pct = (v: number) => `%${(v * 100).toFixed(1)}`;

type RunState =
  | { kind: "idle" }
  | { kind: "running"; jobId: string; coverageId: string; progress: number; message: string }
  | { kind: "error"; message: string };

function errorText(err: unknown): string {
  const text = err instanceof Error ? err.message : String(err);
  const detail = text.match(/"detail":\s*"([^"]+)"/);
  return detail ? detail[1] : text;
}

/** `prefill` (e.g. from a finished Spirula run) fills the scan form; a new object re-applies it. */
export interface CoveragePrefill { path: string; plyPath?: string; sparseDir?: string }

export function CoveragePage({ active, prefill }: { active: boolean; prefill?: CoveragePrefill | null }) {
  const [results, setResults] = useState<CoverageMeta[]>([]);
  const [selected, setSelected] = useState<CoverageDetail | null>(null);
  const [path, setPath] = useState("");
  const [plyPath, setPlyPath] = useState("");
  const [sparseDir, setSparseDir] = useState("");
  useEffect(() => {
    if (!prefill) return;
    setPath(prefill.path);
    setPlyPath(prefill.plyPath ?? "");
    setSparseDir(prefill.sparseDir ?? "");
  }, [prefill]);
  const [run, setRun] = useState<RunState>({ kind: "idle" });
  const [view, setView] = useState<CoverageView | null>(null);
  const [showTrain, setShowTrain] = useState(false);
  const [listError, setListError] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);

  const refresh = useCallback(async () => {
    try {
      setResults(await listCoverage());
      setListError(null);
    } catch (err) {
      setListError(errorText(err));
    }
  }, []);

  const open = useCallback(async (id: string) => {
    try {
      const detail = await getCoverage(id);
      setSelected(detail);
      setView(null);
    } catch (err) {
      setListError(errorText(err));
    }
  }, []);

  useEffect(() => {
    refresh();
    return () => {
      if (pollRef.current !== null) window.clearTimeout(pollRef.current);
    };
  }, [refresh]);

  const poll = useCallback((jobId: string, coverageId: string) => {
    pollRef.current = window.setTimeout(async () => {
      try {
        const job = await getJobStatus(jobId);
        if (job.status === "completed") {
          setRun({ kind: "idle" });
          await refresh();
          await open(coverageId);
          return;
        }
        if (job.status === "failed") {
          setRun({ kind: "error", message: (job.error ?? "İş başarısız").split("\n")[0] });
          return;
        }
        setRun({ kind: "running", jobId, coverageId, progress: job.overall_progress ?? 0, message: job.phase?.message ?? "" });
      } catch {
        // Backend restart or a dropped request: keep polling.
      }
      poll(jobId, coverageId);
    }, POLL_MS);
  }, [open, refresh]);

  const start = async () => {
    try {
      const res = await startCoverage({ path: path.trim(), ply_path: plyPath.trim(), sparse_dir: sparseDir.trim() });
      setRun({ kind: "running", jobId: res.job_id, coverageId: res.coverage_id, progress: 0, message: "Sırada" });
      poll(res.job_id, res.coverage_id);
    } catch (err) {
      setRun({ kind: "error", message: errorText(err) });
    }
  };

  const remove = async (id: string) => {
    if (!window.confirm("Bu kapsama sonucu silinsin mi? (Splat dosyasına dokunulmaz.)")) return;
    try {
      await deleteCoverage(id);
      if (selected?.id === id) setSelected(null);
      await refresh();
    } catch (err) {
      setListError(errorText(err));
    }
  };

  // Test views first when the run has them; otherwise the training views. Worst first.
  const views = useMemo(() => {
    if (!selected) return [];
    const hasTest = selected.views.some((v) => v.test);
    const pool = selected.views.filter((v) => (hasTest && !showTrain ? v.test : !v.test));
    return [...pool].sort((a, b) => b.weak - a.weak);
  }, [selected, showTrain]);

  useEffect(() => {
    if (!view && views.length) setView(views[0]);
  }, [views, view]);

  const step = (d: number) => {
    if (!views.length) return;
    const i = view ? views.indexOf(view) : -1;
    setView(views[Math.max(0, Math.min(views.length - 1, i + d))]);
  };

  const urls = useMemo(
    () => (selected ? { normal: coverageSplatUrl(selected.id, "normal"), heat: coverageSplatUrl(selected.id, "heat") } : null),
    [selected],
  );

  return (
    <div className="coverage-page">
      <aside className="coverage-side">
        <h3>Yeni tarama</h3>
        <label className="coverage-field">
          Koşu klasörü ya da splat .ply
          <input value={path} onChange={(e) => setPath(e.target.value)} placeholder="G:\Drive'ım\GaussianTests\…\ROOM2" />
        </label>
        <details>
          <summary>Elle belirt</summary>
          <label className="coverage-field">
            Splat .ply
            <input value={plyPath} onChange={(e) => setPlyPath(e.target.value)} />
          </label>
          <label className="coverage-field">
            COLMAP modeli (sparse/0)
            <input value={sparseDir} onChange={(e) => setSparseDir(e.target.value)} />
          </label>
        </details>
        <button
          type="button"
          className="btn-primary"
          disabled={run.kind === "running" || !(path.trim() || plyPath.trim())}
          onClick={start}
        >
          Kapsamayı hesapla
        </button>
        {run.kind === "running" && (
          <div className="coverage-progress" role="status">
            <progress max={1} value={run.progress} /> {Math.round(run.progress * 100)}% · {run.message}
          </div>
        )}
        {run.kind === "error" && <div className="coverage-error" role="alert">{run.message}</div>}
        <p className="muted">
          Splat + eğitimde kullanılan kameralar gerekir. Spirula koşu klasörünü (Drive'daki) seçmen yeter;
          splat ve sparse/0 kendiliğinden bulunur.
        </p>

        <h3>Sonuçlar</h3>
        {listError && <div className="coverage-error">{listError}</div>}
        {!results.length && <p className="muted">Henüz tarama yok.</p>}
        <ul className="coverage-results">
          {results.map((r) => (
            <li key={r.id} className={selected?.id === r.id ? "selected" : ""}>
              <button type="button" className="coverage-result-open" onClick={() => open(r.id)}>
                <strong>{r.name}</strong>
                <small>zayıf {pct(r.summary.weak_share)} · görülmemiş {pct(r.summary.unseen_share)}</small>
              </button>
              <button type="button" className="coverage-icon-btn" title="Sil" onClick={() => remove(r.id)}>✕</button>
            </li>
          ))}
        </ul>
      </aside>

      <section className="coverage-main">
        {selected && urls && active ? (
          <DualSplatViewer normalUrl={urls.normal} heatUrl={urls.heat} up={selected.up} view={view} />
        ) : (
          <div className="coverage-empty">
            Bir sonuç seç ya da yeni tarama başlat. Kırmızı: az kamera gördü (eksik çekim), yeşil: iyi kapsanmış.
          </div>
        )}
      </section>

      <aside className="coverage-side coverage-right">
        {selected ? (
          <>
            <h3>{selected.name}</h3>
            <dl className="coverage-stats">
              <dt>Splat</dt><dd>{selected.summary.opaque_splats.toLocaleString()} opak / {selected.summary.splats.toLocaleString()}</dd>
              <dt>Kamera</dt><dd>{selected.cameras - selected.test_views} eğitim{selected.test_views ? ` + ${selected.test_views} test` : ""}</dd>
              <dt>Zayıf</dt><dd>{pct(selected.summary.weak_share)} <small>(≤5 kamera)</small></dd>
              <dt>Hiç görülmemiş</dt><dd>{pct(selected.summary.unseen_share)}</dd>
              <dt>Medyan görüş</dt><dd>{selected.summary.count_percentiles["50"] ?? "-"}</dd>
            </dl>
            <p className="muted">
              Hiç çekilmemiş yerde splat da oluşmaz: haritada boş/siyah kalan yerler de eksik sayılır.
            </p>

            <div className="coverage-views-head">
              <h3>{selected.test_views && !showTrain ? "Test görüşleri" : "Eğitim görüşleri"} · en zayıftan</h3>
              {selected.test_views > 0 && (
                <label className="coverage-inline">
                  <input type="checkbox" checked={showTrain} onChange={(e) => { setShowTrain(e.target.checked); setView(null); }} />
                  eğitim
                </label>
              )}
            </div>
            <div className="coverage-step">
              <button type="button" onClick={() => step(-1)}>◀</button>
              <button type="button" onClick={() => step(1)}>▶</button>
              <span className="muted">WASD/oklar: yürü · sürükle: bak</span>
            </div>
            <ul className="coverage-views">
              {views.map((v) => (
                <li key={v.index}>
                  <button type="button" className={view === v ? "active" : ""} onClick={() => setView(v)}>
                    <span>{v.name.replace(/^.*[\\/]/, "")}</span>
                    <span className="coverage-bar" style={{ ["--w" as string]: `${Math.round(v.weak * 100)}%` }} />
                    <small>{pct(v.weak)}</small>
                  </button>
                </li>
              ))}
            </ul>
            {view && selected.images_dir && (
              <figure className="coverage-photo">
                <img src={coveragePhotoUrl(selected.id, view.index)} alt={view.name} />
                <figcaption>Çekilen kare: {view.name}</figcaption>
              </figure>
            )}
          </>
        ) : (
          <p className="muted">Sonuç seçilmedi.</p>
        )}
      </aside>
    </div>
  );
}
