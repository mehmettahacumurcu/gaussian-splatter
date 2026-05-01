/**
 * NvsEvalPanel — 4D Quality v6.1 NVS Evaluation görünümü.
 *
 * Held-out cam metrics (PSNR/SSIM/LPIPS) + smooth orbit mp4.
 * T10 — visual pass: Cards + StatPills.
 */
import { useEffect, useState } from "react";
import { getJobEval, orbitVideoUrl, type NvsEvalReport } from "../api";
import { Card } from "./ui/Card";
import { StatPill } from "./ui/StatPill";

interface Props {
  scene: string;
}

export function NvsEvalPanel({ scene }: Props) {
  const [report, setReport] = useState<NvsEvalReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const loadReport = async () => {
    if (!scene.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const r = await getJobEval(scene.trim());
      setReport(r);
    } catch (e) {
      setError(String(e));
      setReport(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (scene.trim()) loadReport();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scene]);

  if (!scene.trim()) {
    return <div style={{ padding: 20, color: "#888" }}>Sahne adı gir, NVS eval raporu yüklensin.</div>;
  }

  return (
    <div style={{ padding: 20 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
        <h2 style={{ margin: 0 }}>NVS Evaluation — {scene}</h2>
        <button className="btn-secondary" onClick={loadReport} disabled={loading}>
          {loading ? "Yükleniyor..." : "Yenile"}
        </button>
      </div>

      {error && <div className="hint hint-err" style={{ marginBottom: 12 }}>Hata: {error}</div>}

      {report && !report.available && (
        <Card title="NVS evaluation yok">
          <p className="hint">
            Bu sahne için çalıştırılmamış. {report.message && <span>({report.message})</span>}
          </p>
          <p className="hint">
            Submit ederken "NVS Evaluation" toggle'ını aç — training sonrası otomatik çalışır.
          </p>
        </Card>
      )}

      {report && report.available && (
        <>
          <Card
            title={
              report.held_out_cam
                ? `Held-out cam: ${report.held_out_cam}`
                : "Temporal hold-out (single-view)"
            }
          >
            {(() => {
              const m = report.held_out_metrics ?? report.temporal_holdout;
              if (!m) return <p className="hint">Metric yok.</p>;
              return (
                <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                  <StatPill label="PSNR" value={`${m.psnr.toFixed(2)} dB`} tone="accent" />
                  <StatPill label="SSIM" value={m.ssim.toFixed(4)} />
                  <StatPill label="LPIPS ↓" value={m.lpips.toFixed(4)} />
                  <StatPill label="Frames" value={String(m.n_frames)} />
                </div>
              );
            })()}
          </Card>

          {report.orbit?.success && report.orbit_url && (
            <Card title={`Smooth Orbit Camera (${report.orbit.n_frames} frame)`}>
              <video
                src={orbitVideoUrl(scene)}
                controls
                style={{ width: "100%", maxWidth: 800, borderRadius: 4 }}
              />
              <p className="hint" style={{ marginTop: 8 }}>
                Catmull-Rom spline ile train cam'lardan smooth path. Sahnenin "demo" videosu.
              </p>
            </Card>
          )}
          {report.orbit && !report.orbit.success && (
            <div className="hint hint-err">Orbit render başarısız: {report.orbit.error}</div>
          )}
        </>
      )}
    </div>
  );
}
