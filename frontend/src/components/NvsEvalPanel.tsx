/**
 * NvsEvalPanel — 4D Quality v6.1 NVS Evaluation görünümü.
 *
 * Held-out cam metrics (PSNR/SSIM/LPIPS) + smooth orbit mp4 göstergesi.
 * Backend GET /jobs/{id}/eval'dan rapor çeker, GET /jobs/{id}/orbit.mp4 ile video.
 */
import { useEffect, useState } from "react";
import { getJobEval, orbitVideoUrl, type NvsEvalReport } from "../api";

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
    return (
      <div style={{ padding: 20, color: "#888" }}>
        Sahne adi gir, NVS eval raporu yuklensin.
      </div>
    );
  }

  return (
    <div style={{ padding: 20 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
        <h2 style={{ margin: 0 }}>NVS Evaluation — {scene}</h2>
        <button className="btn-secondary" onClick={loadReport} disabled={loading}>
          {loading ? "Yukleniyor..." : "Yenile"}
        </button>
      </div>

      {error && (
        <div className="hint hint-err" style={{ marginBottom: 12 }}>
          Hata: {error}
        </div>
      )}

      {report && !report.available && (
        <div className="hint">
          NVS evaluation bu sahne icin calistirilmamis.{" "}
          {report.message && <span>({report.message})</span>}
          <p style={{ marginTop: 8, fontSize: 12 }}>
            Job submit ederken "NVS Evaluation" toggle'ini acmali, training sonrasi
            otomatik calisir.
          </p>
        </div>
      )}

      {report && report.available && (
        <>
          {/* Metrics card */}
          <div
            style={{
              background: "var(--bg-1)",
              border: "1px solid var(--border)",
              borderRadius: 6,
              padding: 16,
              marginBottom: 16,
            }}
          >
            <h3 style={{ margin: "0 0 10px", fontSize: 14 }}>
              {report.held_out_cam
                ? `Held-out cam: ${report.held_out_cam}`
                : "Temporal hold-out (single-view)"}
            </h3>
            {(() => {
              const m = report.held_out_metrics ?? report.temporal_holdout;
              if (!m) return <p className="hint">Metric yok.</p>;
              return (
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(4, 1fr)",
                    gap: 12,
                  }}
                >
                  <Metric label="PSNR" value={m.psnr.toFixed(2)} unit="dB" highlight />
                  <Metric label="SSIM" value={m.ssim.toFixed(4)} />
                  <Metric label="LPIPS" value={m.lpips.toFixed(4)} lower="↓" />
                  <Metric label="Frames" value={String(m.n_frames)} />
                </div>
              );
            })()}
          </div>

          {/* Orbit video */}
          {report.orbit?.success && report.orbit_url && (
            <div
              style={{
                background: "var(--bg-1)",
                border: "1px solid var(--border)",
                borderRadius: 6,
                padding: 16,
              }}
            >
              <h3 style={{ margin: "0 0 10px", fontSize: 14 }}>
                Smooth Orbit Camera ({report.orbit.n_frames} frame)
              </h3>
              <video
                src={orbitVideoUrl(scene)}
                controls
                style={{ width: "100%", maxWidth: 800, borderRadius: 4 }}
              />
              <p className="hint" style={{ marginTop: 8 }}>
                Catmull-Rom spline ile train cam'lardan smooth path. Sahnenin "demo" videosu.
              </p>
            </div>
          )}
          {report.orbit && !report.orbit.success && (
            <div className="hint hint-err">
              Orbit render basarisiz: {report.orbit.error}
            </div>
          )}
        </>
      )}
    </div>
  );
}

function Metric({
  label,
  value,
  unit,
  highlight,
  lower,
}: {
  label: string;
  value: string;
  unit?: string;
  highlight?: boolean;
  lower?: string;
}) {
  return (
    <div>
      <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
        {label} {lower && <span title="lower is better">{lower}</span>}
      </div>
      <div
        style={{
          fontSize: 20,
          fontWeight: 600,
          color: highlight ? "var(--accent)" : "var(--text)",
        }}
      >
        {value}
        {unit && <span style={{ fontSize: 12, marginLeft: 4 }}>{unit}</span>}
      </div>
    </div>
  );
}
