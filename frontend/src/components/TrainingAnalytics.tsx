/**
 * TrainingAnalytics — tek sahne için training metriklerini görselleştirir.
 *
 * Kaynak: GET /jobs/<scene>/metrics → metrics.jsonl parse edilmiş JSON array
 *         GET /jobs/<scene>/summary → final summary.json
 *         GET /jobs/<scene>/events  → events.log
 *
 * Her chart pure SVG — recharts / chart.js dependency yok.
 * Auto-refresh: 3 saniyede bir metrics poll eder (job hala training'deyse).
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  getJobEvents,
  getJobMetrics,
  getJobSummary,
  type TrainMetric,
} from "../api";

interface Props {
  scene: string;            // scene adı veya job id
  autoRefresh?: boolean;    // true = 3 sn'de bir poll
}

type ChartConfig = {
  title: string;
  yKey: keyof TrainMetric | ((m: TrainMetric) => number);
  color: string;
  yLog?: boolean;
  yFormat?: (v: number) => string;
};

const CHARTS: ChartConfig[] = [
  {
    title: "Total loss",
    yKey: "loss",
    color: "#ff6b6b",
    yFormat: (v) => v.toFixed(4),
  },
  {
    title: "PSNR",
    yKey: "psnr",
    color: "#4ecdc4",
    yFormat: (v) => v.toFixed(2) + " dB",
  },
  {
    title: "N (gaussian count)",
    yKey: "n_points",
    color: "#ffe66d",
    yFormat: (v) => v.toLocaleString(),
  },
  {
    title: "Δpos mean (motion magnitude)",
    yKey: "dpos_mean",
    color: "#95e1d3",
    yFormat: (v) => v.toFixed(4),
  },
  {
    title: "Loss components",
    yKey: (m) => m.recon, // overridden in multi-series
    color: "#a8dadc",
  },
  {
    title: "it/s",
    yKey: "it_per_sec",
    color: "#c77dff",
    yFormat: (v) => v.toFixed(1),
  },
];

/**
 * Simple SVG line chart.
 */
function LineChart({
  data,
  yValues,
  color = "#4ecdc4",
  yLabel = "",
  width = 480,
  height = 180,
  yLog = false,
  multiSeries,
}: {
  data: TrainMetric[];
  yValues?: number[];
  color?: string;
  yLabel?: string;
  width?: number;
  height?: number;
  yLog?: boolean;
  multiSeries?: { label: string; values: number[]; color: string }[];
}) {
  if (data.length === 0) {
    return <div style={{ padding: 20, color: "#888" }}>Veri yok</div>;
  }

  const padL = 50, padR = 15, padT = 10, padB = 25;
  const W = width - padL - padR;
  const H = height - padT - padB;

  const xs = data.map((d) => d.iter);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);

  const allSeries =
    multiSeries ??
    (yValues ? [{ label: yLabel, values: yValues, color }] : []);

  const allY = allSeries.flatMap((s) => s.values.filter((v) => Number.isFinite(v)));
  if (allY.length === 0) return <div style={{ padding: 20, color: "#888" }}>Veri yok</div>;

  let yMin = Math.min(...allY);
  let yMax = Math.max(...allY);
  if (yLog) {
    yMin = Math.log10(Math.max(yMin, 1e-6));
    yMax = Math.log10(Math.max(yMax, 1e-5));
  }
  const yRange = yMax - yMin || 1;

  const xNorm = (x: number) => padL + ((x - xMin) / Math.max(xMax - xMin, 1)) * W;
  const yNorm = (y: number) => {
    const ly = yLog ? Math.log10(Math.max(y, 1e-6)) : y;
    return padT + (1 - (ly - yMin) / yRange) * H;
  };

  // Axis ticks
  const xTickCount = 5;
  const yTickCount = 4;
  const xTicks = Array.from({ length: xTickCount + 1 }, (_, i) =>
    xMin + ((xMax - xMin) * i) / xTickCount,
  );
  const yTicks = Array.from({ length: yTickCount + 1 }, (_, i) => {
    const v = yMin + (yRange * i) / yTickCount;
    return yLog ? Math.pow(10, v) : v;
  });

  return (
    <svg width={width} height={height} style={{ background: "#1a1a1a", borderRadius: 4 }}>
      {/* Grid + y-axis labels */}
      {yTicks.map((v, i) => {
        const ly = yLog ? Math.log10(Math.max(v, 1e-6)) : v;
        const y = padT + (1 - (ly - yMin) / yRange) * H;
        return (
          <g key={`y${i}`}>
            <line x1={padL} y1={y} x2={width - padR} y2={y} stroke="#2a2a2a" strokeWidth={1} />
            <text x={padL - 5} y={y + 4} fill="#888" fontSize="10" textAnchor="end" fontFamily="monospace">
              {v > 1000 ? (v / 1000).toFixed(1) + "k" : v.toFixed(v < 1 ? 3 : 1)}
            </text>
          </g>
        );
      })}
      {/* X-axis labels */}
      {xTicks.map((v, i) => {
        const x = xNorm(v);
        return (
          <text
            key={`x${i}`}
            x={x}
            y={height - 8}
            fill="#888"
            fontSize="10"
            textAnchor="middle"
            fontFamily="monospace"
          >
            {Math.round(v).toLocaleString()}
          </text>
        );
      })}
      {/* Lines */}
      {allSeries.map((s, si) => {
        const path = s.values
          .map((v, i) => {
            if (!Number.isFinite(v)) return null;
            const cmd = i === 0 ? "M" : "L";
            return `${cmd}${xNorm(xs[i])},${yNorm(v)}`;
          })
          .filter(Boolean)
          .join(" ");
        return (
          <path
            key={si}
            d={path}
            stroke={s.color}
            strokeWidth="1.5"
            fill="none"
          />
        );
      })}
      {/* Legend */}
      {multiSeries && (
        <g>
          {multiSeries.map((s, i) => (
            <g key={i} transform={`translate(${padL + 5},${padT + 5 + i * 14})`}>
              <rect width="10" height="3" y="4" fill={s.color} />
              <text x="14" y="9" fill="#ddd" fontSize="10" fontFamily="monospace">
                {s.label}
              </text>
            </g>
          ))}
        </g>
      )}
    </svg>
  );
}

function StatCard({ label, value, hint }: { label: string; value: string | number; hint?: string }) {
  return (
    <div
      style={{
        background: "#232323",
        border: "1px solid #2a2a2a",
        borderRadius: 6,
        padding: "10px 14px",
        minWidth: 120,
      }}
    >
      <div style={{ color: "#888", fontSize: 11, textTransform: "uppercase", letterSpacing: 0.5 }}>
        {label}
      </div>
      <div style={{ color: "#eee", fontSize: 18, fontWeight: 600, fontFamily: "monospace" }}>
        {value}
      </div>
      {hint && <div style={{ color: "#666", fontSize: 10, marginTop: 2 }}>{hint}</div>}
    </div>
  );
}

export function TrainingAnalytics({ scene, autoRefresh = true }: Props) {
  const [metrics, setMetrics] = useState<TrainMetric[]>([]);
  const [summary, setSummary] = useState<Record<string, unknown> | null>(null);
  const [events, setEvents] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const load = useCallback(async () => {
    if (!scene) return;
    setLoading(true);
    try {
      const [m, e] = await Promise.all([
        getJobMetrics(scene),
        getJobEvents(scene, 100),
      ]);
      setMetrics(m.metrics);
      setEvents(e.events);
      // Summary opsiyonel (sadece bitti sonrası)
      try {
        const s = await getJobSummary(scene);
        setSummary(s);
      } catch {
        setSummary(null);
      }
      setErr(null);
    } catch (e) {
      setErr(String(e));
    } finally {
      setLoading(false);
    }
  }, [scene]);

  useEffect(() => {
    load();
    if (!autoRefresh) return;
    const id = window.setInterval(load, 3000);
    return () => window.clearInterval(id);
  }, [load, autoRefresh]);

  // Last metric snapshot
  const last = metrics[metrics.length - 1];
  const progress = last && last.n_iters > 0 ? (last.iter / last.n_iters) * 100 : 0;

  // Extract y-series
  const lossComps = useMemo(() => {
    if (metrics.length === 0) return null;
    return [
      { label: "recon", values: metrics.map((m) => m.recon), color: "#a8dadc" },
      { label: "depth", values: metrics.map((m) => m.depth), color: "#ffa07a" },
      { label: "track", values: metrics.map((m) => m.track), color: "#98fb98" },
      { label: "smooth", values: metrics.map((m) => m.smooth), color: "#dda0dd" },
      { label: "rigid", values: metrics.map((m) => m.rigid), color: "#f0e68c" },
      { label: "scale", values: metrics.map((m) => m.scale), color: "#ff69b4" },
    ];
  }, [metrics]);

  if (!scene) {
    return <div style={{ padding: 20, color: "#888" }}>Bir sahne seç.</div>;
  }

  return (
    <div style={{ padding: "10px 20px", color: "#eee" }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
        <h3 style={{ margin: 0 }}>Training Analytics</h3>
        <span style={{ color: "#888", fontSize: 12 }}>{scene}</span>
        {loading && <span style={{ color: "#ffe66d", fontSize: 11 }}>refreshing…</span>}
        {autoRefresh && !loading && (
          <span style={{ color: "#444", fontSize: 10 }}>auto-refresh 3s</span>
        )}
        <button
          onClick={load}
          style={{ marginLeft: "auto", padding: "4px 10px", fontSize: 11 }}
          className="btn-secondary"
        >
          Refresh
        </button>
      </div>

      {err && (
        <div style={{ color: "#ff6b6b", marginBottom: 10, fontSize: 12 }}>Hata: {err}</div>
      )}

      {/* Stat cards */}
      {last && (
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 16 }}>
          <StatCard
            label="Progress"
            value={`${last.iter.toLocaleString()} / ${last.n_iters.toLocaleString()}`}
            hint={`${progress.toFixed(1)}%`}
          />
          <StatCard label="Loss" value={last.loss.toFixed(4)} />
          <StatCard label="PSNR" value={`${last.psnr.toFixed(2)} dB`} />
          <StatCard label="N points" value={last.n_points.toLocaleString()} />
          <StatCard
            label="Δpos mean"
            value={last.dpos_mean.toFixed(4)}
            hint={`max ${last.dpos_max.toFixed(3)}`}
          />
          <StatCard
            label="Speed"
            value={`${last.it_per_sec.toFixed(1)} it/s`}
            hint={`${(last.t / 60).toFixed(1)} dk geçti`}
          />
          <StatCard label="Warmup" value={`${(last.warmup * 100).toFixed(0)}%`} />
        </div>
      )}

      {/* Charts grid */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(500px, 1fr))",
          gap: 14,
          marginBottom: 20,
        }}
      >
        <div>
          <div style={{ fontSize: 13, color: "#ccc", marginBottom: 4 }}>Total Loss</div>
          <LineChart
            data={metrics}
            yValues={metrics.map((m) => m.loss)}
            color="#ff6b6b"
            yLog
          />
        </div>
        <div>
          <div style={{ fontSize: 13, color: "#ccc", marginBottom: 4 }}>PSNR</div>
          <LineChart data={metrics} yValues={metrics.map((m) => m.psnr)} color="#4ecdc4" />
        </div>
        <div>
          <div style={{ fontSize: 13, color: "#ccc", marginBottom: 4 }}>
            Gaussian count (N)
          </div>
          <LineChart
            data={metrics}
            yValues={metrics.map((m) => m.n_points)}
            color="#ffe66d"
          />
        </div>
        <div>
          <div style={{ fontSize: 13, color: "#ccc", marginBottom: 4 }}>
            Δpos mean (motion magnitude)
          </div>
          <LineChart
            data={metrics}
            yValues={metrics.map((m) => m.dpos_mean)}
            color="#95e1d3"
          />
        </div>
        {lossComps && (
          <div style={{ gridColumn: "1 / -1" }}>
            <div style={{ fontSize: 13, color: "#ccc", marginBottom: 4 }}>
              Loss components (log-scale)
            </div>
            <LineChart
              data={metrics}
              multiSeries={lossComps}
              yLog
              width={1000}
              height={220}
            />
          </div>
        )}
      </div>

      {/* Events log */}
      <div>
        <div style={{ fontSize: 13, color: "#ccc", marginBottom: 4 }}>
          Events (son {events.length})
        </div>
        <pre
          style={{
            background: "#1a1a1a",
            border: "1px solid #2a2a2a",
            padding: 10,
            fontSize: 11,
            fontFamily: "monospace",
            color: "#bbb",
            maxHeight: 240,
            overflow: "auto",
            borderRadius: 4,
          }}
        >
          {events.join("\n") || "(henüz event yok)"}
        </pre>
      </div>

      {/* Summary (run bittiyse) */}
      {summary && (
        <div style={{ marginTop: 16 }}>
          <div style={{ fontSize: 13, color: "#ccc", marginBottom: 4 }}>Run Summary</div>
          <pre
            style={{
              background: "#1a1a1a",
              border: "1px solid #2a2a2a",
              padding: 10,
              fontSize: 11,
              fontFamily: "monospace",
              color: "#bbb",
              maxHeight: 300,
              overflow: "auto",
              borderRadius: 4,
            }}
          >
            {JSON.stringify(summary, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}
