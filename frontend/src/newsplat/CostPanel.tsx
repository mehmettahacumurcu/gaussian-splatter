import { formatCount, formatDuration, formatGb } from "./format";
import type { Estimate, EstimateResponse, GpuName } from "./spirulaApi";

interface Props {
  estimate: EstimateResponse | null;
  target: GpuName;
  onTarget(gpu: GpuName): void;
  frames: number | null;
}

/** Live VRAM / time estimate for the chosen GPU plus a compact comparison of every GPU. */
export function CostPanel({ estimate, target, onTarget, frames }: Props) {
  if (!estimate) {
    return <aside className="ns-cost" aria-label="Donanım maliyeti"><p className="ns-hint">Tahmin hesaplanıyor…</p></aside>;
  }
  const row = estimate.estimates.find((e) => e.gpu === target) ?? estimate.estimates[0];
  const reading = estimate.gpu.gpus[0];
  return (
    <aside className="ns-cost" aria-label="Donanım maliyeti">
      <div className="ns-cost-targets" role="radiogroup" aria-label="Hedef GPU">
        {estimate.estimates.map((e) => (
          <button key={e.gpu} type="button" role="radio" aria-checked={e.gpu === row.gpu}
            className={`${e.gpu === row.gpu ? "active" : ""} ${e.fits === false ? "nofit" : ""}`}
            onClick={() => onTarget(e.gpu)}>
            {e.gpu === "local" ? "Bu PC" : e.gpu}
          </button>
        ))}
      </div>

      <div className="ns-cost-main">
        <span className="ns-cost-caption">{row.gpu_label}</span>
        <span className="ns-cost-big" data-testid="vram-estimate">~{formatGb(row.vram_mib)} <small>VRAM</small></span>
        <VramBar row={row} usedByOthers={row.gpu === "local" ? reading?.used_mib ?? null : null}
          total={row.gpu === "local" ? reading?.total_mib ?? null : row.available_mib} />
        <FitLine row={row} />
      </div>

      <dl className="ns-cost-facts">
        <div><dt>Splatlar</dt><dd>{formatGb(row.vram_splats_mib)}</dd></div>
        <div><dt>Görüntü tamponları + Spirula</dt><dd>{formatGb(row.vram_base_mib)}</dd></div>
        <div><dt>Eğitim süresi</dt><dd>{formatDuration(row.train_seconds)}</dd></div>
        {frames ? <div><dt>Kare</dt><dd>~{frames}</dd></div> : null}
      </dl>

      <table className="ns-cost-table">
        <thead><tr><th>GPU</th><th>VRAM</th><th>Eğitim</th></tr></thead>
        <tbody>
          {estimate.estimates.map((e) => (
            <tr key={e.gpu} className={e.gpu === row.gpu ? "active" : ""} onClick={() => onTarget(e.gpu)}>
              <td>{e.gpu === "local" ? "Bu PC" : e.gpu}</td>
              <td className={e.fits === false ? "bad" : e.fits ? "good" : ""}>
                {e.available_mib != null ? `${formatGb(e.vram_mib)} / ${formatGb(e.available_mib)}` : formatGb(e.vram_mib)}
              </td>
              <td>{e.fits === false ? "sığmaz" : formatDuration(e.train_seconds)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="ns-hint">
        3060 Ti ölçümlerinden kaba tahmin; tepe bellek sahneye göre ±%30 oynar. Süre yalnız eğitim; SfM ve geometri hariç.
      </p>
    </aside>
  );
}

function VramBar({ row, usedByOthers, total }: { row: Estimate; usedByOthers: number | null; total: number | null }) {
  if (!total) return null;
  const pct = (mib: number) => `${Math.min(100, (mib / total) * 100)}%`;
  const others = usedByOthers ?? 0;
  return (
    <div className="ns-bar" role="img"
      aria-label={`${formatGb(row.vram_mib)} tahmin, ${formatGb(total)} toplam${others ? `, ${formatGb(others)} başka uygulamalarda` : ""}`}>
      {others > 0 && <span className="ns-bar-others" style={{ width: pct(others) }} title={`Başka uygulamalar: ${formatGb(others)}`} />}
      <span className={`ns-bar-run ${row.fits === false ? "bad" : ""}`} style={{ width: pct(row.vram_mib) }} />
    </div>
  );
}

function FitLine({ row }: { row: Estimate }) {
  if (row.fits == null) return <p className="ns-hint">GPU okunamadı (nvidia-smi yok).</p>;
  if (row.fits) {
    const spare = (row.available_mib ?? 0) - row.vram_mib;
    return <p className="ns-fit good">Sığar: ~{formatGb(spare)} boş kalır</p>;
  }
  if (!row.max_splats || row.max_splats < 100_000) {
    return (
      <p className="ns-fit bad">
        Sığmaz: şu an yalnız {formatGb(Math.max(0, row.available_mib ?? 0))} boş.
        {row.gpu === "local" ? " GPU kullanan uygulamaları kapat ya da Colab'i seç." : ""}
      </p>
    );
  }
  return <p className="ns-fit bad">Sığmaz: bu GPU'da en fazla ~{formatCount(row.max_splats)} splat</p>;
}
