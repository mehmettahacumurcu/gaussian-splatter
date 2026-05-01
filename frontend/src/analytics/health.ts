import type { TrainMetric, JobMode } from "../api";
import { PRESET_BASELINES } from "../presets";

export type Verdict = "converging" | "stalled" | "diverging" | "insufficient";

export interface HealthReport {
  verdict: Verdict;
  windowSize: number;
  lossSlope: number;
  meanLoss: number;
  hasNaN: boolean;
  psnrCurrent?: number;
  psnrBaseline?: number;
  psnrDelta?: number;
}

function slope(x: number[], y: number[]): number {
  const n = x.length;
  if (n < 2) return 0;
  let sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (let i = 0; i < n; i++) {
    sx += x[i]; sy += y[i]; sxx += x[i] * x[i]; sxy += x[i] * y[i];
  }
  const denom = n * sxx - sx * sx;
  if (denom === 0) return 0;
  return (n * sxy - sx * sy) / denom;
}

export function computeHealth(
  metrics: TrainMetric[],
  mode: JobMode,
  preset?: string,
): HealthReport {
  if (metrics.length < 30) {
    return {
      verdict: "insufficient",
      windowSize: metrics.length,
      lossSlope: 0,
      meanLoss: 0,
      hasNaN: false,
    };
  }

  const windowSize = Math.max(30, Math.floor(metrics.length * 0.2));
  const recent = metrics.slice(-windowSize);
  const xs = recent.map((m) => m.iter);
  const ys = recent.map((m) => m.loss);

  const hasNaN = recent.some(
    (m) => !Number.isFinite(m.loss) || !Number.isFinite(m.psnr),
  );

  const lossSlope = slope(xs, ys);
  const meanLoss = ys.reduce((a, b) => a + b, 0) / ys.length;
  const eps = Math.max(1e-7, Math.abs(meanLoss) * 1e-4);

  let verdict: Verdict;
  if (hasNaN || lossSlope > eps) verdict = "diverging";
  else if (lossSlope < -eps) verdict = "converging";
  else verdict = "stalled";

  const last = metrics[metrics.length - 1];
  const psnrCurrent = Number.isFinite(last.psnr) ? last.psnr : undefined;
  const baseline = preset ? PRESET_BASELINES[mode]?.[preset]?.psnr : undefined;
  const psnrDelta = psnrCurrent !== undefined && baseline !== undefined
    ? psnrCurrent - baseline
    : undefined;

  return {
    verdict,
    windowSize,
    lossSlope,
    meanLoss,
    hasNaN,
    psnrCurrent,
    psnrBaseline: baseline,
    psnrDelta,
  };
}
