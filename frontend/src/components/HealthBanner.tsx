import { Banner } from "./ui/Banner";
import type { HealthReport } from "../analytics/health";

interface Props {
  report: HealthReport;
}

const VERDICT_TONE: Record<HealthReport["verdict"], "ok" | "warn" | "err" | "info"> = {
  converging: "ok",
  stalled: "warn",
  diverging: "err",
  insufficient: "info",
};

const VERDICT_TITLE: Record<HealthReport["verdict"], string> = {
  converging: "Converging",
  stalled: "Stalled",
  diverging: "Diverging",
  insufficient: "Insufficient data",
};

export function HealthBanner({ report }: Props) {
  return (
    <Banner tone={VERDICT_TONE[report.verdict]} title={VERDICT_TITLE[report.verdict]}>
      <div>
        Loss slope <code>{report.lossSlope.toExponential(2)}</code> per iter
        {" · "}window {report.windowSize}
        {report.hasNaN && <span> · contains NaN</span>}
      </div>
      {report.psnrCurrent !== undefined && (
        <div>
          PSNR {report.psnrCurrent.toFixed(2)} dB
          {report.psnrBaseline !== undefined && report.psnrDelta !== undefined && (
            <> (preset baseline ≈ {report.psnrBaseline} —{" "}
              <strong>{report.psnrDelta >= 0 ? "+" : ""}{report.psnrDelta.toFixed(2)}</strong>)
            </>
          )}
        </div>
      )}
    </Banner>
  );
}
