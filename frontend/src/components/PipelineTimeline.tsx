import type { PhaseSegment } from "../analytics/events";
import { Card } from "./ui/Card";

interface Props {
  segments: PhaseSegment[];
}

const PHASE_COLOR: Record<string, string> = {
  foundation: "#a8dadc",
  colmap: "#ffe66d",
  init: "#dda0dd",
  train: "#4ecdc4",
  training: "#4ecdc4",
  eval: "#95e1d3",
};

export function PipelineTimeline({ segments }: Props) {
  if (segments.length === 0) {
    return (
      <Card title="Pipeline timeline">
        <div className="hint">Timeline unavailable for this run (no phase events).</div>
      </Card>
    );
  }
  const w = 100 / segments.length;
  return (
    <Card title="Pipeline timeline">
      <div className="pl-timeline">
        {segments.map((s, i) => (
          <div
            key={i}
            className="pl-timeline-seg"
            style={{
              width: `${w}%`,
              background: PHASE_COLOR[s.phase] ?? "#888",
            }}
            title={`${s.phase}${s.startTs ? ` · ${s.startTs}` : ""}`}
          >
            <span className="pl-timeline-label">{s.phase}</span>
          </div>
        ))}
      </div>
    </Card>
  );
}
