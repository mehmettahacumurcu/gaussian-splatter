import type { ReactNode } from "react";

interface Props {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  tone?: "default" | "accent" | "ok" | "warn" | "err";
}

export function StatPill({ label, value, hint, tone = "default" }: Props) {
  return (
    <div className={`ui-stat ui-stat-${tone}`}>
      <div className="ui-stat-label">{label}</div>
      <div className="ui-stat-value">{value}</div>
      {hint && <div className="ui-stat-hint">{hint}</div>}
    </div>
  );
}
