import type { ReactNode } from "react";

type Tone = "info" | "ok" | "warn" | "err";

interface Props {
  tone?: Tone;
  title?: ReactNode;
  children?: ReactNode;
}

const ICON: Record<Tone, string> = { info: "ℹ", ok: "✓", warn: "⚠", err: "✗" };

export function Banner({ tone = "info", title, children }: Props) {
  return (
    <div className={`ui-banner ui-banner-${tone}`} role="status">
      <span className="ui-banner-icon" aria-hidden>{ICON[tone]}</span>
      <div className="ui-banner-body">
        {title && <div className="ui-banner-title">{title}</div>}
        {children && <div className="ui-banner-msg">{children}</div>}
      </div>
    </div>
  );
}
