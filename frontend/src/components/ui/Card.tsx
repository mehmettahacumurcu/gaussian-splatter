import type { ReactNode } from "react";

interface Props {
  title?: ReactNode;
  actions?: ReactNode;
  tone?: "default" | "ok" | "warn" | "err";
  className?: string;
  children: ReactNode;
}

export function Card({ title, actions, tone = "default", className, children }: Props) {
  return (
    <section className={`ui-card ui-card-${tone} ${className ?? ""}`}>
      {(title || actions) && (
        <header className="ui-card-header">
          {title && <h3 className="ui-card-title">{title}</h3>}
          {actions && <div className="ui-card-actions">{actions}</div>}
        </header>
      )}
      <div className="ui-card-body">{children}</div>
    </section>
  );
}
