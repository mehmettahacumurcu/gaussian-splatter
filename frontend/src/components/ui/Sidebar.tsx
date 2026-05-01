import { useState, type ReactNode } from "react";

interface Props {
  side?: "right" | "left";
  defaultOpen?: boolean;
  width?: number;
  children: ReactNode;
}

export function Sidebar({ side = "right", defaultOpen = true, width = 300, children }: Props) {
  const [open, setOpen] = useState(defaultOpen);
  const collapsedWidth = 18;
  return (
    <aside
      className={`ui-sidebar ui-sidebar-${side} ${open ? "open" : "collapsed"}`}
      style={{ width: open ? width : collapsedWidth }}
    >
      <button
        type="button"
        className="ui-sidebar-toggle"
        aria-label={open ? "Collapse sidebar" : "Expand sidebar"}
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
      >
        {open ? (side === "right" ? "▶" : "◀") : (side === "right" ? "◀" : "▶")}
      </button>
      {open && <div className="ui-sidebar-body">{children}</div>}
    </aside>
  );
}
