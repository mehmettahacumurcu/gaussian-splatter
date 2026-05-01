import { useId, useState, type ReactNode } from "react";

interface Props {
  label?: string;
  children: ReactNode;
  initialOpen?: boolean;
}

export function InfoButton({ label = "More info", children, initialOpen = false }: Props) {
  const [open, setOpen] = useState(initialOpen);
  const id = useId();
  return (
    <>
      <button
        type="button"
        className={`ui-info-btn ${open ? "open" : ""}`}
        aria-expanded={open}
        aria-controls={id}
        aria-label={label}
        onClick={() => setOpen((o) => !o)}
      >
        <span aria-hidden>ⓘ</span>
      </button>
      <div
        id={id}
        role="region"
        className={`ui-info-details ${open ? "open" : ""}`}
        hidden={!open}
      >
        {children}
      </div>
    </>
  );
}
