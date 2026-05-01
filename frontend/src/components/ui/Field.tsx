import type { ReactNode } from "react";
import { InfoButton } from "./InfoButton";

interface Props {
  label: ReactNode;
  help?: ReactNode;
  hintRight?: ReactNode;
  children: ReactNode;
}

export function Field({ label, help, hintRight, children }: Props) {
  return (
    <label className="ui-field">
      <div className="ui-field-row">
        <span className="ui-field-label">{label}</span>
        {help && <InfoButton>{help}</InfoButton>}
        {hintRight && <span className="ui-field-hint">{hintRight}</span>}
      </div>
      {children}
    </label>
  );
}
