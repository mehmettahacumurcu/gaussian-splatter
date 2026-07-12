import type { DriveFolderResolution } from "./drivePath";

interface Props {
  value: string;
  resolution: DriveFolderResolution;
  onChange(value: string): void;
}

export function DriveFolderSection({ value, resolution, onChange }: Props) {
  return (
    <section className={`nb-stage ${resolution.ok ? "is-complete" : ""}`}>
      <div className="nb-stage-dot" aria-hidden="true" />
      <div className="nb-stage-body">
        <p className="nb-stage-kicker">01 / source</p>
        <h2>Google Drive input folder</h2>
        <label className="nb-field nb-path-field">
          <span>Drive folder</span>
          <input
            type="text"
            value={value}
            onChange={(event) => onChange(event.target.value)}
            placeholder="captures/myroom"
            aria-describedby="drive-folder-help"
          />
        </label>
        <p id="drive-folder-help" className="nb-help">
          Name an existing folder below MyDrive. This is not a local file path.
        </p>
        {resolution.ok ? (
          <div className="nb-path-readout">
            <span>Input</span><code>{resolution.displayInput}</code>
            <span>Output</span><code>{resolution.displayOutput}</code>
          </div>
        ) : value ? (
          <p className="nb-error" role="alert">{resolution.error}</p>
        ) : null}
      </div>
    </section>
  );
}
