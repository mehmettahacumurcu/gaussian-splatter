import { useEffect, useState } from "react";
import type { GeneratedNotebook } from "./types";

interface Props {
  artifact: GeneratedNotebook;
  onRegenerate(): void;
  onEdit(): void;
}

function download(url: string, filename: string) {
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
}

export function NotebookDownload({ artifact, onRegenerate, onEdit }: Props) {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    const next = URL.createObjectURL(artifact.blob);
    setUrl(next);
    download(next, artifact.filename);
    return () => URL.revokeObjectURL(next);
  }, [artifact]);
  return (
    <aside className="nb-download" aria-live="polite">
      <p className="nb-download-label">Notebook ready</p>
      <strong>{artifact.filename}</strong>
      <p>Connect a Colab GPU runtime, choose Run all, and grant Drive access.</p>
      <div className="nb-download-actions">
        <button type="button" onClick={() => url && download(url, artifact.filename)}>Download again</button>
        <button type="button" onClick={onRegenerate}>Regenerate</button>
        <button type="button" onClick={onEdit}>Edit settings</button>
      </div>
    </aside>
  );
}
